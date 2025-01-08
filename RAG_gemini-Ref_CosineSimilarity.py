import streamlit as st
import pdfplumber
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import psycopg2
import numpy as np
import re
import ast
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity

# Carrega as variáveis de ambiente
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Faz a conexão com o banco de dados PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    return conn

# Função para criar a tabela no banco de dados + extensão pgvector (se ainda não existir)
def create_table_if_not_exists():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;  
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT,
            content TEXT,
            embedding vector(768) 
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Função para extrair texto do PDF usando pdfplumber
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Função para dividir o texto em pedaços de frases inteiras
def get_text_chunks(text):
    sentence_endings = re.compile(r'([.,!,?])')  # identifica os caracteres de terminação de frases
    sentences = sentence_endings.split(text)
    chunks = []  # Armazena os pedaços finais do texto 
    temp_sentence = ""   # Variável temporária para montar frases completas.

    for sentence in sentences:
        temp_sentence += sentence
        if sentence.endswith(('.', '!', '?')):
            temp_sentence = temp_sentence.strip()
            if len(temp_sentence.split()) >= 20:  # Verifica se a frase tem pelo menos 20 palavras. Se sim, adiciona ao chunks.
                chunks.append(temp_sentence)
            temp_sentence = ""
    # Se, ao final da iteração, ainda houver texto, verifica se contém pelo menos 5 palavras. Se sim, adiciona o texto restante a chunks.
    if temp_sentence and len(temp_sentence.split()) >= 5:  
        chunks.append(temp_sentence.strip())
    return chunks

# Função para inserir o vetor no PostgreSQL
def insert_vector_into_db(embedding, content, title):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO documents (title, content, embedding)
        VALUES (%s, %s, %s)
    """, (title, content, embedding))

    conn.commit()
    cursor.close()
    conn.close()

# Função para buscar documentos semelhantes no banco de dados (usa distância de cosseno)
def search_similar_documents(query_vector, top_k=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, title, content, embedding FROM documents;")
    results = cursor.fetchall()  # Obtém todos os resultados da consulta.
    cursor.close()
    conn.close()

    # Converte os embeddings armazenados como string para arrays numpy
    embeddings = np.array([np.array(ast.literal_eval(result[3])) for result in results])

    # Calcula a similaridade de cosseno entre o vetor de consulta e os embeddings no banco de dados
    similarities = cosine_similarity([query_vector], embeddings)[0]

    # Ordena os documentos pela similaridade em ordem decrescente e seleciona os top_k mais similares
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Retorna os documentos mais semelhantes
    return [Document(page_content=results[i][2], metadata={"title": results[i][1], "embedding": embeddings[i]}) for i in top_indices]

# Função para criar o modelo de resposta
def get_conversational_chain():
    prompt_template = """
    Você é um chatbot que responde perguntas sobre normas de trabalhos acadêmicos. Responda à pergunta de forma clara a partir do contexto fornecido.
    Certifique-se de fornecer todos os detalhes importantes.
    Se houver contexto em inglês, traduza e adicione a resposta.
    Se a resposta não estiver no contexto fornecido, basta dizer "A resposta não está disponível no contexto, aguarde atualizações!"\n\n
    
    Contexto:\n {context}?\n
    Pergunta: \n{question}\n

    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Função para calcular a similaridade de cosseno entre duas respostas
def cosine_similarity_score(response_model, response_expected, embeddings):
    embedding_model = embeddings.embed_query(response_model)
    embedding_expected = embeddings.embed_query(response_expected)
    
    return cosine_similarity([embedding_model], [embedding_expected])[0][0]

# Função para lidar com a entrada do usuário
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    query_vector = embeddings.embed_query(user_question)
    docs = search_similar_documents(query_vector)

    if not docs:
        st.write("Não foram encontradas informações relevantes no banco de dados.")
        return

    # A resposta esperada será a mais relevante dos documentos retornados
    # Selecionamos o documento com maior similaridade
    top_doc = docs[0]
    expected_answer = top_doc.page_content  # A resposta esperada vem do conteúdo do top documento

    # Configura o modelo de resposta
    chain = get_conversational_chain()

    # Passa os documentos e a pergunta para o modelo
    response = chain(
        {"input_documents": docs, "question": user_question}, 
        return_only_outputs=True
    )

    # Exibe a resposta
    generated_answer = response["output_text"]
    st.write("Resposta: ", generated_answer)

    # Coleta fontes únicas
    unique_titles = set(doc.metadata['title'] for doc in docs)
    
    # Exibe as fontes
    st.write("Fontes:")
    for title in unique_titles:
        st.write(f"- {title}")
    
    # MÉTRICA DE AVALIAÇÃO    
    # Calcula a similaridade entre a resposta gerada e a resposta esperada
    similarity = cosine_similarity_score(generated_answer, expected_answer, embeddings)
    st.write(f"AVALIAÇÃO:")
    st.write(f"A similaridade entre a resposta gerada e a esperada é: {similarity}")
    if similarity > 0.8:
        st.success("Excelente! Alta similaridade.")
    elif similarity > 0.5:
        st.warning("Similaridade moderada. Pode ser melhor.")
    else:
        st.error("Baixa similaridade. Revise o modelo ou a resposta.")
    if not generated_answer or not expected_answer:
        st.error("Respostas não fornecidas. Verifique os dados de entrada.")

# Função principal do Streamlit
def main():
    st.set_page_config("ChatPDF")
    st.header("Olá, eu sou a Norma! Sua assistente de trabalhos acadêmicos.💁‍♀️")

    create_table_if_not_exists()

    user_question = st.text_input("Digite sua pergunta e pressione 'enter':")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📄Chat PDF Acadêmico")
        pdf_docs = st.file_uploader("Carregue seus arquivos PDF com normas de trabalhos acadêmicos e clique no botão 'Enviar'",
                                    accept_multiple_files=True)
        if st.button("Enviar"):
            with st.spinner("Processando..."):
                for pdf in pdf_docs:
                    title = pdf.name  # Nome do arquivo como título
                    with pdfplumber.open(pdf) as pdf_reader:
                        raw_text = ""
                        for page in pdf_reader.pages:
                            raw_text += page.extract_text()

                    text_chunks = get_text_chunks(raw_text)
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                    for chunk in text_chunks:
                        vector = embeddings.embed_documents([chunk])
                        insert_vector_into_db(vector[0], chunk, title)

                    full_document_embedding = embeddings.embed_documents([raw_text])
                    insert_vector_into_db(full_document_embedding[0], raw_text, title)

                st.success("Finalizado!")

# Executa o Streamlit
if __name__ == "__main__":
    main()
