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
from langchain.docstore.document import Document

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
    sentence_endings = re.compile(r'([.,!,?])')  # pontuação para delimitar finais de frase
    sentences = sentence_endings.split(text)  # Divide o texto nas sentenças

    chunks = []
    temp_sentence = ""
    for sentence in sentences:
        temp_sentence += sentence
        if sentence.endswith(('.', '!', '?')):
            temp_sentence = temp_sentence.strip()
            if len(temp_sentence.split()) >= 20:  # mínimo de palavras se não tiver ponto
                chunks.append(temp_sentence)
            temp_sentence = ""
    
    if temp_sentence and len(temp_sentence.split()) >= 10:
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

# Função para buscar documentos semelhantes no banco de dados
def search_similar_documents(query_vector, top_k=10):
    conn = get_db_connection()
    cursor = conn.cursor()

    query_vector_pg = f'[{", ".join(map(str, query_vector))}]'

    cursor.execute("""
        SELECT id, title, content, embedding
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s;
    """, (query_vector_pg, top_k))

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    # Transforma os resultados em objetos do tipo Document com metadados
    return [Document(page_content=result[2], metadata={"title": result[1], "embedding": np.array(result[3])}) for result in results]

# Função para criar o modelo de resposta
def get_conversational_chain():
    prompt_template = """
    Você é um chatbot que responde perguntas sobre normas de trabalhos acadêmicos. Responda à pergunta o mais detalhadamente possível 
    a partir do contexto fornecido, certifique-se de fornecer todos os detalhes.
    Se houver contexto em inglês, traduza e adicione a resposta.
    Se a resposta não estiver no contexto fornecido, basta dizer "a resposta não está disponível no contexto"\n\n
    
    Contexto:\n {context}?\n
    Pergunta: \n{question}\n

    Resposta:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Função para lidar com a entrada do usuário
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    query_vector = embeddings.embed_query(user_question)
    docs = search_similar_documents(query_vector)

    if not docs:
        st.write("Não foram encontradas informações relevantes no banco de dados.")
        return

    # Configura o modelo de resposta
    chain = get_conversational_chain()

    # Passa os documentos e a pergunta para o modelo
    response = chain(
        {"input_documents": docs, "question": user_question},  # Aqui usamos "input_documents"
        return_only_outputs=True
    )

    # Exibe a resposta
    st.write("Resposta: ", response["output_text"])

    # Coleta fontes únicas
    unique_titles = set(doc.metadata['title'] for doc in docs)
    
    # Exibe as fontes
    st.write("Fontes:")
    for title in unique_titles:
        st.write(f"- {title}")

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
