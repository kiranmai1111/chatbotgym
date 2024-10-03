import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO
import time

# Load the API Key
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyCtyGp4yXkmsy06LmyXDUh6dpcnxO00bsc"))

# Function to read and chunk PDF file (with caching)
@st.cache_data
def read_and_chunk_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf_file = BytesIO(f.read())
        pdf_reader = PdfReader(pdf_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

# Create Embedding Store (cached)
@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Load vector store (cached)
@st.cache_resource
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings)

# Create Conversation Chain
def get_conversation_chain_pdf():
    prompt_template = """
    Your role is to be a meticulous researcher. 

    Context: \n{context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Processing User Input
def user_input(user_query):
    vector_store = load_vector_store()  # Load cached vector store
    docs = vector_store.similarity_search(user_query)
    chain = get_conversation_chain_pdf()
    response = chain.run(input_documents=docs, question=user_query)
    st.write(response)

# Main function
def main():
    st.header("Welcome to Mind and Muscle, Ask Anything")

    pdf_file_path = './Welcome to Mind and Muscle.pdf'
    text_chunks = read_and_chunk_pdf(pdf_file_path)  # Cached text processing
    get_vector_store(text_chunks)  # Preprocess embeddings (cached)

    user_query = st.text_input("Drop your Question")
    if user_query:
        start_time = time.time()
        user_input(user_query)
        st.write(f"Response time: {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
