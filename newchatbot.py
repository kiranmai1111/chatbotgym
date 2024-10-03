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


# Function to read PDF file
def read_pdf(pdf):
    text = ""
    for file in pdf:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Document Chunking (limit chunks to speed up)
def get_chunks(text, max_chunks=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks[:max_chunks]  # Limit the number of chunks


# Create Embedding Store with progress bar
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    progress_bar = st.progress(0)
    vector_store = FAISS()

    for i, chunk in enumerate(text_chunks):
        vector_store.add_text([chunk], embedding=embeddings)
        progress_bar.progress((i + 1) / len(text_chunks))  # Update progress bar

    vector_store.save_local("faiss_index")
    return vector_store


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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    load_vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = load_vector_db.similarity_search(user_query)
    chain = get_conversation_chain_pdf()
    response = chain.run(input_documents=docs, question=user_query)
    st.write(response)


def main():
    st.header("Welcome to Mind and Muscle, Ask Anything")

    # Example: Load a PDF file programmatically
    pdf_file_path = './Welcome to Mind and Muscle.pdf'

    # Check if FAISS index already exists to skip embedding
    if not os.path.exists("faiss_index"):
        with open(pdf_file_path, 'rb') as f:
            pdf_file = BytesIO(f.read())
            pdf_file.name = os.path.basename(pdf_file_path)  # Set a name for the file

            # Read and process the PDF
            start_time = time.time()
            raw_text = read_pdf([pdf_file])
            text_chunks = get_chunks(raw_text)
            st.write(f"PDF reading and chunking completed in {time.time() - start_time:.2f} seconds")

            # Create embeddings and store in FAISS
            start_time = time.time()
            get_vector_store(text_chunks)
            st.write(f"Embeddings and FAISS index creation took {time.time() - start_time:.2f} seconds")
    else:
        st.write("FAISS index already exists, skipping embedding generation.")

    # User input for query
    user_query = st.text_input("Drop your Question")
    if user_query:
        user_input(user_query)


if __name__ == "__main__":
    main()
