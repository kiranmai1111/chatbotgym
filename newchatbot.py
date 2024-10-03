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

# Small test text to simulate faster processing
def small_test_text():
    return "This is a small test document to simulate a quick response for embeddings and FAISS index creation."

# Function to read PDF file
def read_pdf(pdf):
    text = ""
    for file in pdf:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Document Chunking with aggressive limits for faster testing
def get_chunks(text, max_chunks=5):  # Limit to 5 chunks for fast processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks[:max_chunks]  # Limit the number of chunks

# Create Embedding Store with detailed logging and progress bar
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    progress_bar = st.progress(0)
    vector_store = FAISS()

    for i, chunk in enumerate(text_chunks):
        st.write(f"Processing chunk {i+1}/{len(text_chunks)}...")  # Log which chunk is being processed
        vector_store.add_text([chunk], embedding=embeddings)
        progress_bar.progress((i + 1) / len(text_chunks))  # Update progress bar
    
    st.write("Saving FAISS index...")
    vector_store.save_local("faiss_index")
    st.write("FAISS index saved.")
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

    # Use small text instead of large PDF for faster initial testing
    raw_text = small_test_text()  # This will simulate a smaller input
    text_chunks = get_chunks(raw_text)  # Chunk the text (limited to 5 chunks)

    # Create embeddings and store in FAISS
    if not os.path.exists("faiss_index"):
        start_time = time.time()
        get_vector_store(text_chunks)  # Generate vector store and save FAISS index
        st.write(f"Embeddings and FAISS index creation took {time.time() - start_time:.2f} seconds")
    else:
        st.write("FAISS index already exists, skipping embedding generation.")

    # User input for query
    user_query = st.text_input("Drop your Question")
    if user_query:
        user_input(user_query)

if __name__ == "__main__":
    main()
