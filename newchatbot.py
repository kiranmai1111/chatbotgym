import streamlit as st
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO

# Load the API Key
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyCtyGp4yXkmsy06LmyXDUh6dpcnxO00bsc"))

# Function to read PDF file using pdfplumber
def read_pdf(pdf):
    text = ""
    for file in pdf:
        with pdfplumber.open(file) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ''  # Handle cases where text extraction might fail
    return text

# Document Chunking
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)  # Adjust chunk size for small PDFs
    chunks = text_splitter.split_text(text)
    return chunks  # No limit on chunks for a single line

# Create Embedding Store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS()
    for chunk in text_chunks:
        vector_store.add_text([chunk], embedding=embeddings)
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
    pdf_file_path = './kiranmai.pdf'

    # Check if FAISS index already exists to skip embedding
    if not os.path.exists("faiss_index"):
        with open(pdf_file_path, 'rb') as f:
            pdf_file = BytesIO(f.read())
            pdf_file.name = os.path.basename(pdf_file_path)  # Set a name for the file

            # Read and process the PDF
            raw_text = read_pdf([pdf_file])
            st.write(f"Raw text extracted: {raw_text}")  # Display the extracted text for verification
            text_chunks = get_chunks(raw_text)

            # Create embeddings and store in FAISS
            get_vector_store(text_chunks)
            st.write("Embeddings and FAISS index creation completed.")
    else:
        st.write("FAISS index already exists, skipping embedding generation.")

    # User input for query
    user_query = st.text_input("Drop your Question")
    if user_query:
        user_input(user_query)

if __name__ == "__main__":
    main()
