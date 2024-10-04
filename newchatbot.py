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
import base64

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


# Document Chunking
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Create Embedding Store
def get_vector_store(text_chunks):
    st.write("Starting vector store creation...")
    
    # Optimize chunk size if needed
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.write("Embeddings created.")

    # Check how many chunks are being processed
    st.write(f"Number of chunks: {len(text_chunks)}")
    
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        st.write("Vector store created successfully.")
        
        # Save the vector store
        vector_store.save_local("faiss_index")
        st.write("Vector store saved successfully.")
    except Exception as e:
        st.write(f"Error creating vector store: {e}")

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
    # Set up the main UI
    st.header("Welcome to Mind and Muscle, Ask Anything")

 
    # Example: Load a PDF file programmatically
    pdf_file_path = './kiranmai.pdf'

    with open(pdf_file_path, 'rb') as f:
        pdf_file = BytesIO(f.read())
        pdf_file.name = os.path.basename(pdf_file_path)  # Set a name for the file

        raw_text = read_pdf([pdf_file])  # Read the PDF file
        # Display PDF content
        st.subheader("Content of the PDF:")
        st.text_area("PDF Content", raw_text, height=300) 
        text_chunks = get_chunks(raw_text)  # Chunk the text
        st.write("Hello")
        st.write(text_chunks)
        get_vector_store(text_chunks)  # Store embeddings
        
       

    user_query = st.text_input("Drop your Question")
    if user_query:
        user_input(user_query)


if __name__ == "__main__":
    main()
