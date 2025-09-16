"""
==========================================================================================
RAG ENGINE - OFFLINE PROCESSING SCRIPT (OpenAI Version)
==========================================================================================
Purpose:
This script uses the OpenAI API to process a source PDF and create a high-quality
FAISS vector store. This is the fast and powerful "baking" step.

How to Use:
1. Ensure your OPENAI_API_KEY is in the .env file and you have billing set up.
2. Place the PDF you want to process in the project folder.
3. Run the script from your terminal: python rag_engine.py
==========================================================================================
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def process_pdf(pdf_path):
    """Loads and splits the PDF document into chunks."""
    print(f"Loading document from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, index_name):
    """Creates and saves a FAISS vector store using OpenAI embeddings."""
    print("Creating vector store with OpenAI embeddings... This will be fast.")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_name)
    print(f"Vector store created and saved locally as '{index_name}'.")
    return vector_store

if __name__ == '__main__':
    load_dotenv()
    
    PDF_PATH = "msft_report.pdf"
    INDEX_NAME = "microsoft_faiss_index"

    if os.path.exists(PDF_PATH):
        print(f"Starting processing for '{PDF_PATH}'...")
        text_chunks = process_pdf(PDF_PATH)
        if text_chunks:
            create_vector_store(text_chunks, INDEX_NAME)
        print("Processing complete.")
    else:
        print(f"Error: The file '{PDF_PATH}' was not found in the project folder.")