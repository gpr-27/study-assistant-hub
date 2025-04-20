from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def create_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-large")

def get_text_splitter():
    # this splits text into chunks
    return RecursiveCharacterTextSplitter(
        chunk_size=10000,  
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

def load_folder_documents(folder_path):
    """
    Goes through a folder of PDFs organized by subject and loads them into a vector db.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Hey, this folder doesn't exist: {folder_path}")

    all_docs = []
    text_splitter = get_text_splitter()
    
    # go through each subject folder
    for subject_folder in folder_path.iterdir():
        if subject_folder.is_dir():
            subject = subject_folder.name
            
            # get all pdfs from this subject's folder
            loader = DirectoryLoader(
                str(subject_folder),
                glob="**/*.pdf",  
                loader_cls=PyPDFLoader
            )
            docs = loader.load()
            
            # Add metadata to documents
            for doc in docs:
                doc.metadata["subject"] = subject
                doc.metadata["document_type"] = "content"
            
            # Split into chunks and add to collection
            split_docs = text_splitter.split_documents(docs)
            all_docs.extend(split_docs)
    
    if not all_docs:
        raise ValueError(f"No PDFs found in {folder_path} :(")
    
    # create vector db (FAISS)
    embeddings = create_embeddings()
    db = FAISS.from_documents(all_docs, embeddings)
    return db

def load_single_pdf(pdf_path):

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise ValueError(f"Can't find this PDF: {pdf_path}")
    
    # Load the PDF
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    # Add metadata to documents
    for doc in docs:
        doc.metadata["source"] = str(pdf_path)
        doc.metadata["document_type"] = "content"
    
    text_splitter = get_text_splitter()
    split_docs = text_splitter.split_documents(docs)
    
    # create a small vector db just for this pdf
    embeddings = create_embeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    return db
