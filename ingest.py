# Loads and indexes documents

import os
import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

os.environ["OPENAI_API_KEY"] = "insert OPENAI API key here"

def ingest_documents(directory_path):
    all_files = glob.glob(os.path.join(directory_path, "*.txt")) + \
                glob.glob(os.path.join(directory_path, "*.pdf"))

    documents = []
    for file_path in all_files:
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            continue  # Unsupported file type
        documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="db")

    print("✅ Documents indexed.")