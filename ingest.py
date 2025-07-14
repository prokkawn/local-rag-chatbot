# Loads and indexes documents

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

os.environ["OPENAI_API_KEY"] = "insert your key here"

# Paths
DOCS_DIR = "example_docs"
PERSIST_DIR = "vector_store"

def load_documents(directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Add more types later
            path = os.path.join(directory, filename)
            loader = TextLoader(path)
            docs.extend(loader.load())
    return docs

def main():
    print("Loading documents...")
    documents = load_documents(DOCS_DIR)

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f"Total chunks: {len(chunks)}. Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = OpenAIEmbeddings()

    print("Storing in ChromaDB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("✅ Vector store saved!")

if __name__ == "__main__":
    main()
