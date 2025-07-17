
# RAG logic using LangChain

import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

# Load API key
os.environ["OPENAI_API_KEY"] = "insert OPENAI API key here"

# Constants
PERSIST_DIR = "vector_store"

def ask_question(query):
    embeddings = HuggingFaceEmbeddings()
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever, return_source_documents = True)
    result = qa_chain.invoke(query)
    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }
