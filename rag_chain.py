# RAG logic using LangChain

import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI # OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = "insert OPENAI API key here"

# Constants
PERSIST_DIR = "vector_store"

def load_vectorstore():
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return vectordb

def get_rag_chain():
    vectordb = load_vectorstore()
    
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain
