# local-rag-chatbot
A lightweight, local Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, HuggingFace, and OpenAI. Upload documents, ask natural-language questions, and get grounded, source-aware responses in real time.

# 🧠 Local RAG Chatbot

A lightweight Retrieval-Augmented Generation chatbot that uses LangChain and ChromaDB to answer questions grounded in your local documents.

## 🚀 Features
- Upload your own `.pdf`, `.txt`, or `.md` files
- Index documents with ChromaDB and embed via HuggingFace
- Ask questions in natural language
- Get grounded responses and relevant sources
- Simple CLI or Streamlit UI

## 🛠️ Stack
- Python
- LangChain
- ChromaDB
- OpenAI API
- Streamlit (optional)

## 📦 Setup

```bash
git clone https://github.com/your-username/local-rag-chatbot.git
cd local-rag-chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
