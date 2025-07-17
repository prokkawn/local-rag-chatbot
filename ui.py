# ui.py
import os
import shutil
import streamlit as st
import tempfile
from ingest import ingest_documents
from rag_chain import ask_question
from collections import defaultdict

# Clear temp_docs/ on every run
doc_folder = "temp_docs"
if os.path.exists(doc_folder):
    shutil.rmtree(doc_folder)
os.makedirs(doc_folder, exist_ok=True)

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("🧠 Local RAG Chatbot")

st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Choose .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)

doc_folder = "temp_docs"
os.makedirs(doc_folder, exist_ok=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(doc_folder, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    with st.spinner("Indexing documents..."):
        ingest_documents(doc_folder)
    st.success("Documents successfully indexed.")

query = st.text_input("Ask a question about your documents:")
if query:
    with st.spinner("Generating answer..."):
        response = ask_question(query)
    st.markdown("### Answer")
    st.write(response['answer'])

    source_map = defaultdict(list)
    for doc in response["sources"]:
        source_map[doc.metadata.get("source", "Unknown")].append(doc.page_content.strip())

    for source, chunks in source_map.items():
        st.markdown(f"### 📄 `{source}`")
        for i, content in enumerate(set(chunks)):  # Deduplicate again
            with st.expander(f"Chunk {i+1}"):
                st.code(content[:1500], language="text")
