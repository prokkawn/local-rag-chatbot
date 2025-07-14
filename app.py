# Main chatbot Script

from rag_chain import get_rag_chain
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("🧠 Local RAG Chatbot")
    print("Type your question below (or type 'exit' to quit):\n")

    qa_chain = get_rag_chain()

    while True:
        query = input(">> ")

        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        result = qa_chain.invoke(query)

        print("\n🔍 Answer:\n")
        print(result['result'])

        print("\n📚 Sources:")
        for doc in result['source_documents']:
            source = doc.metadata.get("source", "Unknown")
            preview = doc.page_content[:200].replace('\n', ' ')
            print(f"- {source}: {preview}...")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
