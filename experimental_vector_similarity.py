from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Dogs are friendly and loyal pets.",
    "The Moon is Earth’s only natural satellite.",
    "GPT-4 is a large language model developed by OpenAI.",
    "Cats often purr when they are happy.",
    "Neural networks are used in deep learning."
]

def main():
    print("🧠 Local RAG Chatbot")
    print("Type your question below (or type 'exit' to quit):\n")

    model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
    doc_embeddings = model.encode(documents)

    while True:
        query = input(">> ")

        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        query_embeddings = model.encode([query])
        similarity = cosine_similarity(query_embeddings, doc_embeddings)[0]

        top_k = 3
        top_indices = np.argsort(similarity)[::-1][:top_k]

        print("\n Top Matching Documents:\n")
        for idx in top_indices:
            print(f"Score: {similarity[idx]:.4f} | Text: {documents[idx]}")

if __name__ == "__main__":
    main()


