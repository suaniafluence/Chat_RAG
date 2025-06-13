import os
from sentence_transformers import SentenceTransformer
import chromadb

# Simple RAG pipeline: load files from 'docs' folder, build embeddings and search.

DOCS_DIR = "docs"
DB_DIR = "simple_chroma"
MODEL_NAME = "all-MiniLM-L6-v2"

# Create directories if needed
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Initialize embedding model and Chroma collection
model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("docs")


def index_documents():
    """Read text files from DOCS_DIR and store embeddings."""
    documents = []
    ids = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DOCS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(text)
            ids.append(filename)
    if documents:
        embeddings = model.encode(documents).tolist()
        collection.add(documents=documents, embeddings=embeddings, ids=ids)
        print(f"Indexed {len(documents)} documents.")
    else:
        print("No text documents found in 'docs' folder.")


def query(question: str, k: int = 3):
    """Retrieve relevant docs and print them."""
    query_emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=k)
    docs = results.get("documents", [[]])[0]
    print("\n--- Retrieved docs ---")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc[:200]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a simple RAG example")
    parser.add_argument("--index", action="store_true", help="Index documents")
    parser.add_argument("--ask", type=str, help="Ask a question")
    args = parser.parse_args()

    if args.index:
        index_documents()
    if args.ask:
        query(args.ask)
