import os
import json
import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

class VectorStore:
    def __init__(self, persist_directory: str = "faiss_db"):
        """Initialize the vector store with FAISS."""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize FAISS index with cosine similarity (Inner Product)
        self.dimension = 384  # Must match BGE-small
        self.index = faiss.IndexFlatIP(self.dimension)

        # Store metadata
        self.documents = []
        self.metadata = []

        # Load existing index if available
        self._load_index()

        # Initialize embedding and reranking models
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _load_index(self):
        """Load existing FAISS index and metadata if available."""
        index_path = os.path.join(self.persist_directory, "index.faiss")
        metadata_path = os.path.join(self.persist_directory, "metadata.json")

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']

    def _save_index(self):
        """Save the FAISS index and metadata to disk."""
        index_path = os.path.join(self.persist_directory, "index.faiss")
        metadata_path = os.path.join(self.persist_directory, "metadata.json")

        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f, ensure_ascii=False, indent=2)

    def load_embeddings(self, embeddings_dir: str = "chunked_texts"):
        """Load embeddings from all JSON files in the directory."""
        if not os.path.exists(embeddings_dir):
            print(f"Embeddings directory {embeddings_dir} not found.")
            return

        # Clear existing data
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []

        for filename in os.listdir(embeddings_dir):
            if filename.endswith('_embeddings.json'):
                source_file = filename.replace('_embeddings.json', '.pdf')
                embeddings_file = os.path.join(embeddings_dir, filename)

                print(f"Loading embeddings from {filename}...")
                with open(embeddings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = data['chunks']
                    embeddings = data['embeddings']
                    headers = data.get('headers', ["Unknown"] * len(chunks))

                for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
                    embedding = np.array(embedding, dtype=np.float32)
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine
                    self.index.add(np.array([embedding]))
                    self.documents.append(text)
                    self.metadata.append({
                        'source': source_file,
                        'chunk_index': i,
                        'file_type': 'pdf',
                        'header': headers[i]
                    })

        self._save_index()
        print(f"Loaded {len(self.documents)} chunks from {len(os.listdir(embeddings_dir))} files")

    def query(self, query_text: str, n_results: int = 5, rerank: bool = True) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents."""
        query_embedding = self.model.encode(query_text)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize

        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            n_results
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                results.append({
                    'text': self.documents[idx],
                    'distance': float(distances[0][i]),
                    'metadata': self.metadata[idx]
                })

        if rerank and results:
            texts = [r['text'] for r in results]
            pairs = [[query_text, doc] for doc in texts]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(scores, results), reverse=True, key=lambda x: x[0])
            results = [r for _, r in ranked]

        return results

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents from the store."""
        return [
            {
                'text': doc,
                'metadata': meta
            }
            for doc, meta in zip(self.documents, self.metadata)
        ]

if __name__ == "__main__":
    vector_store = VectorStore()
    vector_store.load_embeddings()

    queries = [
        "What is controlling in management?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = vector_store.query(query)
        for result in results:
            print(f"\nText: {result['text'][:100]}...")
            print(f"Distance: {result['distance']}")
            print(f"Source: {result['metadata']['source']}")
            print(f"Header: {result['metadata']['header']}")
            print("---")