import os
from typing import List, Dict
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import json

# Initialize models lazily
_model = None
_reranker = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _model

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker

def load_extracted_texts() -> Dict[str, str]:
    texts = {}
    extracted_dir = 'extracted_text'
    if not os.path.exists(extracted_dir):
        print("No extracted text directory found!")
        return texts

    for filename in os.listdir(extracted_dir):
        if filename.endswith('_text.txt'):
            file_path = os.path.join(extracted_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts[filename] = f.read()
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    return texts

def clean_text(text: str) -> str:
    """Clean the text by removing extra whitespace and normalizing"""

    # Remove unwanted bullet character
    text = text.replace('', '')

    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text

def split_by_headers(text: str, min_chunk_length=300) -> List[str]:
    lines = text.splitlines()
    sections = []
    current_chunk = []
    
    for line in lines:
        if re.match(r'^[A-Z\\s]{5,}$', line.strip()):
            # Store previous chunk if it's long enough
            if current_chunk and len(' '.join(current_chunk)) >= min_chunk_length:
                sections.append("\n".join(current_chunk))
                current_chunk = []
        current_chunk.append(line)
    
    if current_chunk:
        sections.append("\n".join(current_chunk))
    
    return sections


def extract_heading(chunk: str) -> str:
    lines = chunk.strip().splitlines()
    for line in lines:
        if re.match(r'^[A-Z\s]{5,}$', line.strip()):
            return line.strip()
    return "Unknown"

def create_embeddings(chunks: List[str]) -> List[np.ndarray]:
    model = get_model()
    embeddings = model.encode(chunks)
    return embeddings

def rerank_chunks(query: str, chunks: List[str]) -> List[str]:
    reranker = get_reranker()
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    reranked = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
    return reranked

def process_all_texts() -> Dict[str, Dict[str, List]]:
    texts = load_extracted_texts()
    processed_texts = {}
    for filename, text in texts.items():
        cleaned_text = clean_text(text)
        chunks = split_by_headers(cleaned_text)
        embeddings = create_embeddings(chunks)
        headers = [extract_heading(chunk) for chunk in chunks]
        processed_texts[filename] = {
            'chunks': chunks,
            'headers': headers,
            'embeddings': embeddings.tolist()
        }
    return processed_texts

def save_chunked_texts(processed_texts: Dict[str, Dict[str, List]]):
    if not os.path.exists('chunked_texts'):
        os.makedirs('chunked_texts')

    for filename, data in processed_texts.items():
        chunks_file = os.path.join('chunked_texts', f"{os.path.splitext(filename)[0]}_chunks.txt")
        try:
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(data['chunks'], 1):
                    f.write(f"=== Chunk {i} | Header: {data['headers'][i-1]} ===\n")
                    f.write(chunk)
                    f.write("\n\n")
            print(f"Saved chunked text to {chunks_file}")
        except Exception as e:
            print(f"Error saving chunks for {filename}: {str(e)}")

        embeddings_file = os.path.join('chunked_texts', f"{os.path.splitext(filename)[0]}_embeddings.json")
        try:
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'chunks': data['chunks'],
                    'headers': data['headers'],
                    'embeddings': data['embeddings']
                }, f, indent=2)
            print(f"Saved embeddings to {embeddings_file}")
        except Exception as e:
            print(f"Error saving embeddings for {filename}: {str(e)}")

def main():
    print("Starting text chunking and embedding process...")
    try:
        processed_texts = process_all_texts()
        if processed_texts:
            print("\nSaving chunked texts and embeddings...")
            save_chunked_texts(processed_texts)
            print("\nProcess completed successfully!")
        else:
            print("\nNo texts were found to process.")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
