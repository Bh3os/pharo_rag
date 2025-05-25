#!/usr/bin/env python3


import os
import json
import numpy as np
import argparse
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions

def load_embeddings(embeddings_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
   
    # Load embeddings
    embeddings_file = os.path.join(embeddings_dir, 'embeddings.npy')
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings with shape {embeddings.shape} from {embeddings_file}")
    
    # Load metadata
    metadata_file = os.path.join(embeddings_dir, 'chunk_metadata.json')
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} chunks from {metadata_file}")
    
    # Load chunks with text
    chunks_file = os.path.join(embeddings_dir, 'chunks_with_embeddings.json')
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    texts = [chunk["text"] for chunk in chunks]
    print(f"Loaded {len(texts)} text chunks from {chunks_file}")
    
    return embeddings, metadata, texts

def create_chroma_collection(embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                            texts: List[str], output_dir: str) -> chromadb.Collection:
    
    chroma_dir = os.path.join(output_dir, 'chroma_db')
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Create or get collection
    collection_name = "ancient_egypt_collection"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing Chroma collection: {collection_name}")
    except:
        collection = client.create_collection(name=collection_name)
        print(f"Created new Chroma collection: {collection_name}")
    
    # Prepare IDs, embeddings, documents, and metadata for Chroma
    ids = [f"chunk_{i}" for i in range(len(texts))]
    
    # Convert metadata to format compatible with Chroma
    chroma_metadata = []
    for meta in metadata:
        # Convert all values to strings to ensure compatibility
        chroma_meta = {k: str(v) for k, v in meta.items()}
        chroma_metadata.append(chroma_meta)
    
    
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_ids = ids[i:end_idx]
        batch_embeddings = embeddings[i:end_idx].tolist()
        batch_texts = texts[i:end_idx]
        batch_metadata = chroma_metadata[i:end_idx]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadata
        )
        print(f"Added batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} to Chroma collection")
    
    print(f"Chroma collection created with {collection.count()} items")
    return collection

def test_chroma_retrieval(collection: chromadb.Collection, query_text: str, 
                         query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Test the Chroma collection with a query.
    
    Args:
        collection: Chroma collection
        query_text: Query text (for logging only)
        query_embedding: Query embedding vector
        top_k: Number of results to return
        
    Returns:
        List of retrieval results
    """
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    print(f"Query: {query_text}")
    print(f"Found {len(results['documents'][0])} results")
    
    # Format results
    formatted_results = []
    for i in range(len(results['documents'][0])):
        formatted_results.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i] if 'distances' in results else None,
            "id": results['ids'][0][i]
        })
    
    return formatted_results

def create_chroma_retriever(embeddings_dir: str, output_dir: str) -> Dict[str, Any]:
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embeddings and metadata
    embeddings, metadata, texts = load_embeddings(embeddings_dir)
    
    # Create Chroma collection
    collection = create_chroma_collection(embeddings, metadata, texts, output_dir)
    
    # Create retriever dictionary
    retriever = {
        "collection": collection,
        "embedding_dim": embeddings.shape[1],
        "collection_name": collection.name
    }
    
    # Save retriever info
    retriever_info = {
        "collection_name": collection.name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_documents": collection.count(),
        "chroma_dir": os.path.join(output_dir, 'chroma_db')
    }
    
    retriever_file = os.path.join(output_dir, 'chroma_retriever_info.json')
    with open(retriever_file, 'w', encoding='utf-8') as f:
        json.dump(retriever_info, f)
    print(f"Saved retriever info to {retriever_file}")
    
    return retriever

def main():
    
    parser = argparse.ArgumentParser(description='Create Chroma retriever for RAG system')
    parser.add_argument('--embeddings', type=str, required=True, help='Directory containing embeddings and metadata')
    parser.add_argument('--output', type=str, required=True, help='Output directory for Chroma database')
    
    args = parser.parse_args()
    
    print(f"Creating Chroma retriever from embeddings in {args.embeddings}")
    
    retriever = create_chroma_retriever(args.embeddings, args.output)
    
    print(f"Chroma retriever created successfully")

if __name__ == "__main__":
    main()
