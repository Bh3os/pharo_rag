

import os
import json
import argparse
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# Model configuration
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"

def get_embeddings(model, tokenizer, texts, batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generate embeddings for a list of texts using the Hugging Face model.
    
    Args:
        model: The Hugging Face model
        tokenizer: The Hugging Face tokenizer
        texts: List of text strings to embed
        batch_size: Number of texts to process in each batch
        device: Device to run the model on
        
    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the texts
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token embedding as the sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)
        
        
        time.sleep(0.1)
    
    return all_embeddings

def process_chunks(chunks_file, output_dir, batch_size=8):
    """
    Process text chunks and generate embeddings.
    
    Args:
        chunks_file: Path to the JSON file containing text chunks
        output_dir: Directory to save the embeddings
        batch_size: Number of chunks to process in each batch
        
    Returns:
        Tuple of (number of chunks processed, embedding dimension)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the chunks file
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")
    
    # Extract texts from chunks
    texts = [chunk["text"] for chunk in chunks]
    
    # Load model and tokenizer with trust_remote_code=True
    print(f"Loading model and tokenizer for {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    
    print(f"Generating embeddings using {MODEL_NAME}")
    embeddings = get_embeddings(model, tokenizer, texts, batch_size, device)
    
    
    for i, embedding in enumerate(embeddings):
        chunks[i]["embedding"] = embedding.tolist()
    
    
    chunks_with_embeddings_file = os.path.join(output_dir, 'chunks_with_embeddings.json')
    with open(chunks_with_embeddings_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f)
    print(f"Saved chunks with embeddings to {chunks_with_embeddings_file}")
    
    # Save embeddings separately as numpy array for faster loading
    embeddings_array = np.array(embeddings)
    embeddings_file = os.path.join(output_dir, 'embeddings.npy')
    np.save(embeddings_file, embeddings_array)
    print(f"Saved embeddings array to {embeddings_file}")
    
    # Save chunk IDs and metadata mapping
    chunk_metadata = [{
        "chunk_id": chunk["chunk_id"],
        "section_idx": chunk["section_idx"],
        "section_title": chunk["section_title"],
        "start_char": chunk["start_char"],
        "end_char": chunk["end_char"]
    } for chunk in chunks]
    
    metadata_file = os.path.join(output_dir, 'chunk_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f)
    print(f"Saved chunk metadata to {metadata_file}")
    
    # Save embedding statistics
    embedding_dim = len(embeddings[0]) if embeddings else 0
    stats = {
        "total_chunks": len(chunks),
        "embedding_dimension": embedding_dim,
        "model_name": MODEL_NAME,
        "batch_size": batch_size
    }
    
    stats_file = os.path.join(output_dir, 'embedding_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f)
    print(f"Saved embedding statistics to {stats_file}")
    
    return len(chunks), embedding_dim

def main():
    
    parser = argparse.ArgumentParser(description='Generate embeddings for text chunks using Hugging Face')
    parser.add_argument('--chunks', type=str, required=True, help='Path to the chunks JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for embeddings')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    
    args = parser.parse_args()
    
    print(f"Processing chunks file: {args.chunks}")
    print(f"Batch size: {args.batch_size}")
    
    num_chunks, embedding_dim = process_chunks(
        args.chunks, 
        args.output,
        args.batch_size
    )
    
    print(f"Processing complete. Generated {embedding_dim}-dimensional embeddings for {num_chunks} chunks.")

if __name__ == "__main__":
    main()
