

import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple

def clean_text(text: str) -> str:
    
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    text = re.sub(r'\n\s*The Oxford History of the Ancient Near East\s*\n', '\n', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    text = re.sub(r'\xa0', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    text = re.sub(r'\[\d+\]', '', text)
    
    return text.strip()

def split_into_sections(text: str) -> List[str]:
   
    section_patterns = [
        r'CHAPTER \d+[.\s:].*?\n',
        r'Chapter \d+[.\s:].*?\n',
        r'\n\d+\.\s+[A-Z].*?\n',
        r'\n[A-Z][A-Z\s]+\n',
        r'\n[IVX]+\.\s+.*?\n'
    ]
    
    
    combined_pattern = '|'.join(section_patterns)
    
    
    potential_headings = re.findall(combined_pattern, text)
    
    
    if not potential_headings:
        return [section.strip() for section in text.split('\n\n') if section.strip()]
    
    
    sections = []
    last_end = 0
    
    for heading in potential_headings:
        heading_start = text.find(heading, last_end)
        
        
        if heading_start > last_end:
            section_text = text[last_end:heading_start].strip()
            if section_text:
                sections.append(section_text)
        
        
        next_heading_start = len(text)
        for next_heading in potential_headings:
            pos = text.find(next_heading, heading_start + len(heading))
            if pos > heading_start and pos < next_heading_start:
                next_heading_start = pos
        
        
        section_text = text[heading_start:next_heading_start].strip()
        if section_text:
            sections.append(section_text)
        
        last_end = next_heading_start
    
    
    if last_end < len(text):
        section_text = text[last_end:].strip()
        if section_text:
            sections.append(section_text)
    
    return sections

def create_overlapping_chunks(sections: List[str], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    
    chunks = []
    chunk_id = 0
    
    for section_idx, section in enumerate(sections):
        
        if len(section) < 50:
            continue
        
        
        first_line = section.split('\n')[0].strip()
        section_title = first_line if len(first_line) < 100 else "Section"
        
        
        if len(section) <= chunk_size:
            chunks.append({
                "chunk_id": chunk_id,
                "section_idx": section_idx,
                "section_title": section_title,
                "text": section,
                "start_char": 0,
                "end_char": len(section)
            })
            chunk_id += 1
            continue
        
        
        sentences = re.split(r'(?<=[.!?])\s+', section)
        current_chunk = []
        current_length = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            
            if current_length + sentence_length > chunk_size and current_length > 0:
                
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "chunk_id": chunk_id,
                    "section_idx": section_idx,
                    "section_title": section_title,
                    "text": chunk_text,
                    "start_char": start_char,
                    "end_char": start_char + len(chunk_text)
                })
                chunk_id += 1
                
                
                overlap_chars = 0
                sentences_to_keep = []
                
                
                for s in reversed(current_chunk):
                    if overlap_chars + len(s) <= overlap:
                        sentences_to_keep.insert(0, s)
                        overlap_chars += len(s) + 1  
                    else:
                        break
                
                
                current_chunk = sentences_to_keep
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                start_char = start_char + len(chunk_text) - current_length
            
        
            current_chunk.append(sentence)
            current_length += sentence_length + (1 if current_length > 0 else 0)  # Add space except for first sentence
        
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "chunk_id": chunk_id,
                "section_idx": section_idx,
                "section_title": section_title,
                "text": chunk_text,
                "start_char": start_char,
                "end_char": start_char + len(chunk_text)
            })
            chunk_id += 1
    
    return chunks

def process_text_file(input_file: str, output_dir: str, chunk_size: int = 1000, overlap: int = 200) -> Tuple[int, int]:
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    
    cleaned_text = clean_text(text)
    
    
    cleaned_file = os.path.join(output_dir, 'cleaned_text.txt')
    with open(cleaned_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    print(f"Saved cleaned text to {cleaned_file}")
    
    # Split into sections
    sections = split_into_sections(cleaned_text)
    
    # Save sections
    sections_file = os.path.join(output_dir, 'sections.json')
    with open(sections_file, 'w', encoding='utf-8') as f:
        json.dump(sections, f, indent=2)
    print(f"Split text into {len(sections)} sections and saved to {sections_file}")
    
    # Create overlapping chunks
    chunks = create_overlapping_chunks(sections, chunk_size, overlap)
    
    # Save chunks
    chunks_file = os.path.join(output_dir, 'chunks.json')
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    print(f"Created {len(chunks)} overlapping chunks and saved to {chunks_file}")
    
    # Save chunk statistics
    stats = {
        "total_sections": len(sections),
        "total_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunk_lengths": [len(chunk["text"]) for chunk in chunks],
        "avg_chunk_length": sum(len(chunk["text"]) for chunk in chunks) / len(chunks) if chunks else 0
    }
    
    stats_file = os.path.join(output_dir, 'chunk_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved chunk statistics to {stats_file}")
    
    return len(sections), len(chunks)

def main():
    parser = argparse.ArgumentParser(description='Preprocess and chunk text for RAG system')
    parser.add_argument('--input', type=str, required=True, help='Input text file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed chunks')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Target chunk size in characters')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap size in characters')
    
    args = parser.parse_args()
    
    print(f"Processing file: {args.input}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
    
    num_sections, num_chunks = process_text_file(
        args.input, 
        args.output,
        args.chunk_size,
        args.overlap
    )
    
    print(f"Processing complete. Created {num_chunks} chunks from {num_sections} sections.")

if __name__ == "__main__":
    main()
