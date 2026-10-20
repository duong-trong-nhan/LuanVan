# -*- coding: utf-8 -*-
"""
HerittageFNS: Phase 3 - Text Chunking, Vectorization & ChromaDB Storage
========================================================================
- Source: definition_rag_optimized.md (structured musical knowledge)
- Semantic Chunking & Metadata extraction (heritage, aspect, chunk_id, sources)
- Multilingual Sentence Embeddings: paraphrase-multilingual-MiniLM-L12-v2
- Persistent Vector DB: ChromaDB (./data/chromadb/)
- Automated test query & verification for Neuro-Symbolic / RAG explanations
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Setup paths
docs_path = Path("./definition_rag_optimized.md")
chroma_dir = Path("./data/chromadb")
chroma_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Parse & Clean Markdown Chunks
# -----------------------------------------------------------------------------
print("="*70)
print("1. Parsing and Cleaning Chunks from definition_rag_optimized.md...")
print("="*70)

with open(docs_path, 'r', encoding='utf-8') as f:
    raw_content = f.read()

# Mapping from chunk_id prefix to traditional music genre
CHUNK_GENRE_MAP = {
    'ca_tru': 'Ca trù',
    'cai_luong': 'Cải lương',
    'chau_van': 'Chầu văn',
    'cheo': 'Chèo',
    'hat_xam': 'Hát xẩm',
    'quan_ho': 'Quan họ',
}

# Split by "## CHUNK:" markers
raw_chunks = re.split(r'\n## CHUNK:\s*', raw_content)
chunks = []

for rc in raw_chunks[1:]:  # skip initial file preamble
    lines_list = rc.strip().split('\n')
    chunk_id = lines_list[0].strip()
    body = '\n'.join(lines_list[1:]).strip()
    
    # Check if this is a music genre chunk
    genre_key = None
    for k in CHUNK_GENRE_MAP:
        if chunk_id.startswith(k):
            genre_key = k
            break
            
    # Explicitly filter out developer guidelines (e.g. rag_vector_metadata, rag_multimodal_evidence, etc.)
    if not genre_key:
        print(f"Skipping non-genre guideline chunk: {chunk_id}")
        continue
        
    genre = CHUNK_GENRE_MAP[genre_key]
    aspect = chunk_id[len(genre_key)+1:]
    
    # Extract source URLs if present
    sources = []
    src_match = re.search(r'\*\*Nguồn tham khảo:\*\*(.*)', body, re.DOTALL)
    clean_body = body
    if src_match:
        urls = re.findall(r'https?://[^\s\)]+', src_match.group(1))
        if urls:
            sources.extend(urls)
        # Strip source section from text to keep pure domain knowledge for embedding
        clean_body = body[:src_match.start()].strip()
        
    # Strip markdown separator lines and clean whitespace
    clean_body = re.sub(r'^-{3,}', '', clean_body, flags=re.MULTILINE).strip()
    clean_body = re.sub(r'\n{3,}', '\n\n', clean_body).strip()
    
    chunks.append({
        "chunk_id": chunk_id,
        "heritage": genre,
        "aspect": aspect,
        "source": "; ".join(sources) if sources else "definition_rag_optimized.md",
        "text": clean_body
    })

print(f"\n-> Extracted {len(chunks)} clean semantic chunks across 6 traditional genres:")
genre_counts = pd.Series([c["heritage"] for c in chunks]).value_counts()
for g, cnt in genre_counts.items():
    print(f"   • {g:12s}: {cnt} chunks")

# Save cleaned chunks to JSON for reference
chunks_json_path = chroma_dir / "rag_indexed_chunks.json"
with open(chunks_json_path, 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print(f"-> Saved chunks inspection metadata to: {chunks_json_path}")

# -----------------------------------------------------------------------------
# 2. Vector Embedding with Multilingual MiniLM
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("2. Generating Multilingual Sentence Embeddings...")
print("="*70)

embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"Loading embedding model: {embedding_model_name}...")
embedder = SentenceTransformer(embedding_model_name)

chunk_texts = [c["text"] for c in chunks]
embeddings = embedder.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)
print(f"-> Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

# -----------------------------------------------------------------------------
# 3. Persistent Storage in ChromaDB
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("3. Storing into ChromaDB Persistent Collection...")
print("="*70)

client = chromadb.PersistentClient(path=str(chroma_dir))
collection_name = "vietnamese_traditional_music_rag"

# Reset collection if exists to ensure clean index
try:
    client.delete_collection(name=collection_name)
except Exception:
    pass

collection = client.create_collection(
    name=collection_name,
    metadata={"description": "Vietnamese Traditional Music Knowledge Base for Neuro-Symbolic RAG"}
)

ids = [c["chunk_id"] for c in chunks]
metadatas = [
    {
        "heritage": c["heritage"],
        "aspect": c["aspect"],
        "source": c["source"]
    }
    for c in chunks
]
documents = chunk_texts

collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    documents=documents
)

print(f"-> Successfully indexed {collection.count()} chunks in ChromaDB at {chroma_dir}!")

# -----------------------------------------------------------------------------
# 4. Verification & Sample Neuro-Symbolic Retrieval Queries
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("4. Testing Neuro-Symbolic Explanation Retrieval...")
print("="*70)

test_queries = [
    ("Ca trù", "Kỹ thuật luyến láy, đổ hột nảy hạt và vai trò của phách và đàn đáy"),
    ("Quan họ", "Kỹ thuật hát Vang Rền Nền Nảy và phong cách giao duyên liền anh liền chị"),
    ("Cải lương", "Điệu hát Vọng cổ, kỹ thuật rung ngân sâu và dàn nhạc cổ nhạc Nam Bộ"),
    ("Chầu văn", "Âm hưởng tiếng đàn Nguyệt, nhịp điệu dồn dập trong nghi lễ hầu đồng"),
    ("Hát xẩm", "Tính chất tự sự đường phố, nhạc cụ đàn nhị, sênh và trống mảnh"),
    ("Chèo", "Các làn điệu Chèo truyền thống, giọng hát cao sáng và không gian sân đình")
]

for target_genre, q_text in test_queries:
    print(f"\n[QUERY] {target_genre}: \"{q_text}\"")
    q_emb = embedder.encode([q_text], normalize_embeddings=True).tolist()
    
    results = collection.query(
        query_embeddings=q_emb,
        n_results=2,
        where={"heritage": target_genre}
    )
    
    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        aspect = results["metadatas"][0][i]["aspect"]
        dist = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
        doc_snippet = results["documents"][0][i][:160].replace('\n', ' ')
        print(f"   Rank {i+1} | ID: {chunk_id:25s} (Khía cạnh: {aspect}) | Khoảng cách: {dist:.4f}")
        print(f"          Snippet: {doc_snippet}...")

print("\n" + "="*70)
print("Hoàn tất Giai đoạn 3 (Text Chunking, Vectorization & ChromaDB Storage)!")
print("="*70)
