import os
import glob
import json
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import requests
import re
from dotenv import load_dotenv
from bs4 import XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
load_dotenv()

# ----------- Extraction -----------

def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text

# ----------- Chunking -----------

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

# ----------- Load and Chunk Filings -----------

html_files = glob.glob(os.path.join('data', '*.htm*'))
filings_text = {}
for file in html_files:
    filings_text[file] = extract_text_from_html(file)

filings_chunks = {}
for file, text in filings_text.items():
    filings_chunks[file] = chunk_text(text)

# ----------- Embedding & FAISS -----------

model = SentenceTransformer('all-MiniLM-L6-v2')

chunk_texts = []
chunk_metadata = []

for file, chunks in filings_chunks.items():
    for idx, chunk in enumerate(chunks):
        chunk_texts.append(chunk)
        chunk_metadata.append({'file': file, 'chunk_id': idx})

embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ----------- Groq LLM Synthesis -----------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def groq_answer(query, context, model="llama-3.1-8b-instant"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        f"Answer the following question using only the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for financial filings."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print("Groq API error:", response.text)
        return "Groq API error."

# ----------- Retrieval -----------

def retrieve(query, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(chunk_metadata):
            continue
        meta = chunk_metadata[idx]
        file = meta['file']
        chunk_id = meta['chunk_id']
        chunk_text = filings_chunks[file][chunk_id]
        results.append({
            'file': file,
            'chunk_id': chunk_id,
            'text': chunk_text[:500] + '...' if len(chunk_text) > 500 else chunk_text
        })
    return results

# ----------- Agent Orchestration -----------

def agent_query(query, top_k=3):
    comparative_keywords = ["compare", "growth", "change", "across", "highest", "percentage", "difference"]
    needs_decomposition = any(word in query.lower() for word in comparative_keywords)
    sub_queries = []
    answers = []
    sources = []

    if needs_decomposition:
        if "compare" in query.lower() and "across" in query.lower():
            companies = ["Microsoft", "Alphabet", "NVIDIA"]
            years = re.findall(r"\b20\d{2}\b", query)
            metric = re.findall(r"compare (.*?) across", query.lower())
            for company in companies:
                for year in years:
                    sq = f"{company} {metric[0] if metric else 'revenue'} {year}"
                    sub_queries.append(sq)
        elif "growth" in query.lower() or "change" in query.lower():
            company = re.findall(r"(Microsoft|Alphabet|NVIDIA)", query, re.I)
            years = re.findall(r"\b20\d{2}\b", query)
            metric = re.findall(r"(revenue|margin|spending|investment|income|cloud)", query.lower())
            if company and years:
                for year in years:
                    sq = f"{company[0]} {metric[0] if metric else 'revenue'} {year}"
                    sub_queries.append(sq)
            else:
                sub_queries = [query]
        else:
            sub_queries = [q.strip() for q in re.split(r" and |, ", query) if q.strip()]
    else:
        sub_queries = [query]

    for sq in sub_queries:
        chunks = retrieve(sq, top_k=top_k)
        context = "\n\n".join([c['text'] for c in chunks])
        answer = groq_answer(sq, context) if context else "No relevant information found."
        answers.append(answer)
        for c in chunks:
            sources.append({
                "file": c['file'],
                "chunk_id": c['chunk_id'],
                "excerpt": c['text'][:200] + "..." if len(c['text']) > 200 else c['text']
            })

    synthesized_answer = " | ".join(answers)
    response = {
        "query": query,
        "answer": synthesized_answer,
        "reasoning": "Decomposed into sub-queries and used Groq to synthesize answers from retrieved chunks.",
        "sub_queries": sub_queries,
        "sources": sources[:top_k*len(sub_queries)]
    }
    return response