# Financial Q&A System with Agent Capabilities

## Overview
This project is an AI Engineering Assignment for a RAG Sprint Challenge. It implements a focused Retrieval-Augmented Generation (RAG) system with basic agent capabilities to answer financial questions about Google, Microsoft, and NVIDIA using their recent 10-K filings. The system supports query decomposition and multi-step reasoning for complex questions.

## Features
- Vector-based RAG pipeline: text extraction, chunking, embeddings, vector search, and retrieval
- Agent orchestration: query decomposition, multi-step retrieval, and synthesis
- JSON output with answers, reasoning, sub-queries, and sources
- Supports simple, comparative, and cross-company financial queries

## Data Scope
- Companies: Google (GOOGL), Microsoft (MSFT), NVIDIA (NVDA)
- Documents: Annual 10-K filings for 2022, 2023, 2024
- Format: HTML (see `data/` folder)
- Source: SEC EDGAR database

## Setup Instructions
1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the demo:
   ```
   python main.py
   ```
   Sample queries will be answered and results saved to `responses.json`.

## File Structure
- `main.py`: Runs sample queries and saves results
- `docassist.py`: Core RAG and agent logic
- `data/`: Contains 10-K filings
- `responses.json`: Stores output for test queries
- `requirements.txt`: Python dependencies

## Design Choices
- **Chunking:** Semantic chunks (200–1000 tokens) for retrieval
- **Embeddings:** Uses a sentence-transformers model for vectorization
- **Vector Store:** In-memory (e.g., FAISS or ChromaDB)
- **Agent Logic:** Function-based query decomposition and synthesis

## Supported Query Types
1. Basic metrics (e.g., "What was Microsoft's total revenue in 2023?")
2. Year-over-year comparison (e.g., "How did NVIDIA's data center revenue grow from 2022 to 2023?")
3. Cross-company analysis (e.g., "Which company had the highest operating margin in 2023?")
4. Segment analysis (e.g., "What percentage of Google's revenue came from cloud in 2023?")
5. AI strategy (e.g., "Compare AI investments mentioned by all three companies in their 2024 10-Ks")

## Output Format
JSON responses include:
- `query`: The original question
- `answer`: Synthesized answer
- `reasoning`: Explanation of how the answer was derived
- `sub_queries`: List of sub-queries executed
- `sources`: Excerpts and references from filings

## How to Extend
- Add more filings to `data/`
- Update `main.py` with new queries
- Enhance agent logic in `docassist.py`

## Notes
- No production deployment, UI, or authentication
- Focus on clean engineering and retrieval accuracy
- Demo output is printed and saved to `responses.json`

