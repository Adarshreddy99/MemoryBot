import os
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from uuid import uuid4

# Import local model loaders
from load_model import load_llm, load_embedder

load_dotenv()

# ----------------------------
# Environment Setup
# ----------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

UPLOAD_DIR = os.getenv("UPLOAD_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR")
EMBED_BACKUP_FILE = os.path.join(EMBEDDINGS_DIR, "backup.json")

EMBED_MODEL_PATH = os.getenv("LOCAL_EMBED_MODEL_PATH")

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
duckduckgo = DuckDuckGoSearchRun()

# Load local LLM
llm = load_llm()

# ----------------------------
# PDF Upload → Chunk → Embed
# ----------------------------

def _chunk_text(text: str, source: str) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return [
        {
            "id": f"{source}_chunk_{i}",
            "text": chunk,
            "source": source,
            "chunk_id": i
        }
        for i, chunk in enumerate(chunks)
    ]

def _embed_chunks(chunks: List[Dict]) -> List[PointStruct]:
    texts = [c["text"] for c in chunks]
    vectors = embedding_model.embed_documents(texts)
    return [
        PointStruct(
            id=chunk["id"],
            vector=vec,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"]
            }
        )
        for chunk, vec in zip(chunks, vectors)
    ]

def _append_to_backup(chunks: List[Dict]):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    backup_data = []
    if os.path.exists(EMBED_BACKUP_FILE):
        with open(EMBED_BACKUP_FILE, "r") as f:
            backup_data = json.load(f)
    backup_data.extend(chunks)
    with open(EMBED_BACKUP_FILE, "w") as f:
        json.dump(backup_data, f, indent=2)

def _save_pdf(pdf_path: str) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = os.path.basename(pdf_path)
    saved_path = os.path.join(UPLOAD_DIR, filename)
    os.rename(pdf_path, saved_path)
    return saved_path

def process_and_store_pdf(pdf_path: str):
    saved_path = _save_pdf(pdf_path)
    filename = os.path.splitext(os.path.basename(saved_path))[0]

    loader = PyPDFLoader(saved_path)
    pages = loader.load()
    text = "\n".join(p.page_content for p in pages)

    chunks = _chunk_text(text, filename)
    vectors = _embed_chunks(chunks)

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=vectors)
    _append_to_backup(chunks)

# ----------------------------
# MMR Retrieval from Qdrant
# ----------------------------

def retrieve_chunks(query: str, top_k: int = 5, score_threshold: float = 0.4) -> Tuple[List[Dict], float]:
    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )

    embedded_query = embedding_model.embed_query(query)

    docs_with_scores = qdrant.max_marginal_relevance_search_by_vector(
        embedding=embedded_query,
        k=top_k,
        fetch_k= top_k * 2,
        lambda_mult=0.6,
    )

    chunks = []
    max_score = 0
    for doc, score in docs_with_scores:
        max_score = max(max_score, score)
        chunks.append({
            "text": doc.page_content,
            "score": score,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", -1)
        })
    return chunks, max_score


def extract_llm_response(response) -> str:
    """Extract text from llama-cpp-python response"""
    if isinstance(response, dict):
        return response.get("choices", [{}])[0].get("text", "").strip()
    else:
        return str(response).strip()

# ----------------------------
# Query Decomposition with Local LLM
# ----------------------------

def decompose_query(query: str) -> List[str]:
    prompt = f"""You are an expert at breaking down complex questions into simpler sub-questions.

Given the main question below, decompose it into exactly 2 specific, focused sub-questions that together would help answer the original question completely. Each sub-question should be clear, concise, and searchable.

Main Question: {query}

Please provide exactly 2 sub-questions in the following format:
1. [First sub-question]
2. [Second sub-question]

Sub-questions:"""

    response = llm(prompt, max_tokens=200, temperature=0.3, stop=["Sub-questions:", "\n\n"])
    response = extract_llm_response(response)

    # Parse the response to extract sub-questions
    lines = response.strip().split('\n')
    sub_questions = []
    
    for line in lines:
        line = line.strip()
        if line and (line.startswith('1.') or line.startswith('2.')):
            # Remove the number prefix and clean up
            sub_q = line[2:].strip()
            if sub_q:
                sub_questions.append(sub_q)
    
    # Ensure we have exactly 2 sub-questions
    if len(sub_questions) < 2:
        # Fallback: create simple sub-questions
        sub_questions = [
            f"What are the key concepts related to: {query}?",
            f"What are the practical applications or examples of: {query}?"
        ]
    
    return sub_questions[:2]

# ----------------------------
# Answer Generation with Local LLM
# ----------------------------

def generate_answer_from_chunks(question: str, chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant information found in the documents."
    
    # Combine chunk texts
    context = "\n\n".join([f"Source: {chunk['source']}\n{chunk['text']}" for chunk in chunks])
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context from documents.

Context from documents:
{context}

Question: {question}

Please provide a clear, accurate answer based on the context above. If the context doesn't contain enough information to answer the question, mention that explicitly.

Answer:"""

    response = llm(prompt, max_tokens=300, temperature=0.2)
    return extract_llm_response(response)

def generate_web_answer(question: str, web_content: str) -> str:
    prompt = f"""You are a helpful assistant that answers questions based on web search results.

Web search results:
{web_content}

Question: {question}

Please provide a clear, accurate answer based on the web search results above.

Answer:"""

    response = llm(prompt, max_tokens=500, temperature=0.2)
    return extract_llm_response(response)

# ----------------------------
# Full Corrective RAG Flow
# ----------------------------

def corrective_rag(query: str, score_threshold: float = 0.4) -> str:
    # Step 1: Decompose query into sub-questions
    sub_questions = decompose_query(query)
    
    retrieved_chunks = []
    sub_answers = []
    
    # Step 2: Process each sub-question
    for sub_q in sub_questions:
        chunks, max_score = retrieve_chunks(sub_q, score_threshold=score_threshold)
        
        if max_score < score_threshold:
            # Use web search if retrieval score is low
            web_content = duckduckgo.run(sub_q)
            sub_answer = generate_web_answer(sub_q, web_content)
        else:
            # Use retrieved chunks
            retrieved_chunks.extend(chunks)
            sub_answer = generate_answer_from_chunks(sub_q, chunks)
        
        sub_answers.append({"question": sub_q, "answer": sub_answer})
    
    # Step 3: Generate final comprehensive answer
    sub_answers_text = "\n\n".join([f"Q: {sa['question']}\nA: {sa['answer']}" for sa in sub_answers])
    
    chunks_text = ""
    if retrieved_chunks:
        chunks_text = "\n\n".join([f"Source: {chunk['source']}\n{chunk['text']}" for chunk in retrieved_chunks])
    
    final_prompt = f"""You are a helpful assistant that provides comprehensive answers by synthesizing information from multiple sources.

Original Question: {query}

Sub-question Analysis:
{sub_answers_text}

Additional Context from Documents:
{chunks_text}

Based on all the information above, provide a comprehensive, well-structured final answer to the original question. Synthesize the information from the sub-questions and any additional context to give the most complete and accurate response possible.

Final Answer:"""

    final_response = llm(final_prompt, max_tokens=500, temperature=0.3)
    return extract_llm_response(final_response)