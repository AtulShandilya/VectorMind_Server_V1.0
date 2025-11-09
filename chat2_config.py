"""
Configuration file for chat2 RAG pipeline.
Centralizes all configuration parameters for the RAG system.
"""
import os
from typing import Dict, Any

# ---------- ChromaDB Configuration ----------
CHROMA_DIR = os.getenv("CHROMA_DIR_CHAT2", "./chroma_db_chat2")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME_CHAT2", "chat2_collection")

# ---------- Chunking Configuration ----------
CHUNK_CHAR_SIZE = int(os.getenv("CHUNK_CHAR_SIZE", "2000"))
CHUNK_OVERLAP_SIZE = int(os.getenv("CHUNK_OVERLAP_SIZE", "300"))

# ---------- Embedding Configuration ----------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ---------- Retrieval Configuration ----------
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))

# ---------- LLM Configuration ----------
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ---------- FAISS Configuration ----------
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "L2")  # L2 or COSINE

# ---------- Text Processing Configuration ----------
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))  # Limit for LLM processing
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))  # Minimum chunk size

# ---------- Logging Configuration ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def get_config() -> Dict[str, Any]:
    """Returns all configuration as a dictionary."""
    return {
        "chroma_dir": CHROMA_DIR,
        "chroma_collection_name": CHROMA_COLLECTION_NAME,
        "chunk_char_size": CHUNK_CHAR_SIZE,
        "chunk_overlap_size": CHUNK_OVERLAP_SIZE,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "top_k_results": TOP_K_RESULTS,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "gemini_model": GEMINI_MODEL,
        "ollama_model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "gemini_base_url": GEMINI_BASE_URL,
        "default_temperature": DEFAULT_TEMPERATURE,
        "faiss_index_path": FAISS_INDEX_PATH,
        "faiss_index_type": FAISS_INDEX_TYPE,
        "max_text_length": MAX_TEXT_LENGTH,
        "min_chunk_size": MIN_CHUNK_SIZE,
        "log_level": LOG_LEVEL
    }



