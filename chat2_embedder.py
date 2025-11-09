"""
Embedding Module for chat2 RAG Pipeline.
Handles embedding generation and metadata extraction for chunks.
"""
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from chat2_config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

logger = logging.getLogger("chat2_embedder")

class DocumentEmbedder:
    """Handles embedding generation for text chunks."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the document embedder.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        try:
            self.embedder = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
            embeddings = self.embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            ).tolist()
            
            logger.info(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise ValueError(f"Embedding generation failed: {e}")
    
    def extract_metadata(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and enhance metadata from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of enhanced metadata dictionaries
        """
        enhanced_metadata = []
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {}).copy()
            
            # Extract basic statistics
            text = chunk.get("text", "")
            metadata["text_length"] = len(text)
            metadata["word_count"] = len(text.split())
            
            # Preserve existing metadata
            if "chunk_index" in chunk:
                metadata["chunk_index"] = chunk["chunk_index"]
            if "start_char" in chunk:
                metadata["start_char"] = chunk["start_char"]
            if "end_char" in chunk:
                metadata["end_char"] = chunk["end_char"]
            
            enhanced_metadata.append(metadata)
        
        logger.info(f"Extracted metadata for {len(enhanced_metadata)} chunks")
        return enhanced_metadata
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        Process chunks: generate embeddings and extract metadata.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            
        Returns:
            Tuple of (embeddings list, enhanced metadata list)
        """
        if not chunks:
            logger.warning("No chunks provided for processing")
            return [], []
        
        # Extract texts
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Extract metadata
        metadata = self.extract_metadata(chunks)
        
        logger.info(f"Processed {len(chunks)} chunks: {len(embeddings)} embeddings, {len(metadata)} metadata entries")
        return embeddings, metadata



