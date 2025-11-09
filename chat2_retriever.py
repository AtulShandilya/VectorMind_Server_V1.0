"""
Retriever Module for chat2 RAG Pipeline.
Handles user queries, embeds them, and retrieves top-k relevant chunks.
"""
import logging
from typing import List, Dict, Any, Optional
from chat2_embedder import DocumentEmbedder
from chat2_vectorstore import VectorStore
from chat2_config import TOP_K_RESULTS, SIMILARITY_THRESHOLD

logger = logging.getLogger("chat2_retriever")

class DocumentRetriever:
    """Handles query embedding and document retrieval."""
    
    def __init__(self, embedder: DocumentEmbedder, vector_store: VectorStore):
        """
        Initialize the document retriever.
        
        Args:
            embedder: DocumentEmbedder instance for query embedding
            vector_store: VectorStore instance for document retrieval
        """
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("Initialized DocumentRetriever")
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved document dictionaries with relevance scores
        """
        if not query:
            logger.warning("Empty query provided")
            return []
        
        try:
            # Generate query embedding
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            query_embeddings = self.embedder.generate_embeddings([query])
            
            if not query_embeddings:
                logger.error("Failed to generate query embedding")
                return []
            
            query_embedding = query_embeddings[0]
            
            # Search in vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            # Filter by similarity threshold and calculate relevance scores
            filtered_results = []
            for result in results:
                # Convert distance to similarity score (for L2: lower is better, for cosine: higher is better)
                distance = result.get("distance", float('inf'))
                
                # Calculate similarity score (1.0 = most similar, 0.0 = least similar)
                # For cosine similarity (Inner Product), distance is already similarity
                # For L2 distance, convert to similarity
                if distance <= 1.0:  # Likely cosine similarity
                    similarity = 1.0 - distance
                else:  # Likely L2 distance
                    similarity = 1.0 / (1.0 + distance)
                
                result["similarity"] = similarity
                result["relevance_score"] = similarity
                
                # Filter by threshold
                if similarity >= similarity_threshold:
                    filtered_results.append(result)
            
            # Sort by relevance (highest first)
            filtered_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            
            # Log retrieval details
            logger.info(f"=== RETRIEVAL TRACE ===")
            logger.info(f"Query: {query[:100]}...")
            logger.info(f"Total results from vector store: {len(results)}")
            logger.info(f"Filtered results (threshold={similarity_threshold}): {len(filtered_results)}")
            for idx, result in enumerate(filtered_results, 1):
                logger.info(f"  [Rank {idx}] ID: {result.get('id', 'unknown')}, "
                          f"Similarity: {result.get('similarity', 0.0):.4f}, "
                          f"Distance: {result.get('distance', 0.0):.4f}")
            logger.info(f"=== END RETRIEVAL TRACE ===")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise ValueError(f"Document retrieval failed: {e}")
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string for LLM.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            similarity = doc.get("similarity", 0.0)
            text = doc.get("document", "")
            
            # Format metadata
            metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items() if k not in ["chunk_index", "start_char", "end_char"]])
            
            context_parts.append(
                f"[Document {i} (relevance: {similarity:.3f})]\n"
                f"Metadata: {metadata_str}\n"
                f"Content: {text}"
            )
        
        return "\n\n---\n\n".join(context_parts)



