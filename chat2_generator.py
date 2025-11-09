"""
Generator Module for chat2 RAG Pipeline.
Uses retrieved context to generate final answer via LLM.
"""
import logging
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI
from chat2_config import (
    GEMINI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    GEMINI_BASE_URL,
    DEFAULT_TEMPERATURE
)

logger = logging.getLogger("chat2_generator")

class AnswerGenerator:
    """Generates answers using LLM with retrieved context."""
    
    def __init__(self, model_preference: Optional[str] = None, gemini_api_key: Optional[str] = None):
        """
        Initialize the answer generator.
        
        Args:
            model_preference: Model preference ("ollama", "gemma3:12b", or None for Gemini)
            gemini_api_key: API key for Gemini (required for Gemini model)
        """
        self.model_preference = model_preference
        self.gemini_api_key = gemini_api_key
        self.client, self.model_name = self._get_llm_client()
        logger.info(f"Initialized AnswerGenerator with model: {self.model_name}")
    
    def _get_llm_client(self) -> Tuple[OpenAI, str]:
        """Get the appropriate LLM client based on model preference."""
        use_ollama = False
        if self.model_preference:
            model_pref_lower = self.model_preference.lower().strip()
            if model_pref_lower in ["ollama", "gemma3:12b"]:
                use_ollama = True
        
        if use_ollama:
            logger.info(f"Using Ollama model: {OLLAMA_MODEL}")
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            return client, OLLAMA_MODEL
        else:
            if not self.gemini_api_key:
                raise ValueError("Gemini API key is required when using Gemini model")
            logger.info(f"Using Gemini model: {GEMINI_MODEL}")
            client = OpenAI(base_url=GEMINI_BASE_URL, api_key=self.gemini_api_key)
            return client, GEMINI_MODEL
    
    def generate_answer(
        self,
        query: str,
        context: str,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User query
            context: Retrieved context documents
            temperature: LLM temperature (0.0 to 1.0)
            system_prompt: Optional system prompt
            
        Returns:
            Generated answer string
        """
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Build prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Answer the question using ONLY the information from the context.
If the context doesn't contain enough information, say so explicitly.
Be concise, accurate, and cite which document(s) you used when relevant."""
        
        user_prompt = f"""Context Documents:
{context}

User Question: {query}

Please provide a clear and accurate answer based on the context provided above."""
        
        try:
            logger.info(f"Generating answer for query: {query[:100]}...")
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise ValueError(f"LLM call failed: {e}")
    
    def generate_answer_with_sources(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE
    ) -> Dict[str, Any]:
        """
        Generate answer with source information.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document dictionaries
            temperature: LLM temperature
            
        Returns:
            Dictionary with answer and source information
        """
        if not retrieved_docs:
            logger.warning("No retrieved documents provided for answer generation")
            return {
                "answer": "I couldn't find any relevant information in the stored documents to answer your question.",
                "sources": []
            }
        
        # Log retrieved documents info
        logger.info(f"=== DOCUMENT FORMATTING TRACE ===")
        logger.info(f"Total retrieved documents: {len(retrieved_docs)}")
        logger.info(f"Query: {query[:100]}...")
        
        # Format context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get("id", "unknown")
            metadata = doc.get("metadata", {})
            similarity = doc.get("similarity", 0.0)
            relevance_score = doc.get("relevance_score", 0.0)
            distance = doc.get("distance", 0.0)
            text = doc.get("document", "")
            text_preview = text[:100] + "..." if len(text) > 100 else text
            
            # Log each document being formatted
            logger.info(f"  [Rank {i}] Document ID: {doc_id}")
            logger.info(f"    - Similarity: {similarity:.4f}")
            logger.info(f"    - Relevance Score: {relevance_score:.4f}")
            logger.info(f"    - Distance: {distance:.4f}")
            logger.info(f"    - Text Preview: {text_preview}")
            logger.info(f"    - Metadata: {metadata}")
            
            metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items() if k not in ["chunk_index", "start_char", "end_char"]])
            context_parts.append(
                f"[Document {i} (relevance: {similarity:.3f})]\n"
                f"Metadata: {metadata_str}\n"
                f"Content: {text}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Log context summary
        logger.info(f"=== CONTEXT SUMMARY ===")
        logger.info(f"Total context length: {len(context)} characters")
        logger.info(f"Number of document parts: {len(context_parts)}")
        logger.info(f"Context preview (first 500 chars): {context[:500]}...")
        logger.info(f"=== END DOCUMENT FORMATTING TRACE ===")
        
        # Generate answer
        answer = self.generate_answer(query, context, temperature)
        
        # Prepare sources - include all chunks that were sent to LLM
        sources = []
        for i, doc in enumerate(retrieved_docs, 1):
            sources.append({
                "rank": i,
                "source_index": i,
                "id": doc.get("id"),
                "document": doc.get("document", ""),  # Full document text that was sent to LLM
                "metadata": doc.get("metadata", {}),
                "relevance_score": doc.get("relevance_score", 0.0),
                "similarity": doc.get("similarity", 0.0),
                "distance": doc.get("distance", 0.0)
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }

