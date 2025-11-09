"""
Vector Store Module for chat2 RAG Pipeline.
Handles storage and retrieval using ChromaDB and FAISS for similarity search.
"""
import os
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
import chromadb
import numpy as np
from chat2_config import (
    CHROMA_DIR, 
    CHROMA_COLLECTION_NAME, 
    FAISS_INDEX_PATH,
    FAISS_INDEX_TYPE
)

logger = logging.getLogger("chat2_vectorstore")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu or pip install faiss-gpu")

class VectorStore:
    """Manages vector storage using ChromaDB and FAISS."""
    
    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME, persist_dir: str = CHROMA_DIR):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory for persisting ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.faiss_index = None
        self.faiss_ids = []
        self.embedding_dim = None
        
        # Initialize ChromaDB
        try:
            os.makedirs(persist_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=collection_name)
                logger.info(f"Using existing ChromaDB collection: {collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Chat2 RAG pipeline collection"}
                )
                logger.info(f"Created new ChromaDB collection: {collection_name}")
            
            # Load or initialize FAISS index
            self._load_faiss_index()
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metadata values to ChromaDB-compatible types.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = None
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = ", ".join(str(v) for v in value) if value else ""
            elif isinstance(value, dict):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = str(value)
        return sanitized
    
    def _load_faiss_index(self):
        """Load FAISS index from disk if available."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping FAISS index loading")
            return
        
        index_path = os.path.join(FAISS_INDEX_PATH, "faiss.index")
        ids_path = os.path.join(FAISS_INDEX_PATH, "faiss_ids.json")
        
        try:
            if os.path.exists(index_path) and os.path.exists(ids_path):
                self.faiss_index = faiss.read_index(index_path)
                with open(ids_path, 'r') as f:
                    self.faiss_ids = json.load(f)
                self.embedding_dim = self.faiss_index.d
                logger.info(f"Loaded FAISS index with {len(self.faiss_ids)} vectors (dim={self.embedding_dim})")
            else:
                logger.info("No existing FAISS index found, will create new one")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            self.faiss_index = None
    
    def _save_faiss_index(self):
        """Save FAISS index to disk."""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        try:
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
            index_path = os.path.join(FAISS_INDEX_PATH, "faiss.index")
            ids_path = os.path.join(FAISS_INDEX_PATH, "faiss_ids.json")
            
            faiss.write_index(self.faiss_index, index_path)
            with open(ids_path, 'w') as f:
                json.dump(self.faiss_ids, f)
            
            logger.info(f"Saved FAISS index with {len(self.faiss_ids)} vectors")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _create_faiss_index(self, embedding_dim: int):
        """
        Create a new FAISS index.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        if not FAISS_AVAILABLE:
            return
        
        try:
            # Use L2 distance (Euclidean) or Inner Product for cosine similarity
            if FAISS_INDEX_TYPE == "COSINE":
                # For cosine similarity, use Inner Product on normalized vectors
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            else:
                # Default to L2 distance
                self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            
            self.embedding_dim = embedding_dim
            self.faiss_ids = []
            logger.info(f"Created new FAISS index (type={FAISS_INDEX_TYPE}, dim={embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            self.faiss_index = None
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs (will be generated if not provided)
            
        Returns:
            List of document IDs
        """
        if not texts or not embeddings:
            raise ValueError("Texts and embeddings must be provided")
        
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("Texts, embeddings, and metadatas must have the same length")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Sanitize metadata
        sanitized_metadatas = [self._sanitize_metadata(meta) for meta in metadatas]
        
        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=sanitized_metadatas,
                embeddings=embeddings
            )
            logger.info(f"Added {len(ids)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
        
        # Add to FAISS index
        if FAISS_AVAILABLE and embeddings:
            try:
                embedding_dim = len(embeddings[0])
                
                # Initialize FAISS index if needed
                if self.faiss_index is None:
                    self._create_faiss_index(embedding_dim)
                
                # Convert embeddings to numpy array
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                # Normalize for cosine similarity if using Inner Product
                if FAISS_INDEX_TYPE == "COSINE":
                    faiss.normalize_L2(embeddings_array)
                
                # Add to FAISS index
                self.faiss_index.add(embeddings_array)
                self.faiss_ids.extend(ids)
                
                logger.info(f"Added {len(ids)} vectors to FAISS index")
                
                # Save FAISS index
                self._save_faiss_index()
            except Exception as e:
                logger.warning(f"Failed to add to FAISS index: {e}")
        
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using FAISS and ChromaDB.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of result dictionaries with document, metadata, and distance
        """
        results = []
        
        # Use FAISS for fast similarity search if available
        if FAISS_AVAILABLE and self.faiss_index is not None:
            try:
                query_array = np.array([query_embedding], dtype=np.float32)
                
                # Normalize for cosine similarity if using Inner Product
                if FAISS_INDEX_TYPE == "COSINE":
                    faiss.normalize_L2(query_array)
                
                # Search in FAISS
                k = min(top_k, len(self.faiss_ids))
                distances, indices = self.faiss_index.search(query_array, k)
                
                # Get IDs from FAISS results
                faiss_result_ids = [self.faiss_ids[idx] for idx in indices[0] if idx < len(self.faiss_ids)]
                
                # Retrieve full documents from ChromaDB
                if faiss_result_ids:
                    chroma_results = self.collection.get(
                        ids=faiss_result_ids,
                        include=["documents", "metadatas"]
                    )
                    
                    # Combine results
                    for i, (doc_id, distance) in enumerate(zip(faiss_result_ids, distances[0])):
                        if doc_id in chroma_results.get("ids", []):
                            idx = chroma_results["ids"].index(doc_id)
                            results.append({
                                "id": doc_id,
                                "document": chroma_results["documents"][idx],
                                "metadata": chroma_results["metadatas"][idx],
                                "distance": float(distance)
                            })
                
                logger.info(f"FAISS search returned {len(results)} results")
            except Exception as e:
                logger.warning(f"FAISS search failed, falling back to ChromaDB: {e}")
        
        # Fallback to ChromaDB search if FAISS failed or not available
        if not results:
            try:
                chroma_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter_metadata,
                    include=["documents", "metadatas", "distances"]
                )
                
                if chroma_results.get("ids") and chroma_results["ids"][0]:
                    for i, doc_id in enumerate(chroma_results["ids"][0]):
                        results.append({
                            "id": doc_id,
                            "document": chroma_results["documents"][0][i],
                            "metadata": chroma_results["metadatas"][0][i],
                            "distance": chroma_results["distances"][0][i] if chroma_results.get("distances") else 0.0
                        })
                
                logger.info(f"ChromaDB search returned {len(results)} results")
            except Exception as e:
                logger.error(f"ChromaDB search failed: {e}")
                raise
        
        return results
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the vector store."""
        try:
            results = self.collection.get(include=["documents", "metadatas"])
            documents = []
            for i, doc_id in enumerate(results.get("ids", [])):
                documents.append({
                    "id": doc_id,
                    "document": results["documents"][i] if i < len(results["documents"]) else "",
                    "metadata": results["metadatas"][i] if i < len(results["metadatas"]) else {}
                })
            return documents
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
            if results.get("ids"):
                return {
                    "id": results["ids"][0],
                    "document": results["documents"][0] if results.get("documents") else "",
                    "metadata": results["metadatas"][0] if results.get("metadatas") else {}
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def delete_documents(self, ids: List[str]) -> int:
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=ids)
            
            # Remove from FAISS index
            if FAISS_AVAILABLE and self.faiss_index is not None:
                # Rebuild FAISS index without deleted IDs
                remaining_ids = [fid for fid in self.faiss_ids if fid not in ids]
                if len(remaining_ids) < len(self.faiss_ids):
                    # Get remaining embeddings from ChromaDB and rebuild
                    remaining_docs = self.collection.get(ids=remaining_ids, include=["embeddings"])
                    if remaining_docs.get("embeddings"):
                        embeddings = remaining_docs["embeddings"]
                        embedding_dim = len(embeddings[0])
                        self._create_faiss_index(embedding_dim)
                        embeddings_array = np.array(embeddings, dtype=np.float32)
                        if FAISS_INDEX_TYPE == "COSINE":
                            faiss.normalize_L2(embeddings_array)
                        self.faiss_index.add(embeddings_array)
                        self.faiss_ids = remaining_ids
                        self._save_faiss_index()
            
            logger.info(f"Deleted {len(ids)} documents")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def delete_all(self) -> int:
        """Delete all documents."""
        try:
            all_docs = self.get_all_documents()
            if all_docs:
                ids = [doc["id"] for doc in all_docs]
                return self.delete_documents(ids)
            return 0
        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            raise

