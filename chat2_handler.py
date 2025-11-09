"""
Main Handler Module for chat2 RAG Pipeline.
Orchestrates the complete RAG pipeline for /chat2 endpoint.

API Layout for /chat2 endpoint:
================================

POST /chat2
Parameters:
  - select: str (required) - Operation type: "input", "query", or "data"
  - message: str (required) - Content/text/chunk_id based on operation
  - file: UploadFile (optional) - PDF file (for select="input")
  - model: str (optional) - Model selection: "ollama", "gemma3:12b", or None (Gemini default)
  - operation: str (optional) - For select="data": "get", "delete", or "similarity"

Operations:
-----------

1. select="input":
   Pipeline: chat2_ingest.py -> chat2_embedder.py -> chat2_vectorstore.py
   - Reads and cleans document (PDF or text)
   - Chunks document with overlap
   - Generates embeddings
   - Extracts metadata
   - Stores in ChromaDB + FAISS
   Response: Success message with chunk count

2. select="query":
   Pipeline: chat2_retriever.py -> chat2_generator.py
   - Embeds user query
   - Retrieves top-k relevant chunks using FAISS/ChromaDB
   - Generates answer using LLM with retrieved context
   Response: Answer with source citations

3. select="data":
   Operations:
   - operation="get" + message="all": Get all chunks
   - operation="get" + message="chunk_id": Get specific chunk
   - operation="delete" + message="all": Delete all chunks
   - operation="delete" + message="chunk_id": Delete specific chunk
   - operation="similarity" + message="query text": Find top 20 most similar chunks to query
   Response: Chunk data, deletion confirmation, or similarity search results
"""
import logging
from typing import Optional
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse

from chat2_ingest import DocumentIngester
from chat2_embedder import DocumentEmbedder
from chat2_vectorstore import VectorStore
from chat2_retriever import DocumentRetriever
from chat2_generator import AnswerGenerator
from chat2_config import TOP_K_RESULTS, LOG_LEVEL, MAX_FILE_SIZE_BYTES, SIMILARITY_TOP_K

# Initialize logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("chat2_handler")

# Initialize components (singleton pattern)
_ingester = None
_embedder = None
_vector_store = None

def get_ingester() -> DocumentIngester:
    """Get or create DocumentIngester instance."""
    global _ingester
    if _ingester is None:
        _ingester = DocumentIngester()
    return _ingester

def get_embedder() -> DocumentEmbedder:
    """Get or create DocumentEmbedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = DocumentEmbedder()
    return _embedder

def get_vector_store() -> VectorStore:
    """Get or create VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

async def process_input_request(
    message: str,
    file: Optional[UploadFile],
    gemini_api_key: Optional[str],
    model_preference: Optional[str] = None
) -> JSONResponse:
    """
    Process input request: ingest, embed, and store documents.
    
    Pipeline: chat2_ingest.py -> chat2_embedder.py -> chat2_vectorstore.py
    """
    try:
        # Initialize components
        ingester = get_ingester()
        embedder = get_embedder()
        vector_store = get_vector_store()
        
        # Extract text from PDF or use provided text
        pdf_bytes = None
        filename = None
        
        if file:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
            # Check file size before reading
            # Note: We need to read the file to check size, but we'll validate after
            pdf_bytes = await file.read()
            
            # Validate file size
            if len(pdf_bytes) > MAX_FILE_SIZE_BYTES:
                max_size_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"File size ({len(pdf_bytes) / (1024 * 1024):.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
                )
            
            filename = file.filename
            logger.info(f"Processing PDF file: {filename} ({len(pdf_bytes) / (1024 * 1024):.2f} MB)")
        
        # Ingest document (read, clean, chunk)
        logger.info("Step 1: Document ingestion (read, clean, chunk)")
        chunks = ingester.process_document(
            text=message if not pdf_bytes else None,
            pdf_bytes=pdf_bytes,
            filename=filename
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from document")
        
        # Generate embeddings and extract metadata
        logger.info("Step 2: Embedding generation and metadata extraction")
        embeddings, metadatas = embedder.process_chunks(chunks)
        
        if len(embeddings) != len(chunks):
            raise HTTPException(status_code=500, detail="Embedding generation failed")
        
        # Store in vector database
        logger.info("Step 3: Vector store (ChromaDB + FAISS)")
        texts = [chunk["text"] for chunk in chunks]
        ids = vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully processed and stored {len(ids)} chunks")
        
        return JSONResponse({
            "status": "success",
            "message": "Document processed and stored successfully",
            "ingested_chunks": len(ids),
            "chunks_count": len(chunks)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Input processing failed")
        raise HTTPException(status_code=500, detail=f"Input processing failed: {e}")

async def process_query_request(
    message: str,
    gemini_api_key: Optional[str],
    model_preference: Optional[str] = None
) -> JSONResponse:
    """
    Process query request: retrieve context and generate answer.
    
    Pipeline: chat2_retriever.py -> chat2_generator.py
    """
    try:
        if not message:
            raise HTTPException(status_code=400, detail="No query provided in message")
        
        # Initialize components
        embedder = get_embedder()
        vector_store = get_vector_store()
        retriever = DocumentRetriever(embedder, vector_store)
        generator = AnswerGenerator(model_preference, gemini_api_key)
        
        # Retrieve relevant documents
        logger.info("Step 1: Document retrieval")
        retrieved_docs = retriever.retrieve(
            query=message,
            top_k=TOP_K_RESULTS
        )
        
        if not retrieved_docs:
            return JSONResponse({
                "status": "no_results",
                "message": "No relevant context found in the database",
                "answer": "I couldn't find any relevant information in the stored documents to answer your question.",
                "query": message
            })
        
        # Generate answer with sources
        logger.info("Step 2: Answer generation")
        result = generator.generate_answer_with_sources(
            query=message,
            retrieved_docs=retrieved_docs
        )
        
        return JSONResponse({
            "status": "success",
            "query": message,
            "answer": result["answer"],
            "retrieved_chunks": len(retrieved_docs),
            "sources": result["sources"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

async def process_data_request(message: str, operation: Optional[str] = None) -> JSONResponse:
    """
    Process data request: get, delete, or similarity search chunks from ChromaDB.
    
    Operations:
    - "get": Get all chunks or specific chunk by ID
    - "delete": Delete all chunks or specific chunk by ID
    - "similarity": Find top 20 most similar chunks to the query message
    """
    if not operation:
        raise HTTPException(status_code=400, detail="'operation' parameter is required for data select (should be 'get', 'delete', or 'similarity')")
    
    if not message:
        raise HTTPException(status_code=400, detail="'message' parameter is required")
    
    try:
        vector_store = get_vector_store()
        operation = operation.lower().strip()
        text_value = message.strip()
        
        if operation == "get":
            if text_value.lower() == "all":
                # Get all chunks
                documents = vector_store.get_all_documents()
                return JSONResponse({
                    "status": "success",
                    "operation": "get",
                    "count": len(documents),
                    "chunks": documents
                })
            else:
                # Get specific chunk
                document = vector_store.get_document(text_value)
                if not document:
                    return JSONResponse({
                        "status": "not_found",
                        "message": f"Chunk with ID '{text_value}' not found",
                        "chunk_id": text_value
                    })
                return JSONResponse({
                    "status": "success",
                    "operation": "get",
                    "chunk": document
                })
        
        elif operation == "delete":
            if text_value.lower() == "all":
                # Delete all chunks
                deleted_count = vector_store.delete_all()
                return JSONResponse({
                    "status": "success",
                    "operation": "delete",
                    "message": f"Deleted all {deleted_count} chunks",
                    "deleted_count": deleted_count
                })
            else:
                # Delete specific chunk
                deleted_count = vector_store.delete_documents([text_value])
                if deleted_count == 0:
                    return JSONResponse({
                        "status": "not_found",
                        "message": f"Chunk with ID '{text_value}' not found",
                        "chunk_id": text_value
                    })
                return JSONResponse({
                    "status": "success",
                    "operation": "delete",
                    "message": f"Deleted chunk '{text_value}'",
                    "deleted_chunk_id": text_value
                })
        
        elif operation == "similarity":
            # Similarity search: find top 20 most similar chunks
            if not text_value:
                raise HTTPException(status_code=400, detail="'message' parameter is required for similarity operation (should contain the query text)")
            
            # Initialize retriever for similarity search
            embedder = get_embedder()
            retriever = DocumentRetriever(embedder, vector_store)
            
            # Retrieve top 20 similar chunks
            logger.info(f"Performing similarity search for query: {text_value[:100]}...")
            retrieved_docs = retriever.retrieve(
                query=text_value,
                top_k=SIMILARITY_TOP_K,
                similarity_threshold=0.0  # No threshold for similarity search, get top 20
            )
            
            if not retrieved_docs:
                return JSONResponse({
                    "status": "no_results",
                    "operation": "similarity",
                    "message": "No chunks found in the database",
                    "query": text_value,
                    "results": []
                })
            
            # Format results with similarity scores
            results = []
            for i, doc in enumerate(retrieved_docs, 1):
                results.append({
                    "rank": i,
                    "id": doc.get("id"),
                    "document": doc.get("document", ""),
                    "metadata": doc.get("metadata", {}),
                    "similarity_score": doc.get("similarity", 0.0),
                    "relevance_score": doc.get("relevance_score", 0.0),
                    "distance": doc.get("distance", 0.0)
                })
            
            logger.info(f"Similarity search returned {len(results)} results")
            return JSONResponse({
                "status": "success",
                "operation": "similarity",
                "query": text_value,
                "total_results": len(results),
                "results": results
            })
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation '{operation}'. Supported operations: 'get', 'delete', 'similarity'"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Data operation failed")
        raise HTTPException(status_code=500, detail=f"Data operation failed: {e}")

async def handle_chat2_request(
    select: str,
    message: str,
    file: Optional[UploadFile],
    gemini_api_key: Optional[str],
    model_preference: Optional[str] = None,
    operation: Optional[str] = None
) -> JSONResponse:
    """
    Main handler function for /chat2 endpoint.
    Routes to input processing, query processing, or data operations.
    """
    if not select:
        raise HTTPException(status_code=400, detail="'select' parameter is required (should be 'input', 'query', or 'data')")
    
    if not message:
        raise HTTPException(status_code=400, detail="'message' parameter is required")
    
    select_lower = select.strip().lower()
    
    if select_lower == "input":
        return await process_input_request(message, file, gemini_api_key, model_preference)
    elif select_lower == "query":
        return await process_query_request(message, gemini_api_key, model_preference)
    elif select_lower == "data":
        return await process_data_request(message, operation)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid 'select' value '{select}'. Supported values: 'input', 'query', 'data'"
        )

