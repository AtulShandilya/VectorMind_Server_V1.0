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
  - operation: str (optional) - For select="data": "get" or "delete"

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
   Operations: Same as chat1
   - operation="get" + message="all": Get all chunks
   - operation="get" + message="chunk_id": Get specific chunk
   - operation="delete" + message="all": Delete all chunks
   - operation="delete" + message="chunk_id": Delete specific chunk
   Response: Chunk data or deletion confirmation
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
from chat2_config import TOP_K_RESULTS, LOG_LEVEL

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
            pdf_bytes = await file.read()
            filename = file.filename
            logger.info(f"Processing PDF file: {filename} ({len(pdf_bytes)} bytes)")
        
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
    Process data request: get or delete chunks from ChromaDB.
    Same functionality as chat1.
    """
    if not operation:
        raise HTTPException(status_code=400, detail="'operation' parameter is required for data select (should be 'get' or 'delete')")
    
    if not message:
        raise HTTPException(status_code=400, detail="'message' parameter is required (should be 'all' or a chunk ID)")
    
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
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation '{operation}'. Supported operations: 'get', 'delete'"
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

