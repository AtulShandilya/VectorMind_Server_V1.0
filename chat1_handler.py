import os
import io
import json
import uuid
import logging
import base64
from typing import Optional, List, Dict, Any
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------- Configuration ----------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "chat1_collection")
CHUNK_CHAR_SIZE = 2000
CHUNK_OVERLAP_SIZE = 300

# Model configuration
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat1_handler")

# ---------- Initialize embedding model (lightweight for 2GB RAM) ----------
# Using a small, efficient model that works well on constrained systems
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    raise

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

def get_llm_client(model_preference: Optional[str] = None, gemini_api_key: Optional[str] = None) -> tuple:
    """
    Get the appropriate OpenAI client and model name based on model preference.
    
    Args:
        model_preference: Model preference from input ("ollama", "gemma3:12b", or None/empty for Gemini default)
        gemini_api_key: API key for Gemini (required for Gemini model)
    
    Returns:
        Tuple of (OpenAI client, model name)
    """
    # Normalize model preference - check if it's explicitly Ollama
    use_ollama = False
    if model_preference:
        model_pref_lower = model_preference.lower().strip()
        if model_pref_lower in ["ollama", "gemma3:12b"]:
            use_ollama = True
    
    if use_ollama:
        # Use Ollama
        logger.info(f"Using Ollama model: {OLLAMA_MODEL}")
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")  # Ollama doesn't require API key
        return client, OLLAMA_MODEL
    else:
        # Default: Use Gemini (gemini-2.5-flash-preview-05-20)
        if not gemini_api_key:
            raise HTTPException(status_code=500, detail="Gemini API key is required when using Gemini model")
        logger.info(f"Using default Gemini model: {GEMINI_MODEL}")
        client = OpenAI(base_url=GEMINI_BASE_URL, api_key=gemini_api_key)
        return client, GEMINI_MODEL

# ---------- Initialize ChromaDB ----------
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Get or create collection
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        logger.info(f"Using existing collection: {CHROMA_COLLECTION_NAME}")
    except Exception:
        collection = chroma_client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "Chat1 vector database collection"}
        )
        logger.info(f"Created new collection: {CHROMA_COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    raise

# ---------- Utility functions ----------

def sanitize_metadata_for_chromadb(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert metadata values to ChromaDB-compatible types.
    ChromaDB only accepts: str, int, float, bool, SparseVector, or None.
    Lists and dicts are converted to JSON strings.
    """
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = None
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            # Convert list to comma-separated string or JSON string
            sanitized[key] = ", ".join(str(v) for v in value) if value else ""
        elif isinstance(value, dict):
            # Convert dict to JSON string
            sanitized[key] = json.dumps(value)
        else:
            # Convert any other type to string
            sanitized[key] = str(value)
    return sanitized

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    pages.append(text)
            except Exception as e:
                logger.warning(f"Error extracting text from page: {e}")
                pages.append("")
        return "\n".join(pages)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")

async def call_llm(
    prompt: str, 
    client: OpenAI, 
    model: str,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    mime_type: Optional[str] = None,
    extracted_text: Optional[str] = None
) -> str:
    """
    Call Gemini LLM via OpenAI-compatible API with optional file attachment.
    Note: OpenAI-compatible endpoint doesn't support file attachments directly.
    For PDFs, we extract text and send it. For images, we use inline data format.
    """
    try:
        # Determine MIME type if not provided
        if file_bytes and file_name and not mime_type:
            if file_name.lower().endswith('.pdf'):
                mime_type = "application/pdf"
            elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                mime_type = f"image/{file_name.split('.')[-1].lower()}"
            else:
                mime_type = "application/octet-stream"
        
        # Build message content
        if file_bytes and file_name:
            # For images, use inline data format
            if mime_type.startswith("image/"):
                # Encode image to base64
                file_base64 = base64.b64encode(file_bytes).decode('utf-8')
                
                # Use image_url format for images
                content_parts = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{file_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
                messages = [{"role": "user", "content": content_parts}]
            else:
                # For PDFs and other documents, use extracted text
                # The extracted_text should be provided from process_input_request
                if extracted_text:
                    # Include extracted text in the prompt
                    combined_prompt = f"{prompt}\n\nDocument content:\n{extracted_text}"
                else:
                    # Fallback: extract text if not provided
                    if mime_type == "application/pdf":
                        extracted_text = extract_text_from_pdf_bytes(file_bytes)
                        combined_prompt = f"{prompt}\n\nDocument content:\n{extracted_text}"
                    else:
                        combined_prompt = prompt
                
                messages = [{"role": "user", "content": combined_prompt}]
        else:
            # Text-only message
            messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        # Log more details for debugging
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"API response: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

async def process_input_request(
    message: str,
    file: Optional[UploadFile],
    gemini_api_key: Optional[str],
    model_preference: Optional[str] = None
) -> JSONResponse:
    """
    Process input request: send file/text directly to LLM, summarize and chunk, store in ChromaDB.
    """
    # Try to parse as JSON to extract model preference
    model_pref = model_preference
    input_text = message
    
    try:
        # Try parsing as JSON if message is JSON
        if message.strip().startswith("{") and "model" in message:
            parsed = json.loads(message)
            if "model" in parsed:
                model_pref = parsed.get("model")
                # Remove model from input text if it was in JSON
                input_text = parsed.get("text", message)
                logger.info(f"Model preference from input: {model_pref}")
    except (json.JSONDecodeError, KeyError):
        # Not JSON or doesn't have model key, use as-is
        pass
    
    # Get appropriate LLM client based on model preference
    client, model_name = get_llm_client(model_pref, gemini_api_key)
    
    file_bytes = None
    file_name = None
    has_file = False
    extracted_text = None
    
    if file:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_bytes = await file.read()
        file_name = file.filename
        has_file = True
        
        # Extract text from PDF to send to LLM
        # (OpenAI-compatible endpoint doesn't support direct file uploads)
        extracted_text = extract_text_from_pdf_bytes(file_bytes)
        logger.info(f"Extracted text from PDF for LLM processing: {file_name} ({len(extracted_text)} characters)")
        
        # If text is also provided, it will be included in the prompt
        if input_text:
            logger.info(f"Additional text input provided: {len(input_text)} characters")
    else:
        # Use text input only (text after 'input' prefix)
        if not input_text:
            raise HTTPException(status_code=400, detail="No text provided after 'input' prefix and no file uploaded")
        logger.info(f"Using text input only: {len(input_text)} characters")
    
    # Prepare prompt for LLM to summarize and chunk
    if has_file:
        # When file is provided, the LLM will process it directly
        additional_context = f"\n\nAdditional context or instructions: {input_text}" if input_text else ""
        llm_prompt = f"""Please analyze the attached document and:
1. Summarize the document without losing any important information
2. Divide the summary into chunks, each approximately {CHUNK_CHAR_SIZE} characters based on context (not just arbitrary splits - maintain semantic coherence)
3. IMPORTANT: Ensure each chunk overlaps with the previous chunk by approximately {CHUNK_OVERLAP_SIZE} characters at the beginning. This overlap ensures context continuity and prevents information from being cut off between chunks.
4. For each chunk, provide metadata that would be useful for retrieval (e.g., topics, key concepts, document section, etc.)

Example of overlap:
- Chunk 1: characters 0-2000
- Chunk 2: characters 1700-3700 (overlaps with chunk 1 by 300 chars from 1700-2000)
- Chunk 3: characters 3400-5400 (overlaps with chunk 2 by 300 chars from 3400-3700)

Return your response as a JSON object with this exact structure:
{{
    "document_summary": "A comprehensive summary of the entire document",
    "chunks": [
        {{
            "id": "unique_chunk_id_1",
            "text": "chunk text content (approximately {CHUNK_CHAR_SIZE} chars)",
            "metadata": {{
                "topic": "main topic",
                "key_concepts": ["concept1", "concept2"],
                "section": "section name if applicable",
                "chunk_index": 1
            }}
        }},
        {{
            "id": "unique_chunk_id_2",
            "text": "next chunk text content (should overlap with previous chunk by ~{CHUNK_OVERLAP_SIZE} chars at the start)",
            "metadata": {{
                "topic": "another topic",
                "key_concepts": ["concept3"],
                "section": "another section",
                "chunk_index": 2
            }}
        }}
    ]
}}{additional_context}"""
    else:
        # When only text is provided
        text_to_process = input_text[:50000]  # Limit to 50k chars to avoid token limits
        llm_prompt = f"""Please analyze the following text and:
1. Summarize the text without losing any important information
2. Divide the summary into chunks, each approximately {CHUNK_CHAR_SIZE} characters based on context (not just arbitrary splits - maintain semantic coherence)
3. IMPORTANT: Ensure each chunk overlaps with the previous chunk by approximately {CHUNK_OVERLAP_SIZE} characters at the beginning. This overlap ensures context continuity and prevents information from being cut off between chunks.
4. For each chunk, provide metadata that would be useful for retrieval (e.g., topics, key concepts, document section, etc.)

Example of overlap:
- Chunk 1: characters 0-2000
- Chunk 2: characters 1700-3700 (overlaps with chunk 1 by 300 chars from 1700-2000)
- Chunk 3: characters 3400-5400 (overlaps with chunk 2 by 300 chars from 3400-3700)

Return your response as a JSON object with this exact structure:
{{
    "document_summary": "A comprehensive summary of the entire document",
    "chunks": [
        {{
            "id": "unique_chunk_id_1",
            "text": "chunk text content (approximately {CHUNK_CHAR_SIZE} chars)",
            "metadata": {{
                "topic": "main topic",
                "key_concepts": ["concept1", "concept2"],
                "section": "section name if applicable",
                "chunk_index": 1
            }}
        }},
        {{
            "id": "unique_chunk_id_2",
            "text": "next chunk text content (should overlap with previous chunk by ~{CHUNK_OVERLAP_SIZE} chars at the start)",
            "metadata": {{
                "topic": "another topic",
                "key_concepts": ["concept3"],
                "section": "another section",
                "chunk_index": 2
            }}
        }}
    ]
}}

Text to process:
{text_to_process}"""
    
    # Call LLM for summarization and chunking
    try:
        if has_file:
            # Send file to LLM (extracted text will be used since OpenAI-compatible endpoint 
            # doesn't support direct file uploads)
            llm_response = await call_llm(
                llm_prompt, 
                client,
                model_name,
                file_bytes=file_bytes,
                file_name=file_name,
                mime_type="application/pdf",
                extracted_text=extracted_text
            )
        else:
            llm_response = await call_llm(llm_prompt, client, model_name)
        logger.info("Received response from LLM")
    except Exception as e:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")
    
    # Parse LLM response as JSON
    try:
        # Try to extract JSON from response (LLM might wrap it in markdown)
        llm_response_clean = llm_response.strip()
        if llm_response_clean.startswith("```json"):
            llm_response_clean = llm_response_clean[7:]
        if llm_response_clean.startswith("```"):
            llm_response_clean = llm_response_clean[3:]
        if llm_response_clean.endswith("```"):
            llm_response_clean = llm_response_clean[:-3]
        llm_response_clean = llm_response_clean.strip()
        
        parsed = json.loads(llm_response_clean)
        chunks = parsed.get("chunks", [])
        doc_summary = parsed.get("document_summary", "")
        
        if not chunks:
            raise ValueError("No chunks returned from LLM")
            
    except json.JSONDecodeError as e:
        logger.exception(f"Failed to parse LLM response as JSON. Raw response: {llm_response[:500]}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM output as JSON. Please ensure the LLM returns valid JSON format."
        )
    except Exception as e:
        logger.exception(f"Error processing LLM response: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing LLM response: {e}")
    
    # Prepare data for ChromaDB storage
    ids = []
    metadatas = []
    documents = []
    texts_for_embedding = []
    
    for chunk in chunks:
        chunk_id = chunk.get("id") or str(uuid.uuid4())
        chunk_text = chunk.get("text", "").strip()
        chunk_metadata = chunk.get("metadata", {})
        
        if not chunk_text:
            logger.warning(f"Skipping empty chunk with id: {chunk_id}")
            continue
        
        # Add file metadata if available
        if file:
            chunk_metadata["source_file"] = file.filename
        chunk_metadata["chunk_id"] = chunk_id
        
        # Sanitize metadata to ensure ChromaDB compatibility (convert lists/dicts to strings)
        sanitized_metadata = sanitize_metadata_for_chromadb(chunk_metadata)
        
        ids.append(chunk_id)
        documents.append(chunk_text)
        metadatas.append(sanitized_metadata)
        texts_for_embedding.append(chunk_text)
    
    if not ids:
        raise HTTPException(status_code=500, detail="No valid chunks to store after processing")
    
    # Generate embeddings
    try:
        logger.info(f"Generating embeddings for {len(texts_for_embedding)} chunks...")
        embeddings = embed_texts(texts_for_embedding)
        logger.info("Embeddings generated successfully")
    except Exception as e:
        logger.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")
    
    # Store in ChromaDB
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"Successfully stored {len(ids)} chunks in ChromaDB")
    except Exception as e:
        logger.exception("ChromaDB storage failed")
        raise HTTPException(status_code=500, detail=f"ChromaDB storage failed: {e}")
    
    return JSONResponse({
        "status": "success",
        "message": "Document processed and stored successfully",
        "ingested_chunks": len(ids),
        "document_summary": doc_summary,
        "chunks_count": len(chunks)
    })

async def process_query_request(
    message: str,
    gemini_api_key: Optional[str],
    model_preference: Optional[str] = None
) -> JSONResponse:
    """
    Process query request: retrieve relevant context from ChromaDB and get answer from LLM.
    """
    # Try to parse as JSON to extract model preference
    model_pref = model_preference
    query_text = message
    
    try:
        # Try parsing as JSON if message is JSON
        if message.strip().startswith("{") and "model" in message:
            parsed = json.loads(message)
            if "model" in parsed:
                model_pref = parsed.get("model")
                # Remove model from query text if it was in JSON
                query_text = parsed.get("text", message)
                logger.info(f"Model preference from query: {model_pref}")
    except (json.JSONDecodeError, KeyError):
        # Not JSON or doesn't have model key, use as-is
        pass
    
    if not query_text:
        raise HTTPException(status_code=400, detail="No query provided in message")
    
    # Get appropriate LLM client based on model preference
    client, model_name = get_llm_client(model_pref, gemini_api_key)
    
    logger.info(f"Processing query: {query_text[:100]}...")
    
    # Generate embedding for query
    try:
        query_embeddings = embed_texts([query_text])
        query_emb = query_embeddings[0]
    except Exception as e:
        logger.exception("Query embedding generation failed")
        raise HTTPException(status_code=500, detail=f"Query embedding generation failed: {e}")
    
    # Retrieve relevant chunks from ChromaDB
    try:
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=5,  # Retrieve top 5 most relevant chunks
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.exception("ChromaDB query failed")
        raise HTTPException(status_code=500, detail=f"Vector database query failed: {e}")
    
    # Process retrieved results
    retrieved_docs = results.get("documents", [])[0] if results.get("documents") else []
    retrieved_metadatas = results.get("metadatas", [])[0] if results.get("metadatas") else []
    retrieved_distances = results.get("distances", [])[0] if results.get("distances") else []
    
    if not retrieved_docs:
        return JSONResponse({
            "status": "no_results",
            "message": "No relevant context found in the database",
            "answer": "I couldn't find any relevant information in the stored documents to answer your question."
        })
    
    # Combine retrieved documents into context
    context_parts = []
    for i, (doc, metadata, distance) in enumerate(zip(retrieved_docs, retrieved_metadatas, retrieved_distances)):
        metadata_str = json.dumps(metadata, indent=2)
        context_parts.append(f"[Chunk {i+1} (relevance score: {1-distance:.3f})]\nMetadata: {metadata_str}\nContent: {doc}")
    
    combined_context = "\n\n---\n\n".join(context_parts)
    
    # Build prompt for LLM
    llm_prompt = f"""You are given contextual documents retrieved from a vector database based on the user's question. 
Your task is to answer the user's question using ONLY the information provided in the context below.

IMPORTANT:
- Answer the question based strictly on the provided context
- If the context doesn't contain enough information to answer the question, say so explicitly
- Cite which chunk(s) you used when relevant
- Be concise and accurate

User Question: {query_text}

Context Documents:
{combined_context}

Please provide a clear and accurate answer based on the context provided above."""
    
    # Call LLM for answer
    try:
        answer = await call_llm(llm_prompt, client, model_name)
        logger.info("Received answer from LLM")
    except Exception as e:
        logger.exception("LLM call failed for query")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")
    
    return JSONResponse({
        "status": "success",
        "query": query_text,
        "answer": answer,
        "retrieved_chunks": len(retrieved_docs),
        "sources": [
            {
                "chunk_index": i + 1,
                "metadata": meta,
                "relevance_score": 1 - dist
            }
            for i, (meta, dist) in enumerate(zip(retrieved_metadatas, retrieved_distances))
        ]
    })

async def process_data_request(message: str, operation: Optional[str] = None) -> JSONResponse:
    """
    Process data request: get or delete chunks from ChromaDB.
    
    Args:
        message: "all" to get/delete all chunks, or a specific chunk ID
        operation: "get" or "delete" operation type
    """
    if not operation:
        raise HTTPException(status_code=400, detail="'operation' parameter is required for data select (should be 'get' or 'delete')")
    
    if not message:
        raise HTTPException(status_code=400, detail="'message' parameter is required (should be 'all' or a chunk ID)")
    
    operation = operation.lower().strip()
    text_value = message.strip()
    
    if operation == "get":
        # GET operation
        if text_value.lower() == "all":
            # Get all chunk IDs
            try:
                # Get all items from collection
                results = collection.get(include=["documents", "metadatas"])
                chunk_ids = results.get("ids", [])
                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])
                
                chunks_data = []
                for i, chunk_id in enumerate(chunk_ids):
                    chunks_data.append({
                        "id": chunk_id,
                        "document": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {}
                    })
                
                logger.info(f"Retrieved {len(chunk_ids)} chunks from ChromaDB")
                return JSONResponse({
                    "status": "success",
                    "operation": "get",
                    "count": len(chunk_ids),
                    "chunks": chunks_data
                })
            except Exception as e:
                logger.exception("Failed to get all chunks from ChromaDB")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {e}")
        else:
            # Get specific chunk by ID
            try:
                chunk_id = text_value
                results = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
                
                if not results.get("ids"):
                    return JSONResponse({
                        "status": "not_found",
                        "message": f"Chunk with ID '{chunk_id}' not found",
                        "chunk_id": chunk_id
                    })
                
                chunk_data = {
                    "id": results["ids"][0],
                    "document": results["documents"][0] if results.get("documents") else "",
                    "metadata": results["metadatas"][0] if results.get("metadatas") else {}
                }
                
                logger.info(f"Retrieved chunk {chunk_id} from ChromaDB")
                return JSONResponse({
                    "status": "success",
                    "operation": "get",
                    "chunk": chunk_data
                })
            except Exception as e:
                logger.exception(f"Failed to get chunk {text_value} from ChromaDB")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve chunk: {e}")
    
    elif operation == "delete":
        # DELETE operation
        if text_value.lower() == "all":
            # Delete all chunks
            try:
                # Get all IDs first
                results = collection.get()
                all_ids = results.get("ids", [])
                
                if not all_ids:
                    return JSONResponse({
                        "status": "success",
                        "operation": "delete",
                        "message": "No chunks to delete",
                        "deleted_count": 0
                    })
                
                # Delete all chunks
                collection.delete(ids=all_ids)
                logger.info(f"Deleted {len(all_ids)} chunks from ChromaDB")
                
                return JSONResponse({
                    "status": "success",
                    "operation": "delete",
                    "message": f"Deleted all {len(all_ids)} chunks",
                    "deleted_count": len(all_ids)
                })
            except Exception as e:
                logger.exception("Failed to delete all chunks from ChromaDB")
                raise HTTPException(status_code=500, detail=f"Failed to delete chunks: {e}")
        else:
            # Delete specific chunk by ID
            try:
                chunk_id = text_value
                # First check if chunk exists
                results = collection.get(ids=[chunk_id])
                
                if not results.get("ids"):
                    return JSONResponse({
                        "status": "not_found",
                        "message": f"Chunk with ID '{chunk_id}' not found",
                        "chunk_id": chunk_id
                    })
                
                # Delete the chunk
                collection.delete(ids=[chunk_id])
                logger.info(f"Deleted chunk {chunk_id} from ChromaDB")
                
                return JSONResponse({
                    "status": "success",
                    "operation": "delete",
                    "message": f"Deleted chunk '{chunk_id}'",
                    "deleted_chunk_id": chunk_id
                })
            except Exception as e:
                logger.exception(f"Failed to delete chunk {text_value} from ChromaDB")
                raise HTTPException(status_code=500, detail=f"Failed to delete chunk: {e}")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid operation '{operation}'. Supported operations: 'get', 'delete'"
        )

async def handle_chat1_request(
    select: str,
    message: str,
    file: Optional[UploadFile],
    gemini_api_key: Optional[str],
    model_preference: Optional[str] = None,
    operation: Optional[str] = None
) -> JSONResponse:
    """
    Main handler function for /chat1 endpoint.
    Routes to input processing, query processing, or data operations based on select parameter.
    
    Args:
        select: "input", "query", or "data" - determines the operation type
        message: The actual content/text/chunk_id
        file: Optional file upload (for input operation)
        gemini_api_key: API key for Gemini model
        model_preference: Model selection ("ollama", "gemma3:12b", or None for Gemini default)
        operation: Operation for data select ("get" or "delete")
    
    Model selection (default: gemini-2.5-flash-preview-05-20):
    - Supported models: "ollama", "gemma3:12b" (uses Ollama)
    - If no model specified or None/empty, defaults to gemini-2.5-flash-preview-05-20
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

