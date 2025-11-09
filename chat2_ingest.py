"""
Document Ingestion Module for chat2 RAG Pipeline.
Handles reading, cleaning, and chunking of source documents.
"""
import io
import re
import logging
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from chat2_config import CHUNK_CHAR_SIZE, CHUNK_OVERLAP_SIZE, MIN_CHUNK_SIZE

logger = logging.getLogger("chat2_ingest")

class DocumentIngester:
    """Handles document ingestion, cleaning, and chunking."""
    
    def __init__(self, chunk_size: int = CHUNK_CHAR_SIZE, chunk_overlap: int = CHUNK_OVERLAP_SIZE):
        """
        Initialize the document ingester.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initialized DocumentIngester with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Extracted text as string
        """
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                        logger.debug(f"Extracted text from page {page_num}: {len(text)} characters")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {e}")
                    pages.append("")
            
            full_text = "\n".join(pages)
            logger.info(f"Extracted {len(full_text)} characters from PDF ({len(pages)} pages)")
            return full_text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise ValueError(f"PDF extraction failed: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Trim whitespace
        text = text.strip()
        
        logger.debug(f"Cleaned text: {len(text)} characters")
        return text
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            logger.warning("Empty text provided for chunking")
            return []
        
        text = self.clean_text(text)
        
        if len(text) <= self.chunk_size:
            # Text fits in one chunk
            chunk = {
                "text": text,
                "metadata": metadata or {},
                "chunk_index": 0,
                "start_char": 0,
                "end_char": len(text)
            }
            logger.info(f"Text fits in single chunk: {len(text)} characters")
            return [chunk]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_endings = ['. ', '.\n', '! ', '!\n', '? ', '?\n']
                best_break = end
                
                # Search within last 200 characters for sentence boundary
                search_start = max(start + self.chunk_size - 200, start)
                for i in range(end, search_start, -1):
                    if i < len(text) - 1:
                        if text[i:i+2] in sentence_endings:
                            best_break = i + 2
                            break
                
                end = min(best_break, len(text))
            
            chunk_text = text[start:end].strip()
            
            # Skip chunks that are too small (unless it's the last chunk)
            if len(chunk_text) < MIN_CHUNK_SIZE and end < len(text):
                start = end
                continue
            
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata.update({
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "chunk_size": len(chunk_text)
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end
            })
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            chunk_index += 1
            
            # Prevent infinite loop
            if start >= end:
                start = end
        
        logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks
    
    def process_document(
        self, 
        text: Optional[str] = None, 
        pdf_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a document (PDF or text) and return chunks.
        
        Args:
            text: Text content (if provided)
            pdf_bytes: PDF file as bytes (if provided)
            filename: Original filename
            additional_metadata: Additional metadata to attach
            
        Returns:
            List of chunk dictionaries
        """
        if pdf_bytes:
            extracted_text = self.extract_text_from_pdf(pdf_bytes)
            if text:
                # Combine PDF text with additional text
                extracted_text = f"{extracted_text}\n\n{text}"
            text = extracted_text
        
        if not text:
            raise ValueError("Either text or pdf_bytes must be provided")
        
        # Prepare metadata
        metadata = additional_metadata or {}
        if filename:
            metadata["source_file"] = filename
        metadata["document_length"] = len(text)
        
        # Chunk the text
        chunks = self.chunk_text(text, metadata)
        
        logger.info(f"Processed document: {len(chunks)} chunks created")
        return chunks



