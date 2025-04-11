"""Simplified transform module for the ETL pipeline.

This module contains functions for transforming unstructured data into
a structured format without requiring huggingface or sentence-transformers.
"""
import uuid
import random
import math
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import importlib
import traceback

# Set up logging
logger = logging.getLogger(__name__)

# Flag to indicate if we're using the simplified version
USING_SIMPLIFIED = True

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("sentence-transformers is available - will use real embeddings")
    # Load the model lazily when first needed
    _embedding_model = None
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available - will use deterministic pseudo-embeddings")

def get_embedding_model():
    """Get or initialize the embedding model."""
    global _embedding_model
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
        
    if _embedding_model is None:
        try:
            # Use a small, efficient model
            logger.info("Loading all-MiniLM-L6-v2 embedding model...")
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return None
            
    return _embedding_model

def generate_embedding(text: str, dimension: int = 384) -> List[float]:
    """Generate an embedding for text, using real model if available.
    
    Args:
        text: Text to generate embedding for
        dimension: Dimension of the embedding (only used for deterministic fallback)
        
    Returns:
        List of floats representing the embedding vector
    """
    # Try to use the real model first
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        model = get_embedding_model()
        if model:
            try:
                embedding = model.encode(text).tolist()
                return embedding
            except Exception as e:
                logger.error(f"Error generating real embedding: {e}")
                # Fall back to deterministic embedding
    
    # Fall back to deterministic embedding
    return generate_deterministic_embedding(text, dimension)

def generate_deterministic_embedding(text: str, dimension: int = 384) -> List[float]:
    """Generate a deterministic pseudo-random embedding based on text hash.
    
    Args:
        text: Text to generate embedding for
        dimension: Dimension of the embedding to generate
        
    Returns:
        List of floats representing the embedding vector
    """
    # Create a deterministic seed from the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    seed = int(text_hash, 16) % (2**32)
    random.seed(seed)
    
    # Generate a deterministic random embedding
    embedding = [random.uniform(-1, 1) for _ in range(dimension)]
    
    # Normalize the embedding to unit length
    norm = math.sqrt(sum(x**2 for x in embedding))
    embedding = [x/norm for x in embedding]
    
    return embedding

def split_text_into_chunks(text: str, max_chunk_size: int = 1000, overlap: int = 0) -> List[str]:
    """Split text into chunks of specified size with overlap.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Percentage of overlap between chunks (0-100)
        
    Returns:
        List of text chunks
    """
    if not text or max_chunk_size <= 0 or len(text) <= max_chunk_size:
        return [text] if text else []
        
    # Calculate overlap size in characters
    overlap_size = int(max_chunk_size * (overlap / 100))
    
    # Initialize variables
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + max_chunk_size
        
        # If we're at the end of the text, just take the rest
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Try to find a good break point (period, newline, space)
        # Look within the last 20% of the chunk or at least 100 chars
        search_range = max(100, int(max_chunk_size * 0.2))
        good_end = end
        
        # Look for a period followed by space or newline
        for i in range(end, max(start, end - search_range), -1):
            if i < len(text) and text[i-1] == '.' and (i == len(text) or text[i] in ' \n'):
                good_end = i
                break
                
        # If we couldn't find a period, look for a newline
        if good_end == end:
            for i in range(end, max(start, end - search_range), -1):
                if i < len(text) and text[i-1] == '\n':
                    good_end = i
                    break
                    
        # If we still couldn't find a good break, look for a space
        if good_end == end:
            for i in range(end, max(start, end - search_range), -1):
                if i < len(text) and text[i-1] == ' ':
                    good_end = i
                    break
        
        # Add the chunk
        chunks.append(text[start:good_end])
        
        # Update start position for next chunk, accounting for overlap
        start = good_end - overlap_size
        
    return chunks

def process_text(text: str, max_chunk_size: int = 1000, chunk_overlap: int = 0, 
                generate_embeddings: bool = False) -> List[Dict[str, Any]]:
    """Process text into chunks with optional embeddings.
    
    Args:
        text: Text to process
        max_chunk_size: Maximum size of each chunk
        chunk_overlap: Percentage of overlap between chunks
        generate_embeddings: Whether to generate embeddings
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    # Split text into chunks
    text_chunks = split_text_into_chunks(text, max_chunk_size, chunk_overlap)
    
    # Process each chunk
    for i, chunk_text in enumerate(text_chunks):
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "text": chunk_text,
            "index": i,
            "metadata": {
                "source": "simplified_transform",
                "chunk_size": len(chunk_text),
                "timestamp": datetime.now().isoformat(),
                "simplified": not SENTENCE_TRANSFORMERS_AVAILABLE
            }
        }
        
        # Generate embeddings if requested
        if generate_embeddings:
            chunk["embedding"] = generate_embedding(chunk_text)
            
        chunks.append(chunk)
        
    return chunks

def transform_simplified(input_text: List[str], max_chunk_size: int = 1000, 
                       chunk_overlap: int = 0, generate_embeddings: bool = False) -> List[Dict[str, Any]]:
    """Transform text content into structured chunks.
    
    Args:
        input_text: List of text inputs to process
        max_chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Percentage of overlap between chunks
        generate_embeddings: Whether to generate embeddings for chunks
        
    Returns:
        List of chunk dictionaries
    """
    logger.info(f"Starting transformation with {len(input_text)} texts")
    logger.info(f"Chunking with max_size={max_chunk_size}, overlap={chunk_overlap}")
    
    # Print debug information about generate_embeddings
    logger.info(f"generate_embeddings value: {generate_embeddings}, type: {type(generate_embeddings)}")
    print(f"DEBUG: generate_embeddings value: {generate_embeddings}, type: {type(generate_embeddings)}")
    
    # Force convert to boolean to avoid any type issues
    generate_embeddings = bool(generate_embeddings)
    
    try:
        # Make sure generate_embeddings is a boolean
        if not isinstance(generate_embeddings, bool):
            logger.warning(f"generate_embeddings was not a boolean: {generate_embeddings}, type: {type(generate_embeddings)}")
            generate_embeddings = bool(generate_embeddings)
    except Exception as e:
        logger.error(f"Error converting generate_embeddings to boolean: {e}")
        generate_embeddings = False
    
    if generate_embeddings:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("Using sentence-transformers for real embeddings")
        else:
            logger.info("Using deterministic embeddings (sentence-transformers not available)")
    
    all_chunks = []
    
    for i, text in enumerate(input_text):
        logger.info(f"Processing text {i+1}/{len(input_text)}")
        try:
            text_chunks = process_text(
                text, 
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                generate_embeddings=generate_embeddings
            )
            all_chunks.extend(text_chunks)
        except Exception as e:
            logger.error(f"Error processing text {i+1}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    logger.info(f"Transformation complete. Generated {len(all_chunks)} chunks")
    return all_chunks 