"""Simplified pipeline module for the ETL process.

This is a minimal implementation that supports various document formats.
"""
import os
import sys
import json
import logging
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import our simplified transform module
from simple_transform import transform_simplified

# Check for extra dependencies
def check_dependency(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

# Check for document processing support
PDF_SUPPORT = check_dependency("PyPDF2") or check_dependency("pdfminer.six")
DOCX_SUPPORT = check_dependency("python-docx")
HTML_SUPPORT = check_dependency("bs4")  # BeautifulSoup
MARKDOWN_SUPPORT = check_dependency("markdown")

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file formats.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        The extracted text content as a string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return ""
    
    file_extension = file_path.suffix.lower()
    logger.info(f"Extracting text from {file_path} (format: {file_extension})")
    
    try:
        # Handle different file types
        if file_extension == '.pdf' and PDF_SUPPORT:
            return extract_text_from_pdf(file_path)
        elif file_extension == '.docx' and DOCX_SUPPORT:
            return extract_text_from_docx(file_path)
        elif file_extension == '.html' and HTML_SUPPORT:
            return extract_text_from_html(file_path)
        elif file_extension == '.md' and MARKDOWN_SUPPORT:
            return extract_text_from_markdown(file_path)
        elif file_extension in ['.json']:
            return extract_text_from_json(file_path)
        elif file_extension in ['.csv']:
            return extract_text_from_csv(file_path)
        elif file_extension in ['.txt'] or True:  # Default to text for any other format
            return read_text_file(file_path)
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def read_text_file(file_path: str) -> str:
    """Read the contents of a text file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        The contents of the file as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        try:
            # Try again with a different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            return content
        except Exception as e2:
            logger.error(f"Error reading file with fallback encoding: {e2}")
            return ""
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    logger.info(f"Processing PDF file: {file_path}")
    file_name = Path(file_path).name.lower()
    is_manuscript = any(term in file_name for term in ["manuscript", "thesis", "dissertation", "paper", "article", "journal"])
    
    if is_manuscript:
        logger.info("Detected potential manuscript/academic PDF - using specialized settings")
    
    extracted_text = ""

    # Try PyPDF2 first
    if check_dependency("PyPDF2"):
        try:
            logger.info("Attempting extraction with PyPDF2...")
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                try:
                    reader = PyPDF2.PdfReader(file)
                    # Check if PDF is encrypted
                    if reader.is_encrypted:
                        logger.warning("PDF is encrypted - attempting to decrypt with empty password")
                        try:
                            reader.decrypt("")  # Try empty password
                        except:
                            logger.error("Could not decrypt PDF")
                            return "This PDF is encrypted and could not be decrypted automatically."
                    
                    num_pages = len(reader.pages)
                    logger.info(f"PDF has {num_pages} pages")
                    
                    # For manuscripts, we process all pages but concatenate with special markers
                    for page_num in range(num_pages):
                        try:
                            page = reader.pages[page_num]
                            page_text = page.extract_text()
                            
                            # For manuscripts, add page markers to help with context
                            if is_manuscript:
                                page_marker = f"\n\n--- PAGE {page_num+1} ---\n\n"
                                text += page_marker + page_text + "\n\n"
                            else:
                                text += page_text + "\n\n"
                                
                            logger.debug(f"Extracted {len(page_text)} chars from page {page_num+1}")
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {page_num+1}: {e}")
                except Exception as e:
                    logger.error(f"Error reading PDF with PyPDF2: {e}")
            
            if text.strip():
                logger.info(f"Successfully extracted {len(text)} characters with PyPDF2")
                extracted_text = text
            else:
                logger.warning("PyPDF2 extracted empty text, trying pdfminer.six")
        except Exception as e:
            logger.error(f"Error with PyPDF2: {e}")
    
    # Try pdfminer.six as fallback
    if not extracted_text and check_dependency("pdfminer.six"):
        try:
            logger.info("Attempting extraction with pdfminer.six...")
            from pdfminer.high_level import extract_text as pdfminer_extract_text
            text = pdfminer_extract_text(file_path)
            
            # For manuscripts, add some formatting to improve chunking
            if is_manuscript and text.strip():
                # Try to detect and format paragraphs better
                formatted_text = ""
                current_paragraph = ""
                
                for line in text.split('\n'):
                    # If line ends with period, question mark, or exclamation, it's likely end of paragraph
                    if line.strip().endswith(('.', '?', '!')) or not line.strip():
                        if current_paragraph:
                            current_paragraph += " " + line.strip()
                            formatted_text += current_paragraph + "\n\n"
                            current_paragraph = ""
                        elif line.strip():
                            formatted_text += line.strip() + "\n\n"
                    else:
                        current_paragraph += " " + line.strip() if current_paragraph else line.strip()
                
                # Add any remaining paragraph
                if current_paragraph:
                    formatted_text += current_paragraph + "\n\n"
                
                # Use formatted text if it's not empty
                if formatted_text.strip():
                    text = formatted_text
            
            if text.strip():
                logger.info(f"Successfully extracted {len(text)} characters with pdfminer.six")
                extracted_text = text
            else:
                logger.warning("pdfminer.six extracted empty text")
        except Exception as e:
            logger.error(f"Error with pdfminer.six: {e}")
            
    # Try poppler (pdftotext) as a last resort
    if not extracted_text and check_dependency("pdf2text"):
        try:
            logger.info("Attempting extraction with poppler/pdftotext...")
            from pdf2text import pdf2text
            text = pdf2text(file_path)
            
            if text.strip():
                logger.info(f"Successfully extracted {len(text)} characters with poppler/pdftotext")
                extracted_text = text
            else:
                logger.warning("poppler/pdftotext extracted empty text")
        except Exception as e:
            logger.error(f"Error with poppler/pdftotext: {e}")
    
    # If we extracted content, return it
    if extracted_text.strip():
        # Make sure there's actually meaningful content, not just formatting characters
        alphanumeric_chars = sum(c.isalnum() for c in extracted_text)
        if alphanumeric_chars > 50:  # At least 50 alphanumeric chars
            return extracted_text
    
    # If we've made it here, all extraction methods failed
    logger.error("Failed to extract meaningful text from PDF")
    
    # Return a helpful error message
    if is_manuscript:
        return """This manuscript PDF could not be processed automatically. It appears to be:

1. A scan of a physical document (no embedded text)
2. A PDF with complex formatting or security restrictions
3. A PDF with custom fonts or non-standard encoding

RECOMMENDATIONS:
- Try using an OCR tool like Adobe Acrobat or online services to convert this to a searchable PDF first
- If possible, try to find a "born digital" version of this document
- For manuscripts, try copying and pasting the text directly from the PDF viewer into a .txt file

To proceed with this document, you might need to extract the text manually."""
    else:
        return """This PDF could not be processed. It may be encrypted, damaged, or contain scanned images without OCR.
        
To process this document, try:
- Using an OCR tool to convert it to a searchable PDF
- Saving/exporting it in a different format
- Creating a text extract manually from the document
"""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text as a string
    """
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_html(file_path: str) -> str:
    """Extract text from an HTML file.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        Extracted text as a string
    """
    try:
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator="\n")
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}")
        return ""

def extract_text_from_markdown(file_path: str) -> str:
    """Extract text from a Markdown file.
    
    Args:
        file_path: Path to the Markdown file
        
    Returns:
        Extracted text as a string
    """
    try:
        # Just read the markdown file directly
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Optionally convert to HTML and then extract text
        if check_dependency("bs4"):
            import markdown
            from bs4 import BeautifulSoup
            
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            return text
        
        return content
    except Exception as e:
        logger.error(f"Error extracting text from Markdown: {e}")
        return ""

def extract_text_from_json(file_path: str) -> str:
    """Extract text from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Extracted text as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Simple approach: convert to string with indentation
        return json.dumps(data, indent=2)
    except Exception as e:
        logger.error(f"Error extracting text from JSON: {e}")
        return ""

def extract_text_from_csv(file_path: str) -> str:
    """Extract text from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Extracted text as a string
    """
    try:
        if check_dependency("pandas"):
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        else:
            # Fallback to using the csv module
            import csv
            text = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    text.append(",".join(row))
            return "\n".join(text)
    except Exception as e:
        logger.error(f"Error extracting text from CSV: {e}")
        return ""

def save_to_jsonl(chunks: List[Dict], output_dir: str, timestamp: str = None) -> str:
    """Save chunks to a JSON Lines file.
    
    Args:
        chunks: List of chunk dictionaries to save
        output_dir: Directory to save the file to
        timestamp: Optional timestamp to use in the filename
        
    Returns:
        Path to the saved file
    """
    if not chunks:
        logger.warning("No chunks to save")
        return None
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    filename = f"chunks_{timestamp}.jsonl"
    output_path = output_dir / filename
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
                
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving chunks to {output_path}: {e}")
        return None

def run_simplified_pipeline(file_path: str, strategy: str = "basic", 
                          generate_embeddings: bool = False,
                          max_chunk_size: int = 1000, 
                          chunk_overlap: float = 0.1) -> str:
    """Run a simplified ETL pipeline on a file.
    
    Args:
        file_path: Path to the file to process
        strategy: Processing strategy (not used in simplified version)
        generate_embeddings: Whether to generate embeddings
        max_chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks as a fraction
        
    Returns:
        Path to the output file
    """
    logger.info(f"Running simplified pipeline on file: {file_path}")
    logger.info(f"Strategy: {strategy} (note: only basic supported in simplified version)")
    
    # Ensure generate_embeddings is a boolean
    if not isinstance(generate_embeddings, bool):
        logger.warning(f"Converting generate_embeddings from {type(generate_embeddings)} to bool: {bool(generate_embeddings)}")
        generate_embeddings = bool(generate_embeddings)
    
    logger.info(f"Generate embeddings: {generate_embeddings}")
    logger.info(f"Max chunk size: {max_chunk_size}")
    logger.info(f"Chunk overlap: {chunk_overlap}")
    
    start_time = datetime.now()
    
    # Extract text from the document
    content = extract_text_from_file(file_path)
    if not content:
        logger.error(f"Failed to extract content from {file_path}")
        return None
        
    logger.info(f"Extracted {len(content)} characters from {file_path}")
    
    # Transform 
    chunk_overlap_percent = int(chunk_overlap * 100)
    
    # Pass the embeddings parameter as a strictly boolean value
    chunks = transform_simplified(
        [content],  # Pass as a list since our function expects multiple texts
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap_percent,
        generate_embeddings=bool(generate_embeddings)  # Ensure boolean
    )
    
    if not chunks:
        logger.error("Transformation produced no chunks")
        return None
        
    logger.info(f"Transformed content into {len(chunks)} chunks")
    
    # Add document metadata to all chunks
    for chunk in chunks:
        chunk["metadata"]["document_path"] = str(file_path)
        chunk["metadata"]["document_name"] = Path(file_path).name
        chunk["metadata"]["processing_strategy"] = strategy
    
    # Load
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = save_to_jsonl(
        chunks, 
        output_dir="./data/output",
        timestamp=timestamp
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"Pipeline completed in {duration:.2f} seconds")
    logger.info(f"Output saved to: {output_path}")
    
    return output_path 