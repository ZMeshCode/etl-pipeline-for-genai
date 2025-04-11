"""Enhanced Streamlit app for ETL Pipeline.

This version supports multiple document formats and has an improved UI.
"""
import streamlit as st

# IMPORTANT: st.set_page_config must be called before any other Streamlit commands
st.set_page_config(page_title="ETL Platform for AI", layout="wide")

import os
import uuid
from pathlib import Path
import time
import json
import logging
import importlib
from datetime import datetime

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Check for dependencies
def check_dependency(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

# Check for essential dependencies
STREAMLIT_AVAILABLE = check_dependency("streamlit")  # Should always be True if we're running this
PANDAS_AVAILABLE = check_dependency("pandas")  # Needed for display
JSONLINES_AVAILABLE = check_dependency("jsonlines")  # Needed for loading/saving
SENTENCE_TRANSFORMERS_AVAILABLE = check_dependency("sentence_transformers")
PDF_SUPPORT = check_dependency("PyPDF2") or check_dependency("pdfminer.six")
DOCX_SUPPORT = check_dependency("python-docx")

# Create necessary directories
os.makedirs("data/input", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

# Import our simplified pipeline
try:
    from simple_pipeline import run_simplified_pipeline
except ImportError as e:
    st.error(f"Error importing simple pipeline module: {e}")
    logger.error(f"Error importing simple pipeline module: {e}")

# Custom CSS to enhance the app appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .feature-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .feature-item {
        margin-left: 1rem;
    }
    .custom-card {
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)
    
# App title and description
st.markdown('<p class="main-header">Unstructured Data ETL Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform documents into AI-ready data with our advanced processing pipeline</p>', unsafe_allow_html=True)

# Create tabs for the main sections
tab1, tab2, tab3 = st.tabs(["Main", "System Status", "About"])

# Sidebar configuration using dropdowns
st.sidebar.markdown('<p class="sidebar-header">⚙️ Pipeline Configuration</p>', unsafe_allow_html=True)

# Embedding options as dropdown
st.sidebar.markdown("#### Embedding Generation")
generate_embeddings = st.sidebar.selectbox(
    "Generate Vector Embeddings",
    options=["Enabled", "Disabled"],
    index=1,
    help="Generate vector embeddings for AI applications"
)

if generate_embeddings == "Enabled" and not SENTENCE_TRANSFORMERS_AVAILABLE:
    st.sidebar.warning("⚠️ Using deterministic embeddings (sentence-transformers not available)")

# Chunk size as dropdown
st.sidebar.markdown("#### Chunking Configuration")
chunk_size = st.sidebar.selectbox(
    "Maximum Chunk Size",
    options=["500", "1000", "1500", "2000", "2500"],
    index=1,
    help="Maximum size of each text chunk in characters"
)

# Chunk overlap as dropdown
chunk_overlap_values = {
    "None (0%)": 0.0,
    "Low (10%)": 0.1,
    "Medium (20%)": 0.2,
    "High (30%)": 0.3,
    "Very High (50%)": 0.5
}

chunk_overlap = st.sidebar.selectbox(
    "Chunk Overlap",
    options=list(chunk_overlap_values.keys()),
    index=1,
    help="Overlap between adjacent chunks"
)

# Additional processing options
st.sidebar.markdown("#### Advanced Options")
st.sidebar.checkbox(
    "Save Raw Document Copy", 
    value=True,
    help="Save a copy of the original document"
)

with tab1:
    # Main content
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    # File uploader section with enhanced UI
    st.subheader("📄 Upload Document")
    
    # Supported file types
    file_types = ['txt']
    file_type_labels = ['Text (.txt)']
    
    if PDF_SUPPORT:
        file_types.append('pdf')
        file_type_labels.append('PDF (.pdf)')
    else:
        st.info("📌 PDF support not available. Install PyPDF2 or pdfminer.six to enable.")
        
    if DOCX_SUPPORT:
        file_types.append('docx')
        file_type_labels.append('Word (.docx)')
    else:
        st.info("📌 DOCX support not available. Install python-docx to enable.")
    
    # Add HTML and other types
    file_types.extend(['html', 'md', 'json', 'csv'])
    file_type_labels.extend(['HTML (.html)', 'Markdown (.md)', 'JSON (.json)', 'CSV (.csv)'])
    
    # File uploader with all supported types
    uploaded_file = st.file_uploader(
        'Select or drag a document to process', 
        type=file_types,
        help=f"Supported formats: {', '.join(file_type_labels)}"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results section
    if uploaded_file is not None:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**Selected file:** {uploaded_file.name}")
        st.markdown(f"**File type:** {uploaded_file.type if hasattr(uploaded_file, 'type') else 'Unknown'}")
        st.markdown(f"**File size:** {round(uploaded_file.size/1024, 2)} KB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create columns for the status and results
    col1, col2 = st.columns([1, 1])
    
    # Processing button
    process_button = st.button('🚀 Process Document', use_container_width=True)

    # Processing logic
    if process_button:
        if uploaded_file is not None:
            with st.spinner('⏳ Processing document...'):
                # Generate a unique job ID
                job_id = str(uuid.uuid4())
                
                # Create a timestamp for this job
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create job info for display
                job_info = {
                    "job_id": job_id,
                    "timestamp": timestamp,
                    "filename": uploaded_file.name,
                    "status": "PROCESSING"
                }
                
                # Save the uploaded file to a temporary location
                input_file_path = f"data/input/{uploaded_file.name}"
                with open(input_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Show processing status
                    with col1:
                        st.info(f"🔄 Processing file: {uploaded_file.name}")
                        
                    # Get embedding setting - convert string to boolean EXPLICITLY
                    embedding_enabled = generate_embeddings == "Enabled"
                    logger.info(f"Embedding setting: UI={generate_embeddings}, converted to: {embedding_enabled} (type: {type(embedding_enabled).__name__})")
                    
                    # Get the chunk size and overlap values
                    chunk_size_value = int(chunk_size)
                    chunk_overlap_value = float(chunk_overlap_values[chunk_overlap])
                    
                    logger.info(f"Processing with chunk_size={chunk_size_value}, overlap={chunk_overlap_value}")
                    
                    # Run the simplified ETL pipeline
                    output_path = run_simplified_pipeline(
                        file_path=input_file_path,
                        strategy="basic",  # Only basic supported in simplified version
                        generate_embeddings=embedding_enabled,  # Pass as boolean
                        max_chunk_size=chunk_size_value,
                        chunk_overlap=chunk_overlap_value
                    )
                    
                    # Check if output_path is None (indicates failure)
                    if output_path is None:
                        raise Exception("Pipeline failed to produce output. Check logs for details.")
                    
                    # Update job info
                    job_info["status"] = "SUCCESS"
                    job_info["output_path"] = output_path
                    
                    # Display success message
                    with col1:
                        st.success(f"✅ Document processed successfully!")
                        
                    # Try to display the output
                    if output_path and os.path.exists(output_path):
                        with col2:
                            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                            st.subheader("📊 Output Preview")
                            try:
                                # Read the JSONL file
                                with open(output_path, 'r') as f:
                                    # Read the first 5 lines
                                    chunks = []
                                    for i, line in enumerate(f):
                                        if i >= 5:  # Only read first 5 chunks
                                            break
                                        chunks.append(json.loads(line))
                                
                                # Display as table
                                if chunks:
                                    # Check if this is a failed PDF with error message
                                    if len(chunks) == 1 and "This PDF could not be processed" in chunks[0].get("text", ""):
                                        # This is a PDF that couldn't be processed
                                        st.warning("⚠️ PDF Processing Issue Detected")
                                        st.markdown(chunks[0].get("text", ""))
                                        
                                        # Offer OCR alternatives
                                        st.markdown("### OCR Alternatives")
                                        st.markdown("""
                                        You may want to try these free OCR services to convert your PDF:
                                        
                                        - [Adobe Acrobat Online OCR](https://www.adobe.com/acrobat/online/pdf-to-text.html) (Free)
                                        - [Google Drive OCR](https://support.google.com/drive/answer/176692?hl=en) (Upload PDF → Open with Google Docs)
                                        - [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) (Open source tool)
                                        """)
                                    else:
                                        # Normal document processing results
                                        st.write(f"First {len(chunks)} chunks:")
                                        for i, chunk in enumerate(chunks):
                                            with st.expander(f"🔍 Chunk {i+1}"):
                                                st.write(f"**Text**: {chunk.get('text', '')[:200]}...")
                                                st.write(f"**Metadata**: {chunk.get('metadata', {})}")
                                                if "embedding" in chunk:
                                                    st.write(f"**Embedding**: Vector with {len(chunk['embedding'])} dimensions")
                                                    
                                                    # Visualization of embedding (simplified)
                                                    if len(chunk['embedding']) > 10:
                                                        import matplotlib.pyplot as plt
                                                        import numpy as np
                                                        try:
                                                            fig, ax = plt.subplots(figsize=(8, 2))
                                                            ax.imshow([chunk['embedding'][:100]], aspect='auto', cmap='viridis')
                                                            ax.set_yticks([])
                                                            ax.set_xticks([])
                                                            ax.set_title("Embedding Visualization (first 100 dimensions)")
                                                            st.pyplot(fig)
                                                        except:
                                                            pass
                                
                                # Download button with enhanced styling
                                st.markdown("### 📥 Download Results")
                                st.markdown("Get your processed data in JSONL format:")
                                with open(output_path, 'rb') as f:
                                    st.download_button(
                                        label="Download JSONL Data",
                                        data=f,
                                        file_name=os.path.basename(output_path),
                                        mime="application/jsonl"
                                    )
                                
                                # Check if chunks were generated
                                if len(chunks) == 0:
                                    st.warning("No chunks were generated from this file.")
                            except Exception as e:
                                st.error(f"Error displaying preview: {e}")
                            st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    # Update job info
                    job_info["status"] = "FAILURE"
                    job_info["error"] = str(e)
                    
                    # Display detailed error message
                    with col1:
                        st.error(f"❌ Error processing file: {e}")
                        
                        # Show more detailed error information
                        with st.expander("Error Details"):
                            st.markdown("### Troubleshooting")
                            
                            # Check common error types and provide specific guidance
                            error_msg = str(e).lower()
                            
                            if "pdf" in error_msg or "pdf" in uploaded_file.name.lower():
                                st.markdown("""
                                **PDF Processing Error**
                                - The PDF might be encrypted or password-protected
                                - The PDF might contain scanned images without OCR text
                                - Try a different PDF file that contains actual text content
                                """)
                                
                            elif "docx" in error_msg or "docx" in uploaded_file.name.lower():
                                st.markdown("""
                                **DOCX Processing Error**
                                - The document might be corrupted
                                - Try saving it as plain text (.txt) and uploading again
                                """)
                                
                            elif "generate_embeddings" in error_msg:
                                st.markdown("""
                                **Embedding Generation Error**
                                - There appears to be an issue with the embedding configuration
                                - Try disabling embeddings and processing again
                                """)
                            
                            # Generic guidance for all errors
                            st.markdown("""
                            **General Troubleshooting:**
                            1. Try a simpler document (plain text)
                            2. Disable embeddings if they're enabled
                            3. Try a smaller chunk size (500)
                            4. Check that the file isn't corrupted
                            """)
                            
                            # Offer to show the raw error for technical users
                            st.code(f"Error details:\n{e}", language="bash")
        else:
            st.error('⚠️ Please upload a file to process.')

with tab2:
    # System status in the second tab
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 🧰 System Status")
    
    # Create an enhanced status display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Core Components")
        st.markdown(f"- **Streamlit**: {'✅ Operational' if STREAMLIT_AVAILABLE else '❌ Missing'}")
        st.markdown(f"- **Pandas**: {'✅ Operational' if PANDAS_AVAILABLE else '❌ Missing - Display may be limited'}")
        st.markdown(f"- **JSON Lines**: {'✅ Operational' if JSONLINES_AVAILABLE else '❌ Missing - Output will be limited'}")
    
    with col2:
        st.markdown("### AI Components")
        st.markdown(f"- **Sentence Transformers**: {'✅ Operational' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌ Missing - Using deterministic embeddings'}")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.markdown("""
            ℹ️ **Embedding Mode**: Using deterministic embeddings (fallback mode)
            
            For real embeddings, install: 
            ```
            pip install sentence-transformers==2.2.2
            ```
            """)
        else:
            st.markdown("✅ **Embedding Mode**: Using real AI-powered embeddings")
    
    # Document support status
    st.markdown("### Document Format Support")
    
    # Create a grid of document format statuses
    format_cols = st.columns(3)
    
    with format_cols[0]:
        st.markdown("- **Text (.txt)**: ✅ Supported")
        st.markdown(f"- **PDF (.pdf)**: {'✅ Supported' if PDF_SUPPORT else '❌ Not Supported'}")
        
    with format_cols[1]:
        st.markdown(f"- **Word (.docx)**: {'✅ Supported' if DOCX_SUPPORT else '❌ Not Supported'}")
        st.markdown("- **HTML (.html)**: ✅ Supported")
        
    with format_cols[2]:
        st.markdown("- **Markdown (.md)**: ✅ Supported")
        st.markdown("- **JSON/CSV (.json/.csv)**: ✅ Supported")
    
    # Installation instructions for missing components
    if not (PDF_SUPPORT and DOCX_SUPPORT and SENTENCE_TRANSFORMERS_AVAILABLE):
        st.markdown("### Installation Instructions")
        
        install_cmd = "pip install "
        
        if not PDF_SUPPORT:
            install_cmd += "PyPDF2 "
        
        if not DOCX_SUPPORT:
            install_cmd += "python-docx "
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            install_cmd += "sentence-transformers==2.2.2 "
        
        st.code(install_cmd, language="bash")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    # About section in the third tab
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 🔍 About this Application")
    
    st.markdown("""
    ### ETL Pipeline for Generative AI
    
    This application demonstrates a complete ETL (Extract, Transform, Load) pipeline for processing unstructured data into a format suitable for Generative AI applications.
    
    <p class="feature-header">🔹 Key Features</p>
    
    <p class="feature-item">📥 <strong>Extract</strong>: Upload documents in various formats (PDF, DOCX, TXT, etc.)</p>
    <p class="feature-item">⚙️ <strong>Transform</strong>: Process documents into semantic chunks with intelligent splitting algorithms</p>
    <p class="feature-item">📤 <strong>Load</strong>: Export processed data in JSONL format ready for AI applications</p>
    <p class="feature-item">🧠 <strong>Embeddings</strong>: Generate vector embeddings for semantic search and AI retrieval</p>
    <p class="feature-item">🔄 <strong>Extensibility</strong>: Modular design for adding new document types and processing strategies</p>
    
    <p class="feature-header">🔹 Technical Implementation</p>
    
    <p class="feature-item">- Chunking with intelligent boundary detection at sentence and paragraph breaks</p>
    <p class="feature-item">- Embeddings via sentence-transformers (all-MiniLM-L6-v2 model) when available</p>
    <p class="feature-item">- Deterministic fallback embeddings when sentence-transformers is unavailable</p>
    <p class="feature-item">- Configurable chunk size and overlap for fine-tuning to specific use cases</p>
    <p class="feature-item">- Robust error handling with graceful fallbacks</p>
    
    <p class="feature-header">🔹 Use Cases</p>
    
    <p class="feature-item">🤖 <strong>RAG Systems</strong>: Prepare content for Retrieval Augmented Generation</p>
    <p class="feature-item">🔍 <strong>Semantic Search</strong>: Enable powerful search capabilities across document collections</p>
    <p class="feature-item">📊 <strong>Content Analysis</strong>: Break down large documents for efficient processing</p>
    <p class="feature-item">📚 <strong>Knowledge Base Creation</strong>: Transform unstructured data into structured knowledge bases</p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tips section in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("#### 💡 Tips")
st.sidebar.info(
    "• Use 1000-1500 character chunks for general purpose\n"
    "• Enable embeddings for semantic search\n"
    "• Higher overlap helps preserve context between chunks"
)

# Version info
st.sidebar.markdown("---")
st.sidebar.markdown("v1.0.0 | ETL Pipeline for AI") 