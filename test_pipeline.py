#!/usr/bin/env python3
"""Test script for the ETL pipeline.

This script directly runs the pipeline on a PDF file, bypassing the UI.
It's useful for debugging issues with PDF processing.

Usage:
    python test_pipeline.py your_file.pdf

"""
import sys
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def run_test(file_path, generate_embeddings=False):
    """Run the pipeline directly on a file."""
    try:
        # Import the pipeline
        from simple_pipeline import run_simplified_pipeline
        
        print(f"Processing file: {file_path}")
        print(f"Generate embeddings: {generate_embeddings}")
        
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return
        
        # Run the pipeline
        output_path = run_simplified_pipeline(
            file_path=file_path,
            strategy="basic",
            generate_embeddings=generate_embeddings,
            max_chunk_size=1000,
            chunk_overlap=0.1
        )
        
        print(f"Processing complete!")
        print(f"Output saved to: {output_path}")
        
        # Count chunks
        if output_path:
            with open(output_path, 'r') as f:
                chunks = f.readlines()
                print(f"Generated {len(chunks)} chunks")
    
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <file_path> [--embeddings]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    generate_embeddings = "--embeddings" in sys.argv
    
    run_test(file_path, generate_embeddings) 