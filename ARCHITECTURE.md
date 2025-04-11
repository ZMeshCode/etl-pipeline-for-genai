# ETL Pipeline Architecture

This document describes the architecture and design decisions for the ETL pipeline for processing unstructured data for Generative AI applications.

## Overview

The pipeline follows a standard Extract, Transform, Load (ETL) pattern, designed to be modular, robust, and extensible. Each stage of the pipeline is implemented as a separate module to allow for easy maintenance and extension.

```
┌───────────┐    ┌─────────────┐    ┌───────────┐
│           │    │             │    │           │
│  Extract  │───►│  Transform  │───►│   Load    │
│           │    │             │    │           │
└───────────┘    └─────────────┘    └───────────┘
```

## Components

### 1. Configuration (`src/config.py`)

The configuration module manages settings for the pipeline using:
- Environment variables loaded from a `.env` file
- Command-line arguments
- Sensible defaults

It uses Pydantic for validation and provides a clean interface for accessing configuration values throughout the application.

### 2. Extract (`src/extract.py`)

The Extract module is responsible for:
- Scanning input directories for files
- Identifying file types (MIME types)
- Filtering to supported formats
- Providing metadata about the files to process

It produces a list of file information dictionaries that are passed to the Transform stage.

### 3. Transform (`src/transform.py`)

The Transform module uses the `unstructured.io` library to:
- Process files based on their type (PDF, HTML, DOCX, etc.)
- Partition content into meaningful chunks
- Apply the chosen strategy (fast, hi_res, ocr_only)
- Enrich the chunks with metadata (source, page numbers, etc.)

The transformation produces a list of chunk dictionaries containing:
- Extracted text content
- Chunk type (paragraph, title, list, etc.)
- Source metadata
- Processing metadata

### 4. Load (`src/load.py`)

The Load module handles:
- Saving processed chunks to the chosen destination
- Supporting multiple output formats (JSON Lines by default)
- Organizing output (by source or combined)
- Handling errors and ensuring data is properly saved

### 5. Pipeline Orchestration (`src/pipeline.py`)

The main pipeline module orchestrates the ETL process:
- Initializes the pipeline with configuration
- Calls the Extract, Transform, and Load stages in sequence
- Handles errors and provides logging
- Reports processing metrics (time, files processed, chunks generated)

## Data Flow

```
┌──────────────┐     ┌───────────────────┐     ┌───────────────┐
│              │     │                   │     │               │
│ Unstructured │     │  Unstructured.io  │     │  Structured   │
│    Files     │────►│    Processing     │────►│    Chunks     │
│              │     │                   │     │               │
└──────────────┘     └───────────────────┘     └───────────────┘
      │                                               │
      │                                               │
      ▼                                               ▼
┌──────────────┐                             ┌───────────────┐
│              │                             │               │
│   Metadata   │                             │ JSON/Database │
│ Extraction   │                             │    Output     │
│              │                             │               │
└──────────────┘                             └───────────────┘
```

## Design Principles

1. **Modularity**: Each component is designed to be loosely coupled and highly cohesive, allowing for independent development and testing.

2. **Robustness**: The pipeline includes comprehensive error handling and logging to ensure reliability even with diverse and potentially problematic input data.

3. **Extensibility**: The architecture is designed to allow easy extension for:
   - New input formats
   - Advanced transformation strategies 
   - Additional output destinations
   - Integration with workflow orchestrators

4. **Configuration**: The pipeline is highly configurable while providing sensible defaults, allowing users to adapt it to their specific needs.

## Future Enhancements

### Phase 3: Advanced Transformations
- Integration with embedding models
- Custom chunking logic
- Enhanced metadata extraction

### Phase 4: Orchestration
- Integration with workflow managers (Prefect, Airflow, Dagster)
- Containerization for deployment

### Phase 5: Scalability
- Parallel processing for large-scale data
- Cloud storage integration
- Vector database output options 