<div align="center">

# ✨ ETL Pipeline for Processing Unstructured Data for Generative AI ✨

**Unlock the power of your documents for Large Language Models!**

</div>

<p align="center">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge&logo=python">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge">
  <img alt="Code Style: Black" src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge&logo=python">
  <img alt="Build Status" src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=githubactions"> <img alt="Project Status" src="https://img.shields.io/badge/status-active-blueviolet?style=for-the-badge"> </p>

---

## 🎯 Overview

This project implements a robust Extract, Transform, Load (ETL) pipeline designed to process diverse unstructured data sources (PDFs, websites, images, DOCX, etc.) and prepare them for seamless integration with Generative AI applications, particularly Retrieval-Augmented Generation (RAG) systems.

Leveraging the powerful `unstructured.io` library, this pipeline tackles the complexity of parsing, cleaning, and partitioning varied data types, transforming raw information into AI-ready knowledge.

**Problem:** 📄 Organizations possess vast amounts of valuable knowledge locked away in unstructured documents, often inaccessible to modern AI models.

**Solution:** 💡 This pipeline ingests these documents, intelligently processes them into clean, structured data chunks suitable for indexing and retrieval, making information readily available for GenAI applications.

The project includes a user-friendly Streamlit web interface that allows users to upload documents, configure processing parameters, and visualize the results.

---

## ✨ Key Features

* ✅ **Multi-Format Ingestion:** Handles a wide array of unstructured data types (PDF, HTML, DOCX, PNG, EML, etc.) using `unstructured`.
* 🧠 **Intelligent Partitioning:** Breaks down documents into meaningful semantic chunks using various strategies (`fast`, `hi_res`, `ocr_only`).
* 🧹 **Data Cleaning:** Automatically removes boilerplate content, headers/footers, and standardizes output.
* 🏷️ **Metadata Enrichment:** Adds crucial source information (filename, page numbers, element types) to processed chunks.
* 💾 **Flexible Loading:** Designed to load processed data into various destinations (JSON files, PostgreSQL, Vector Databases). *Current default: JSON Lines.*
* ⚙️ **Orchestration Ready:** Structured for easy integration with workflow managers like Prefect, Airflow, or Dagster.
* 📦 **Containerized:** Includes a `Dockerfile` for consistent, reproducible environment setup.
* 🖥️ **Streamlit Web Interface:** An intuitive web UI for uploading, processing, and visualizing document results.
* 📷 **OCR Detection:** Automatic detection of scanned PDFs with smart recommendations for OCR processing alternatives.

---

## 🏗️ Architecture

The pipeline follows a standard, modular ETL process:

1.  **Extract:** Reads raw unstructured files/data from configured sources.
2.  **Transform:** Utilizes `unstructured` for partitioning and cleaning. Enriches with metadata. (Optionally generates embeddings).
3.  **Load:** Saves the processed, structured data chunks to the target destination(s).

For a detailed diagram and technical explanation, please see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## 🛠️ Technology Stack

* **Core Processing:** Python 3.9+
* **Unstructured Data:** `unstructured.io` library & dependencies (`pypdf`, `python-docx`, `lxml`, `tesseract-ocr`, etc.)
* **Data Handling:** Pandas (Optional)
* **Orchestration:** [Placeholder - Prefect / Airflow / Dagster Recommended]
* **Containerization:** Docker
* **Potential Loading Targets:** JSON, PostgreSQL, Vector Databases (ChromaDB, FAISS, Pinecone, Weaviate, Qdrant)

---

## 📂 Project Structure

```
├── ARCHITECTURE.md # Detailed pipeline architecture
├── CONTRIBUTING.md # Contribution guidelines
├── Dockerfile # Docker configuration
├── LICENSE # Project License file (e.g., MIT)
├── README.md # This file
├── data/
│   ├── input/    # Place raw unstructured files here
│   │   └── .gitkeep
│   └── output/   # Processed output data saved here
│       └── .gitkeep
├── notebooks/    # Jupyter notebooks for exploration & testing
│   └── .gitkeep
├── requirements.txt # Python package dependencies
├── scripts/      # Utility scripts
│   └── .gitkeep
├── src/         # Main source code for the ETL pipeline
│   ├── __init__.py
│   ├── extract.py
│   ├── transform.py
│   ├── load.py
│   └── pipeline.py  # Main pipeline script
└── tests/       # Unit and integration tests
    └── .gitkeep
```

---

## 🚀 Setup & Usage

### Prerequisites

* Python 3.9+
* Docker (Recommended)
* System dependencies for `unstructured` (e.g., `tesseract-ocr`, `poppler-utils`, `libreoffice`). Refer to [Unstructured Installation Docs](https://unstructured-io.github.io/unstructured/installing.html).

### Installation

1.  **Clone:**
    ```bash
    git clone https://github.com/zmeshcode/your_repository_name.git
    cd your_repository_name
    ```
2.  **Set up Environment & Install Dependencies:**
    * **Using pip:**
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        pip install -r requirements.txt
        # Install extra dependencies if needed, e.g.:
        # pip install "unstructured[local-inference]" # For hi_res PDF/images
        # pip install "unstructured[ocr]" # For OCR-based strategies
        ```
    * **Using Docker:**
        ```bash
        docker build -t unstructured-etl .
        ```
3.  **Configuration:**
    * Create a `.env` file based on a potential `.env.example`.
    * Configure input/output paths, API keys (if needed), database URIs, etc.

### Running the Pipeline

* **Locally via Command Line:**
    ```bash
    # Ensure raw files are in data/input/
    # Ensure .env file is configured
    python src/pipeline.py
    # Processed files will appear in data/output/
    ```
* **With Streamlit UI:**
    ```bash
    # Run the Streamlit interface
    streamlit run simple_app.py
    # Then access http://localhost:8501 in your browser
    ```
* **With Docker:**
    ```bash
    # Example: Mount local data directory and pass env file
    docker run --rm --env-file .env -v $(pwd)/data:/app/data unstructured-etl
    ```

---

## 🌱 Future Enhancements

* Full integration with an orchestrator (Prefect/Airflow).
* Direct loading to multiple Vector Databases.
* Advanced data quality checks for processed chunks.
* Asynchronous processing (`asyncio`) for I/O bound tasks.
* Support for more `unstructured` connectors and file types.
* Advanced OCR integration for scanned documents.
* Improved visualization of document similarities using embeddings.
* API endpoints for programmatic access to the pipeline.

---

## Embeddings Integration

The pipeline supports generating embeddings for each text chunk, which is useful for semantic search and other AI applications. Two options are available:

### 1. Default Fallback Embeddings

The pipeline includes a robust fallback mechanism that generates deterministic random embeddings when the sentence-transformers library cannot be initialized. These embeddings:

- Have the correct dimensionality (384 for "all-MiniLM-L6-v2")
- Are normalized to unit length
- Are deterministic (same text = same embedding)
- Are tagged with `"embedding_type": "fallback_random"` in the metadata

**Note**: Fallback embeddings are suitable for development and testing but are not semantically meaningful for production use.

### 2. Real Embeddings via sentence-transformers

For production use, proper embeddings should be generated using the sentence-transformers library. Due to dependency conflicts between different versions of huggingface-hub, transformers, and sentence-transformers, you may encounter installation issues.

#### Resolving Dependencies

If you encounter errors when running with embeddings, here are some options:

1. **Use a clean environment**: Create a fresh virtual environment dedicated to this pipeline
   ```bash
   python -m venv fresh_venv
   source fresh_venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Find compatible versions**: For Python 3.12, we've found these package versions may cause conflicts:
   ```
   huggingface-hub==0.14.1 sentence-transformers==2.0.0 transformers==4.30.0
   ```

3. **Continue using fallback embeddings**: If dependency issues persist, you can keep using the fallback mechanism for development and testing.

### Enabling Embeddings

To enable embeddings, use the `--embeddings` flag:

```bash
python src/pipeline.py --embeddings
```

You can also specify a different embedding model:

```bash
python src/pipeline.py --embeddings --embedding-model all-mpnet-base-v2
```

The embedding vector will be included in each chunk as an `embedding` field, and metadata about the embedding type will be in the `metadata` field.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 📬 Contact

Zerayacob Meshesha

* **GitHub:** [zmeshcode](https://github.com/zmeshcode)
* **LinkedIn:** [linkedin.com/in/zerayacob-meshesha](https://www.linkedin.com/in/zerayacob-meshesha)
* **Email:** [zmeshesha1@gmail.com](mailto:zmeshesha1@gmail.com)

Project Link: [https://github.com/zmeshcode/your_repository_name](https://github.com/zmeshcode/your_repository_name) 