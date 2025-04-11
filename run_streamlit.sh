#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Ensure the input and output directories exist
mkdir -p data/input
mkdir -p data/output

# Run the Streamlit app
streamlit run app.py 