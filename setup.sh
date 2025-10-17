#!/bin/bash
# Streamlit Cloud Setup Script: Pre-download NLTK corpora

echo "Downloading NLTK corpora..."
python -m nltk.downloader stopwords punkt wordnet omw-1.4
echo "NLTK setup complete."
