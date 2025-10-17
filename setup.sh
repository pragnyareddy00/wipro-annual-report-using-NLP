#!/bin/bash
# Ensure dependencies are installed in correct order
pip install --upgrade pip
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade gensim
python -m nltk.downloader stopwords punkt wordnet omw-1.4

echo "Downloading NLTK corpora..."
python -m nltk.downloader stopwords punkt wordnet omw-1.4
echo "NLTK setup complete."
