#!/bin/bash
# =========================================================
# Streamlit Cloud Setup Script for Wipro NLP App
# =========================================================

# Upgrade pip to latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies in the correct order
echo "Installing Python packages..."
pip install numpy==1.26.2
pip install scipy==1.10.1
pip install pandas==2.2.0
pip install streamlit==1.29.0
pip install pdfplumber==0.10.0
pip install nltk==3.9.3
pip install textblob==0.17.1
pip install wordcloud==1.8.2.2
pip install matplotlib==3.8.0
pip install gensim==4.3.3
pip install scikit-learn==1.3.0

# Download NLTK corpora to avoid runtime errors
echo "Downloading NLTK corpora..."
python -m nltk.downloader stopwords punkt wordnet omw-1.4

echo "Setup complete!"
