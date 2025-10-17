import nltk

required_resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
for res in required_resources:
    try:
        nltk.data.find(f'corpora/{res}')
    except LookupError:
        nltk.download(res, quiet=True)
# -*- coding: utf-8 -*-
"""
NLP Mini Project: Annual Report Analysis (Streamlit App)

This Streamlit app performs NLP tasks on an uploaded company annual report PDF.
Tasks include text extraction, preprocessing, sentiment analysis, word frequency analysis,
word cloud generation, and topic modeling using LDA.
"""

import streamlit as st
import os
import re
import logging
import warnings
import pandas as pd
import numpy as np
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.models import LdaModel
import io

# ============================================================================== 
# ----------------------------- NLTK Setup Function -----------------------------
# ==============================================================================

@st.cache_resource
def setup_nltk_resources():
    """
    Ensure all required NLTK resources are available.
    Downloads them if missing.
    """
    required_resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    for res in required_resources:
        try:
            if res == 'punkt':  # tokenizer models are in 'tokenizers'
                nltk.data.find(f'tokenizers/{res}')
            else:
                nltk.data.find(f'corpora/{res}')
        except LookupError:
            st.sidebar.info(f"Downloading NLTK resource: {res} ...")
            nltk.download(res, quiet=True)
    st.sidebar.success("✅ All NLTK resources are ready!")
    return True

# Initialize NLTK resources
nltk_ready = setup_nltk_resources()

if nltk_ready:
    st.title("📄 Wipro NLP Report Assistant")
    st.write("NLTK resources are loaded. You can now run your NLP pipeline safely.")

# ============================================================================== 
# ----------------------------- App Config & Setup -----------------------------
# ==============================================================================

NUM_TOPICS = 10
RANDOM_SEED = 42

# Suppress pdfminer warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pdfminer.*')
logging.getLogger("pdfminer").setLevel(logging.WARNING)

# Initialize stopwords and lemmatizer after NLTK is ready
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ============================================================================== 
# ----------------------------- Helper & Processing Functions ------------------
# ==============================================================================

# Task 1: PDF Extraction
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """Extracts text from all pages of an uploaded PDF file."""
    all_text = []
    if uploaded_file is not None:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                total_pages = len(pdf.pages)
                st.sidebar.info(f"Processing PDF: {total_pages} pages...")
                progress_bar = st.sidebar.progress(0)
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
                    progress_bar.progress((i + 1) / total_pages)
            st.sidebar.success(f"Text extracted from {len(all_text)} pages.")
            return "\n".join(all_text)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    return None

# Task 3: Preprocessing Functions
def basic_clean(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_words(text, do_lemmatize=True):
    cleaned_text = basic_clean(text)
    tokens = word_tokenize(cleaned_text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# Task 4: Sentiment Analysis
def calculate_sentiment(text):
    """Calculates polarity and subjectivity using TextBlob."""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception:
        return np.nan, np.nan

# Task 6: Plotting Functions
def generate_wordcloud_figure(tokens):
    wordcloud_text = " ".join(tokens)
    if not wordcloud_text.strip():
        st.warning("Cannot generate WordCloud: No processable words found.")
        return None
    wordcloud = WordCloud(
        width=1000, height=500, background_color='white',
        colormap='viridis', max_words=150, random_state=RANDOM_SEED
    ).generate(wordcloud_text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud - Annual Report Analysis', fontsize=16)
    plt.tight_layout(pad=0)
    return fig

def generate_barchart_figure(top_words):
    if not top_words:
        st.warning("Cannot generate Bar Chart: No frequent words found.")
        return None
    words, counts = zip(*top_words)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(words)), counts[::-1], color='skyblue')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words[::-1])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Words')
    ax.set_title('Top 20 Most Frequent Words')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

# Task 8: LDA Model Training
@st.cache_data
def train_lda_model(sentence_tokens_for_lda):
    if not sentence_tokens_for_lda:
        return None, None, None
    try:
        dictionary = corpora.Dictionary(sentence_tokens_for_lda)
        dictionary.filter_extremes(no_below=5, no_above=0.7, keep_n=10000)
        corpus = [dictionary.doc2bow(text) for text in sentence_tokens_for_lda]
        if not corpus:
            st.warning("Corpus empty after dictionary filtering. Cannot train LDA.")
            return None, None, dictionary
        st.info(f"Training LDA model with {NUM_TOPICS} topics on {len(corpus)} documents...")
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=NUM_TOPICS,
            random_state=RANDOM_SEED,
            chunksize=100,
            passes=10,
            alpha='auto',
            eta='auto',
            iterations=100
        )
        st.success("LDA model training complete.")
        return lda_model, corpus, dictionary
    except Exception as e:
        st.error(f"Error during LDA model training: {e}")
        return None, None, None

# Utility to convert DataFrame to CSV
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

# ============================================================================== 
# ----------------------------- Streamlit App Layout ---------------------------
# ==============================================================================

st.set_page_config(layout="wide")
st.markdown("Upload an annual report (PDF) to perform NLP analysis.")

# Sidebar Upload
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Annual Report PDF", type="pdf")

# Proceed if PDF uploaded and NLTK ready
if uploaded_file is not None and nltk_ready:
    st.markdown("---")
    st.header(f"📊 Analysis Results for: {uploaded_file.name}")

    # --- Step 1: Extract Text ---
    with st.spinner("Step 1/8: Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
    if raw_text:
        st.success("Step 1 Complete: Text Extracted")
        with st.expander("Show Extracted Text Sample"):
            st.text(raw_text[:1000] + "...")

        # --- Step 2: Create Sentences DataFrame ---
        with st.spinner("Step 2/8: Structuring data into sentences..."):
            sentences = sent_tokenize(raw_text)
            df_sents = pd.DataFrame({"sentence": sentences})
            df_sents.index.name = "sent_id"
            st.success(f"Step 2 Complete: Found {len(df_sents)} sentences.")
            st.download_button(
                label="📥 Download Raw Sentences (CSV)",
                data=convert_df_to_csv(df_sents[["sentence"]]),
                file_name=f"{uploaded_file.name}_sentences_raw.csv",
                mime='text/csv',
            )
            with st.expander("Show Sample Sentences"):
                st.dataframe(df_sents.head())

        # --- Step 3: Preprocessing ---
        with st.spinner("Step 3/8: Cleaning and preprocessing text..."):
            df_sents['clean_sentence'] = df_sents['sentence'].apply(basic_clean)
            df_sents['tokens_lemmatized'] = df_sents['sentence'].apply(lambda x: preprocess_words(x, do_lemmatize=True))
            all_tokens_lemmatized = [token for sublist in df_sents['tokens_lemmatized'] for token in sublist]
            st.success("Step 3 Complete: Text Preprocessed.")
            with st.expander("Show Sample Preprocessed Data"):
                st.write("**Original:**", df_sents['sentence'].iloc[5])
                st.write("**Cleaned:**", df_sents['clean_sentence'].iloc[5])
                st.write("**Tokens:**", df_sents['tokens_lemmatized'].iloc[5])

        # --- Step 4: Sentiment Analysis ---
        with st.spinner("Step 4/8: Calculating sentiment..."):
            sentiment_results = df_sents['sentence'].apply(calculate_sentiment)
            df_sents[['sentiment_polarity', 'sentiment_subjectivity']] = pd.DataFrame(sentiment_results.tolist(), index=df_sents.index)
            st.success("Step 4 Complete: Sentiment Calculated.")

            st.subheader("Sentiment Analysis Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Polarity", f"{df_sents['sentiment_polarity'].mean():.4f}")
                st.metric("Min Polarity", f"{df_sents['sentiment_polarity'].min():.4f}")
                st.metric("Max Polarity", f"{df_sents['sentiment_polarity'].max():.4f}")
            with col2:
                st.metric("Avg Subjectivity", f"{df_sents['sentiment_subjectivity'].mean():.4f}")
                st.metric("Min Subjectivity", f"{df_sents['sentiment_subjectivity'].min():.4f}")
                st.metric("Max Subjectivity", f"{df_sents['sentiment_subjectivity'].max():.4f}")

            st.download_button(
                label="📥 Download Sentences with Sentiment (CSV)",
                data=convert_df_to_csv(df_sents[['sentence', 'sentiment_polarity', 'sentiment_subjectivity']]),
                file_name=f"{uploaded_file.name}_sentences_sentiment.csv",
                mime='text/csv',
            )

        # --- Step 5-6: Word Frequency & Visualizations ---
        with st.spinner("Step 5-6/8: Analyzing word frequency and generating visualizations..."):
            if all_tokens_lemmatized:
                word_freq = Counter(all_tokens_lemmatized)
                top_30_words = word_freq.most_common(30)
                df_freq = pd.DataFrame(top_30_words, columns=['Word', 'Frequency'])
                st.subheader("Top 30 Most Frequent Words")
                st.dataframe(df_freq)
                fig_wordcloud = generate_wordcloud_figure(all_tokens_lemmatized)
                fig_barchart = generate_barchart_figure(top_30_words[:20])
                col1, col2 = st.columns(2)
                with col1:
                    if fig_wordcloud: st.pyplot(fig_wordcloud)
                with col2:
                    if fig_barchart: st.pyplot(fig_barchart)
            else:
                st.warning("No tokens found after preprocessing. Skipping visualizations.")

        # --- Step 7-8: Topic Modeling (LDA) ---
        with st.spinner("Step 7-8/8: Topic Modeling..."):
            sentence_tokens_for_lda = df_sents[df_sents['tokens_lemmatized'].apply(len) >= 5]['tokens_lemmatized'].tolist()
            used_sents_indices = df_sents[df_sents['tokens_lemmatized'].apply(len) >= 5].index
            if sentence_tokens_for_lda:
                lda_model, corpus, dictionary = train_lda_model(sentence_tokens_for_lda)
            else:
                st.warning("No sentences long enough for LDA.")
                lda_model, corpus, dictionary = None, None, None

            if lda_model and corpus and dictionary:
                st.subheader(f"Top 10 Words per Topic ({NUM_TOPICS} Topics)")
                topics_words = lda_model.show_topics(num_topics=NUM_TOPICS, num_words=10, formatted=False)
                topic_data = {f"Topic {i+1}": ", ".join([word for word, prop in topic]) for i, topic in topics_words}
                st.dataframe(pd.DataFrame.from_dict(topic_data, orient='index', columns=["Top Words"]))

                topic_distributions = []
                for doc_bow in corpus:
                    doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
                    topic_vector = np.zeros(NUM_TOPICS)
                    for topic_id, prob in doc_topics:
                        topic_vector[topic_id] = prob
                    topic_distributions.append(topic_vector)
                topic_df = pd.DataFrame(topic_distributions, columns=[f"Topic_{t+1}_Prob" for t in range(NUM_TOPICS)])
                df_sents_with_topics = df_sents.loc[used_sents_indices].reset_index().join(topic_df)
                st.download_button(
                    label="📥 Download Sentences with Topic Probabilities (CSV)",
                    data=convert_df_to_csv(df_sents_with_topics),
                    file_name=f"{uploaded_file.name}_sentences_topics.csv",
                    mime='text/csv',
                )
            else:
                st.warning("LDA Model could not be trained.")

        st.markdown("---")
        st.balloons()
        st.header("✅ Analysis Complete!")

    else:
        st.error("Failed to extract text from the PDF. Please try another file.")

elif not nltk_ready:
    st.error("NLTK resource download failed. Please check your internet connection and restart the app.")
else:
    st.info("Please upload a PDF file using the sidebar to start the analysis.")
