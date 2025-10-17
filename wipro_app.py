# -*- coding: utf-8 -*-
"""
NLP Mini Project: Annual Report Analysis (Streamlit App)

This Streamlit app performs NLP tasks on an uploaded company annual report PDF.
Tasks include text extraction, preprocessing, sentiment analysis, word frequency analysis,
word cloud generation, and topic modeling using LDA.
"""
import nltk

# Download stopwords only if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ==============================================================================
# 0. Import Core Libraries & Setup
# ==============================================================================
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
import io # For download buttons

# ==============================================================================
# NLTK Downloads (Conditional) & Config
# ==============================================================================
# Function to download NLTK data if not already present
@st.cache_resource # Cache resource download
def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet.zip')
        nltk.data.find('corpora/stopwords.zip')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/omw-1.4.zip')
        st.sidebar.success("NLTK resources found.")
        return True
    except nltk.downloader.DownloadError:
        st.sidebar.warning("Downloading NLTK resources...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            st.sidebar.success("NLTK resources downloaded.")
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to download NLTK resources: {e}")
            return False
    except Exception as e:
        st.sidebar.error(f"An error occurred during NLTK check: {e}")
        return False

# --- Model Parameters ---
NUM_TOPICS = 10
RANDOM_SEED = 42

# --- Suppress pdfminer warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='pdfminer.*')
logging.getLogger("pdfminer").setLevel(logging.WARNING)

# ==============================================================================
# Helper & Processing Functions (Cached)
# ==============================================================================

# Task 1: PDF Extraction
@st.cache_data # Cache results based on input bytes
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
# Define these outside other functions if they are used globally
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

# Task 4: Sentiment Analysis Function
def calculate_sentiment(text):
    """Calculates polarity and subjectivity using TextBlob."""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception:
        return np.nan, np.nan

# Task 6: Plotting Functions
def generate_wordcloud_figure(tokens):
    """Generates a matplotlib figure for the WordCloud."""
    wordcloud_text = " ".join(tokens)
    if not wordcloud_text.strip():
        st.warning("Cannot generate WordCloud: No processable words found.")
        return None
    try:
        wordcloud = WordCloud(width=1000, height=500, background_color='white',
                              colormap='viridis', max_words=150, random_state=RANDOM_SEED).generate(wordcloud_text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud - Annual Report Analysis', fontsize=16)
        plt.tight_layout(pad=0)
        return fig
    except Exception as e:
        st.error(f"Error generating WordCloud: {e}")
        return None

def generate_barchart_figure(top_words):
    """Generates a matplotlib figure for the top words bar chart."""
    if not top_words:
        st.warning("Cannot generate Bar Chart: No frequent words found.")
        return None
    try:
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
    except Exception as e:
        st.error(f"Error generating Bar Chart: {e}")
        return None

# Task 8: LDA Model Training Function
@st.cache_data # Cache the LDA model and results
def train_lda_model(sentence_tokens_for_lda):
    """Trains the LDA model and returns the model, corpus, and dictionary."""
    if not sentence_tokens_for_lda:
        return None, None, None
    try:
        dictionary = corpora.Dictionary(sentence_tokens_for_lda)
        dictionary.filter_extremes(no_below=5, no_above=0.7, keep_n=10000)
        corpus = [dictionary.doc2bow(text) for text in sentence_tokens_for_lda]
        if not corpus: # Check if corpus is empty after filtering
             st.warning("Corpus is empty after dictionary filtering. Cannot train LDA.")
             return None, None, dictionary # Return dictionary even if corpus is empty

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

# Utility to convert DataFrame to CSV Bytes for download
@st.cache_data
def convert_df_to_csv(df):
   """Converts a Pandas DataFrame to CSV bytes."""
   return df.to_csv(index=True).encode('utf-8')

# ==============================================================================
# Streamlit App Layout
# ==============================================================================

st.set_page_config(layout="wide") # Use wide layout
st.title("📄 NLP Analysis of Annual Reports")
st.markdown("Upload an annual report (PDF) to perform NLP analysis.")

# --- Sidebar for Upload and Settings ---
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Annual Report PDF", type="pdf")

# --- NLTK Check ---
nltk_ready = download_nltk_data()

# --- Main Processing Area ---
if uploaded_file is not None and nltk_ready:
    st.markdown("---")
    st.header(f"📊 Analysis Results for: {uploaded_file.name}")

    # --- Task 1: Extract Text ---
    with st.spinner("Step 1/8: Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    if raw_text:
        st.success("Step 1 Complete: Text Extracted")
        with st.expander("Show Extracted Text Sample"):
            st.text(raw_text[:1000] + "...") # Show a sample

        # --- Task 2: Create DataFrames ---
        with st.spinner("Step 2/8: Structuring data into sentences..."):
            sentences = sent_tokenize(raw_text)
            df_sents = pd.DataFrame({"sentence": sentences})
            df_sents.index.name = "sent_id"
            st.success(f"Step 2 Complete: Found {len(df_sents)} sentences.")
            st.download_button(
                label="📥 Download Raw Sentences (CSV)",
                data=convert_df_to_csv(df_sents[["sentence"]]), # Only sentence column
                file_name=f"{uploaded_file.name}_sentences_raw.csv",
                mime='text/csv',
            )
            with st.expander("Show Sample Sentences"):
                st.dataframe(df_sents.head())

        # --- Task 3: Preprocessing ---
        with st.spinner("Step 3/8: Cleaning and preprocessing text..."):
            df_sents['clean_sentence'] = df_sents['sentence'].apply(basic_clean)
            df_sents['tokens_lemmatized'] = df_sents['sentence'].apply(lambda x: preprocess_words(x, do_lemmatize=True))
            all_tokens_lemmatized = [token for sublist in df_sents['tokens_lemmatized'] for token in sublist] # Flatten list
            st.success("Step 3 Complete: Text Preprocessed.")
            with st.expander("Show Sample Preprocessed Data"):
                st.write("**Original:**", df_sents['sentence'].iloc[5])
                st.write("**Cleaned:**", df_sents['clean_sentence'].iloc[5])
                st.write("**Tokens:**", df_sents['tokens_lemmatized'].iloc[5])

        # --- Task 4: Sentiment Analysis ---
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
                data=convert_df_to_csv(df_sents[['sentence', 'sentiment_polarity', 'sentiment_subjectivity']]), # Select relevant columns
                file_name=f"{uploaded_file.name}_sentences_sentiment.csv",
                mime='text/csv',
            )
            with st.expander("Show Sample Sentences with Sentiment"):
                st.dataframe(df_sents[['sentence', 'sentiment_polarity', 'sentiment_subjectivity']].head())


        # --- Task 6: Word Frequency & Visualizations ---
        # Task 5 is implicitly done by creating all_tokens_lemmatized in Task 3
        with st.spinner("Step 5-6/8: Analyzing word frequency and generating visualizations..."):
            st.subheader("Word Frequency Analysis")
            if all_tokens_lemmatized:
                word_freq = Counter(all_tokens_lemmatized)
                top_30_words = word_freq.most_common(30)
                df_freq = pd.DataFrame(top_30_words, columns=['Word', 'Frequency'])

                st.write("Top 30 Most Frequent Words:")
                st.dataframe(df_freq)

                # Generate Plots
                fig_wordcloud = generate_wordcloud_figure(all_tokens_lemmatized)
                fig_barchart = generate_barchart_figure(top_30_words[:20]) # Top 20 for bar chart

                col1, col2 = st.columns(2)
                with col1:
                    if fig_wordcloud:
                        st.pyplot(fig_wordcloud)
                with col2:
                    if fig_barchart:
                        st.pyplot(fig_barchart)
                st.success("Step 5-6 Complete: Frequency Analyzed and Visualizations Generated.")
            else:
                st.warning("No tokens found after preprocessing to calculate frequency.")
                st.warning("Skipping Word Cloud and Bar Chart.")


        # --- Task 7 & 8: Topic Modeling ---
        with st.spinner("Step 7-8/8: Preparing data and performing Topic Modeling (LDA)..."):
            st.subheader("Topic Modeling (LDA)")
            min_sentence_length = 5
            sentence_tokens_for_lda = df_sents[df_sents['tokens_lemmatized'].apply(len) >= min_sentence_length]['tokens_lemmatized'].tolist()
            used_sents_indices = df_sents[df_sents['tokens_lemmatized'].apply(len) >= min_sentence_length].index

            if not sentence_tokens_for_lda:
                 st.warning(f"No sentences found with minimum length ({min_sentence_length} tokens). Cannot perform LDA.")
                 lda_model, corpus, dictionary = None, None, None
            else:
                st.info(f"Using {len(sentence_tokens_for_lda)} sentences for LDA.")
                # Train model (uses caching)
                lda_model, corpus, dictionary = train_lda_model(sentence_tokens_for_lda)

            if lda_model and corpus and dictionary:
                 st.write(f"**Top 10 Words per Topic (Total {NUM_TOPICS} Topics):**")
                 topics_words = lda_model.show_topics(num_topics=NUM_TOPICS, num_words=10, formatted=False)
                 topic_data = {}
                 for i, topic in topics_words:
                     topic_data[f"Topic {i+1}"] = ", ".join([word for word, prop in topic])
                 st.dataframe(pd.DataFrame.from_dict(topic_data, orient='index', columns=["Top Words"]))

                 # Calculate and prepare topic distributions for download
                 topic_distributions = []
                 for i, doc_bow in enumerate(corpus):
                     doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
                     topic_vector = np.zeros(NUM_TOPICS)
                     for topic_id, prob in doc_topics:
                         topic_vector[topic_id] = prob
                     topic_distributions.append(topic_vector)

                 topic_df = pd.DataFrame(topic_distributions, columns=[f"Topic_{t+1}_Prob" for t in range(NUM_TOPICS)])
                 df_sents_with_topics = df_sents.loc[used_sents_indices].reset_index().join(topic_df) # Join probs

                 st.download_button(
                    label="📥 Download Sentences with Topic Probabilities (CSV)",
                    data=convert_df_to_csv(df_sents_with_topics),
                    file_name=f"{uploaded_file.name}_sentences_topics.csv",
                    mime='text/csv',
                 )

                 with st.expander("Show Top Sentences per Topic"):
                    num_top_sentences = 3
                    for i in range(NUM_TOPICS):
                        topic_col = f"Topic_{i+1}_Prob"
                        st.write(f"--- Topic {i+1} ---")
                        top_sentences = df_sents_with_topics.sort_values(by=topic_col, ascending=False).head(num_top_sentences)
                        for index, row in top_sentences.iterrows():
                             st.write(f"  [Prob={row[topic_col]:.3f}] SentID {row['sent_id']}: {row['sentence'][:250]}...")
                 st.success("Step 7-8 Complete: Topic Modeling Done.")

            else:
                 st.warning("LDA Model could not be trained due to lack of sufficient data or errors.")

        st.markdown("---")
        st.balloons()
        st.header("✅ Analysis Complete!")

    else:
        st.error("Failed to extract text from the PDF. Please try another file.")

elif not nltk_ready:
    st.error("NLTK resource download failed. Please check your internet connection and restart the app.")

else:
    st.info("Please upload a PDF file using the sidebar to start the analysis.")
