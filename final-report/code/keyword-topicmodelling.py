#!/bin/env python3.9

# imports
import os
import string
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import pytextrank
from gensim import corpora
from gensim.models import LdaModel, Word2Vec, CoherenceModel

# requires gensim version 3.8.3
# from gensim.summarization import summarize
from transformers import pipeline

import seaborn as sns

import sqlite3

# download libraries for preprocessing corpus
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Add PyTextRank to SpaCy pipeline
nlp.add_pipe("textrank")
# load a summmarizer so we only have to do this once 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# corpus loader for when the corpus is stored as .txt files in folder input in the same directory as this script
"""
def load_texts_from_folder(folder_path='./input'):
    texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):  # assuming text files
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts
"""

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords
    words = word_tokenize(text.lower())  # Tokenize and make lowercase
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(lemmatized_words)

def extractive_summarization(text, limit_phrases=15, limit_sentences=7):
    """
    Perform extractive summarization using PyTextRank and SpaCy.

    Parameters:
    - text (str): The text to summarize.
    - limit_phrases (int): The maximum number of key phrases to extract (default is 15).
    - limit_sentences (int): The maximum number of sentences to include in the summary (default is 3).

    Returns:
    - str: The extracted summary.
    """
    try:
        # Process the document
        doc = nlp(text)
        
        # Extract the summary sentences using PyTextRank
        summary = "\n".join([str(sent) for sent in doc._.textrank.summary(limit_phrases=limit_phrases, limit_sentences=limit_sentences)])

        return summary
    except Exception as e:
        return f"Error during extractive summarization: {str(e)}"

# tf-idf computation
# computes keyword computation based on the tf-idf rule
# Function to perform keyword extraction using TF-IDF
def extract_keywords(df, text_column, top_n=5):
    """
    Extract top N keywords from the text using TF-IDF.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing the text data.
    - top_n (int): Number of top keywords to extract.
    
    Returns:
    - List of keywords (top N keywords for each row).
    """
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the processed excerpts
    X = vectorizer.fit_transform(df[text_column])

    terms = vectorizer.get_feature_names_out()

    tfidf_matrix = X.toarray()
    top_k_terms = []
    
    # Loop over each document (row in tfidf_matrix)
    for i, row in enumerate(tfidf_matrix):
        # Create a list of (term, score) tuples for this document
        term_scores = list(zip(terms, row))
        
        # Sort by score in descending order and get the top k terms
        sorted_term_scores = sorted(term_scores, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Append the top k terms to the result list
        # Here we are discarding the number associated to the importance of the term -- we may find that this information is useful to use in a model
        top_k_terms.append([x for x, _ in sorted_term_scores])
    return top_k_terms

# Compute topic modellings
# Function to perform topic modeling (LDA) on the text
def perform_topic_modeling(df, text_column, num_topics=3):
    """
    Perform topic modeling on the processed text and return a list of topics for each row.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing the processed text.
    - num_topics (int): Number of topics to extract.
    
    Returns:
    - List of topics (one topic for each row).
    """
    # Tokenize the text in the specified column (split by whitespace)
    df['tokens'] = df[text_column].apply(lambda x: x.lower().split())  # Basic whitespace-based tokenization
    
    # Create a dictionary and corpus for LDA
    dictionary = corpora.Dictionary(df['tokens'])
    corpus = [dictionary.doc2bow(text) for text in df['tokens']]
    
    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    model_list = []
    coherence_values = []
    
    # Try training lda_model over several different topic numbers to determine the ideal number of topics
    for num_topics in range(2, 7):
        lda_model_ranked = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15)
        model_list.append(lda_model_ranked)
        coherence_model_lda = CoherenceModel(model=lda_model_ranked, texts=df['tokens'], dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model_lda.get_coherence())
    optimal_model = model_list[coherence_values.index(max(coherence_values))]
    
    # Print the topics and their associated words
    print("LDA Topics:")
    topics = optimal_model.show_topics(num_topics=num_topics, num_words=5, formatted=True)
    for topic_id, topic in topics:
        print(f"Topic #{topic_id}:")
        print(topic)
        print("-" * 50)

    # Get the most likely topic for each document
    def get_document_topic(corpus, model):
        return [sorted(model.get_document_topics(doc), key=lambda x: x[1], reverse=True)[0][0] for doc in corpus]
    
    topics = get_document_topic(corpus, lda_model)
    return topics

# # Function to get document embedding by averaging word embeddings
# def embed_document(tokens, preprocessed_corpus, model=None):
#     if model is None:
#         model = Word2Vec(sentences=preprocessed_corpus, vector_size=100, window=5, min_count=1, workers=4)
#     embeddings = [model.wv[word] for word in tokens if word in model.wv]
#     if len(embeddings) == 0:
#         return np.zeros(model.vector_size)
#     return np.mean(embeddings, axis=0)

# # Return assigned cluster for each document
# def cluster_documents(num_clusters, doc_embeddings, random_state=42):
#     kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
#     kmeans.fit(doc_embeddings)
#     return kmeans.labels_

# def visualize_document_embeddings(corpus, cluster_labels, doc_embeddings):
#     print("\nDocument Clusters and Topics:")
#     for i, doc in enumerate(corpus):
#         print(f"Document {i + 1}:")
#         print(f" - Cluster: {cluster_labels[i]}")
#         print(f" - Text: {doc}")

#     tsne = TSNE(n_components=2, random_state=None, perplexity=5)
#     reduced_embeddings = tsne.fit_transform(doc_embeddings)

#     # Alternatively, use PCA (faster but linear):
#     # pca = PCA(n_components=2)
#     # reduced_embeddings = pca.fit_transform(doc_embeddings)

#     df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
#     df['Cluster'] = cluster_labels
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(data=df, x='x', y='y', hue='Cluster', palette='Set1', s=100, marker='o', alpha=0.7)

#     # Adding titles and labels
#     plt.title('Document Clusters Visualization (t-SNE))', fontsize=16)
#     plt.xlabel('t-SNE Component 1', fontsize=12)
#     plt.ylabel('t-SNE Component 2', fontsize=12)
#     plt.legend(title="Cluster", loc='best')

#     plt.show()
#     plt.savefig('cluster-visualization.png', format='png')


## Script settings
source_db_path = 'combined.db'  # Replace with your actual SQLite DB path
target_db_path = 'combined.db'

#### Load document corpus
## Connect to the source DB
conn = sqlite3.connect(source_db_path)
## Query from the database
query = """SELECT * FROM processdDB WHERE tag IN ('Business', 'RF', 'URFC', 'Cybersecurity', 'Properties', 'MDAFC', 'QQDMR', 'FSSD') AND value IS NOT NULL;
"""
## Construct dataframe from the query
df = pd.read_sql(query, conn)
## Disconnect from the db
conn.close()

# df = pd.DataFrame(aapl_test_data)
raw_text_column = 'value'

## Preprocess text in dataframe to remove stopwords and standardize formatting
df['cleaned_text'] = df[raw_text_column].apply(preprocess_text)
## Applg extractive summarization to remove redundant words (e.g. business, fiscal, etc.)
df['extracted_summary'] = df[raw_text_column].apply(extractive_summarization)
# df['abstractive_summary'] = df['text'].apply(abstractive_summarization)
## For each document, we want to compute its keywords and topics
# Apply the functions to the DataFrame
df['cleaned_text_topic'] = perform_topic_modeling(df, 'cleaned_text', num_topics=6)
df['raw_text_keywords'] = extract_keywords(df, raw_text_column, top_n=5)

# Raw text -> extracted summary -> preprocessed to clean summary
df['cleaned_summary'] = df['extracted_summary'].apply(preprocess_text)

df['extracted_summary_keywords'] = extract_keywords(df, 'cleaned_summary', top_n=5)

#### Store results
## serialize all entries as JSON-encoded strings
# Later we will want to deserialize all entries using json.loads
df_encoded = df.applymap(json.dumps)
## Connect to the target DB
conn = sqlite3.connect(target_db_path)
## Stage changes to db
# Parameters:
# - 'if_exists' can be 'replace', 'append', or 'fail'.
#   - 'replace' will drop the table if it exists and create a new one.
#   - 'append' will add the data to an existing table.
#   - 'fail' will do nothing if the table already exists.
df_encoded.to_sql('analyzedDB', conn, if_exists='replace', index=False)
## Commit changes and close the connection
conn.commit()  # This is optional, as `to_sql()` commits by default
conn.close()