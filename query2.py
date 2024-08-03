import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the document
file_path = '/Users/swativerma/HR POLICY/HR policy document.txt'
with open(file_path, 'r') as file:
    document = file.read()

# Preprocess and split the document into paragraphs
def preprocess(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
preprocessed_paragraphs = [preprocess(p) for p in paragraphs]

# Initialize the TF-IDF vectorizer and fit it on the paragraphs
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=2)
vectors = vectorizer.fit_transform(preprocessed_paragraphs)

# Function to fetch the top 3 relevant paragraphs
def fetch_relevant_paragraphs(query, top_n=3):
    preprocessed_query = preprocess(query)
    query_vec = vectorizer.transform([preprocessed_query])
    similarity = cosine_similarity(query_vec, vectors)[0]
    
    # Adjust similarity scores based on paragraph position
    position_boost = np.linspace(1.2, 1.0, len(paragraphs))
    adjusted_similarity = similarity * position_boost
    
    top_indices = adjusted_similarity.argsort()[-top_n:][::-1]
    return [paragraphs[i] for i in top_indices]

# Interactive loop for user queries
def interactive_query():
    while True:
        query = input("Enter your query (or type 'exit' to stop): ")
        if query.lower() == 'exit':
            break
        relevant_paragraphs = fetch_relevant_paragraphs(query)
        for i, para in enumerate(relevant_paragraphs, 1):
            print(f"\nParagraph {i}:\n{para}\n")

# Run the interactive query function
interactive_query()
