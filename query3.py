import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def read_document(file_path):
    with open(file_path, 'r') as file:
        document = file.read()
    return document

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def find_relevant_paragraphs(document, query, top_n=3):
    paragraphs = document.split('\n\n')
    preprocessed_paragraphs = [preprocess_text(para) for para in paragraphs]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_paragraphs)
    
    query_vec = vectorizer.transform([preprocess_text(query)])
    relevance_scores = np.dot(tfidf_matrix, query_vec.T).toarray()
    
    top_indices = relevance_scores.flatten().argsort()[-top_n:][::-1]
    top_paragraphs = [paragraphs[i] for i in top_indices]
    
    return top_paragraphs

def main():
    document_path = '/Users/swativerma/HR POLICY/HR policy document.txt'
    document = read_document(document_path)
    
    query = input("Enter your query: ")
    top_paragraphs = find_relevant_paragraphs(document, query)
    
    print("\nTop Relevant Paragraphs:\n")
    for i, para in enumerate(top_paragraphs, 1):
        print(f"Paragraph {i}:\n{para}\n")

if __name__ == "__main__":
    main()
