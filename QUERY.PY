import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

def get_chunks(document, chunk_size=3):
    paragraphs = document.split('\n\n')
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = ' '.join(paragraphs[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def get_relevant_paragraphs(document, query, top_n=3):
    chunks = get_chunks(document)
    preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
    preprocessed_query = preprocess_text(query)
    
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.85, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_chunks + [preprocessed_query])
    
    query_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Load the document from a file
document = load_document('/Users/swativerma/HR POLICY/HR policy document.txt')

def answer_query(query):
    relevant_paragraphs = get_relevant_paragraphs(document, query)
    print(f"Query: {query}\n")
    print("Top 3 relevant paragraphs:")
    for i, paragraph in enumerate(relevant_paragraphs, 1):
        print(f"\n{i}. {paragraph}")

# Interactive loop for user queries
print("Enter your questions about the HR policy. Type 'quit' to exit.")
while True:
    user_query = input("\nEnter your question: ")
    if user_query.lower() == 'quit':
        break
    answer_query(user_query)