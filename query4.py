import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the NLTK data required for text processing
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text into sentences
    sentences = nltk.sent_tokenize(text)
    return sentences

# Function to get the top N relevant paragraphs based on a query
def get_top_n_paragraphs(document, query, n=3):
    paragraphs = preprocess_text(document)
    
    # Combine query and paragraphs for vectorization
    combined_text = [query] + paragraphs
    
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    
    # Compute cosine similarity between the query vector and paragraph vectors
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get the indices of the top N similar paragraphs
    top_n_indices = cosine_similarities.argsort()[-n:][::-1]
    
    # Retrieve and return the top N paragraphs
    top_n_paragraphs = [paragraphs[i] for i in top_n_indices]
    return top_n_paragraphs

# Example usage
if __name__ == "__main__":
    # Load the HR policy document
    with open('HR policy document.txt', 'r') as file:
        hr_policy_document = file.read()
    
    while True:
        # User query
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        # Get the top 3 relevant paragraphs
        top_paragraphs = get_top_n_paragraphs(hr_policy_document, user_query, n=3)
        
        # Display the results
        for i, paragraph in enumerate(top_paragraphs, 1):
            print(f"\nParagraph {i}:\n{paragraph}\n")

