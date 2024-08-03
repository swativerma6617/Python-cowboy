import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Function to preprocess text: removes punctuation, converts to lowercase, and removes stopwords
def preprocess_text(text):
    # Remove punctuation and lowercase the text
    text = re.sub(r'[^\w\s]', '', text).lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    return ' '.join([word for word in text.split() if word not in stop_words])

# Function to combine short headings with their following paragraphs
def combine_headings_with_paragraphs(document):
    # Split the document into paragraphs based on double newlines
    paragraphs = document.split('\n\n')
    combined_paragraphs = []
    temp = ""

    for para in paragraphs:
        stripped_para = para.strip()
        # Check if the paragraph is a heading (short and doesn't end with a period)
        if len(stripped_para.split()) < 10 and not re.match(r'.*\.\s*$', stripped_para):
            temp += " " + stripped_para
        else:
            # Combine the heading with the following paragraph
            if temp:
                combined_paragraphs.append(temp.strip() + "\n" + stripped_para)
                temp = ""
            else:
                combined_paragraphs.append(stripped_para)
    
    # Add any remaining text in temp as the last paragraph
    if temp:
        combined_paragraphs.append(temp.strip())

    # Return the list of combined paragraphs
    return [para for para in combined_paragraphs if para.strip()]

# Function to find the top N relevant paragraphs based on a query
def get_relevant_paragraphs(document, query, top_n=3):
    # Combine headings with their following paragraphs
    paragraphs = combine_headings_with_paragraphs(document)
    # Preprocess the paragraphs and the query
    preprocessed_paragraphs = [preprocess_text(paragraph) for paragraph in paragraphs]
    preprocessed_query = preprocess_text(query)
    
    # TF-IDF vectorization to convert text to numerical features
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_paragraphs + [preprocessed_query])
    
    # Extract the query vector (the last vector) and the paragraph vectors
    query_vector = tfidf_matrix[-1]
    paragraph_vectors = tfidf_matrix[:-1]
    # Compute cosine similarity between the query and each paragraph
    similarities = cosine_similarity(query_vector, paragraph_vectors).flatten()

    # Combine similarity scores with keyword frequency for a weighted scoring
    keywords = re.findall(r'\b\w+\b', preprocessed_query)
    scores = []
    for i, paragraph in enumerate(preprocessed_paragraphs):
        similarity_score = similarities[i]
        keyword_score = sum(1 for keyword in keywords if keyword in paragraph)
        # Calculate combined score with weights for similarity and keyword frequency
        combined_score = similarity_score + 0.5 * keyword_score  # Adjust weight as needed
        scores.append((combined_score, i))
    
    # Sort paragraphs by combined scores in descending order
    scores.sort(reverse=True, key=lambda x: x[0])
    # Get the indices of the top N paragraphs
    top_indices = [i for _, i in scores[:top_n]]
    
    # Return the top N paragraphs
    return [paragraphs[i] for i in top_indices]

# Function to load the document from a file
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Load the HR policy document
document = load_document('/Users/swativerma/HR POLICY/HR policy document.txt')

# Function to answer a query by finding the top relevant paragraphs
def answer_query(query):
    # Find the top relevant paragraphs
    relevant_paragraphs = get_relevant_paragraphs(document, query)
    # Print the query and the top 3 relevant paragraphs
    print(f"Query: {query}\n")
    print("Top 3 relevant paragraphs:")
    for i, paragraph in enumerate(relevant_paragraphs, 1):
        print(f"\n{i}. {paragraph}")

# Interactive loop for user queries
print("Enter your questions about the HR policy. Type 'quit' to exit.")
while True:
    # Take user input for a query
    user_query = input("\nEnter your question: ")
    if user_query.lower() == 'quit':
        break
    # Answer the query
    answer_query(user_query)
