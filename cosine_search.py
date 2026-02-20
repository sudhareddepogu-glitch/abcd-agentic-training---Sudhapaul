from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "I love machine learning",
    "Machine learning is powerful",
    "I enjoy coding in Python",
    "Python is great for data science"
]

# Step 1: Convert text to vectors
vectorizer = CountVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# Step 2: Take user query
query = input("Enter your search query: ")

# Step 3: Convert query to vector
query_vector = vectorizer.transform([query])

# Step 4: Compute cosine similarity
similarities = cosine_similarity(query_vector, doc_vectors)

# Step 5: Get best match
best_match_index = similarities.argmax()

print("\nMost similar document:")
print(documents[best_match_index])