from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "I love machine learning",
    "Machine learning is powerful"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

print(vectorizer.get_feature_names_out())
print(X.toarray())