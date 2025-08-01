from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load embedding model
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Step 2: Input documents
documents = [
    "Delhi is the capital of India.",
    "Mumbai is a major financial city in India.",
    "Tokyo is the capital of Japan.",
    "The Ganga is a holy river in India.",
    "Mount Everest is the highest mountain on Earth."
]

# Step 3: Embed documents
doc_vectors = embedding.embed_documents(documents)

# Step 4: Embed the query
query = "Which city is the capital of India?"
query_vector = embedding.embed_query(query)

# Step 5: Compute cosine similarity
similarities = cosine_similarity([query_vector], doc_vectors)[0]

# Step 6: Print all similarity scores
print("\nSimilarity scores (index, score):")
print(list(enumerate(similarities)))

# Step 7: Get most similar document
index, score = max(enumerate(similarities), key=lambda x: x[1])
print(f"\nMost similar document: [{index}] '{documents[index]}' (Score: {score:.4f})")

# Step 8: Rank and display all documents by similarity
ranked_results = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

print("\nTop similar documents:")
for doc, score in ranked_results:
    print(f"Score: {score:.4f} | Document: {doc}")
