from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.decomposition import PCA
import numpy as np

# Load the embedding model
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# At least 2 documents for PCA to work
docs = [
    "Delhi is the capital of India.",
    "Mumbai is a major financial city in India."
]

# Get 384-dim embeddings
vectors = embedding.embed_documents(docs)

# Reduce to 2 dimensions (since only 2 samples)
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(np.array(vectors))

# Print reduced vectors
for i, vec in enumerate(reduced_vectors):
    print(f"Reduced vector {i + 1}:", vec)
