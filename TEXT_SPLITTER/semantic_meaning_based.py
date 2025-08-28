from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Custom wrapper class providing embed_documents interface expected by SemanticChunker
class HuggingFaceEmbedderWrapper:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device="cpu"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )

    def embed_documents(self, texts):
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of text strings")
        return self.embeddings.embed_documents(texts)

# Instantiate wrapper
hf_embedder = HuggingFaceEmbedderWrapper()

# Create SemanticChunker
text_splitter = SemanticChunker(
    hf_embedder,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

# Example text
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. 
The sun was bright, and the air smelled of earth and fresh grass. 
The Indian Premier League (IPL) is the biggest cricket league in the world. 
People all over the world watch the matches and cheer for their favourite teams.
"""

# Create semantic documents
docs = text_splitter.create_documents([sample])
print(docs)
