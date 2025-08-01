import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint

# Load environment variables
load_dotenv()

# Get token from .env
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Ensure the token is loaded
if not hf_token:
    raise ValueError("❌ HUGGINGFACEHUB_API_TOKEN not found in .env")

# Initialize the Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small",       # text2text model
    task="text2text-generation",
    huggingfacehub_api_token=hf_token
)

# Invoke the model
prompt = "What is the capital of India?"
result = llm.invoke(prompt)

# Output the result
print("Answer:", result)
