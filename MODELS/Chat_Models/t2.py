from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("Hugging Face API token not set. Check .env file.")

print("TOKEN CHECK:", token)

# Create the HF inference endpoint with a supported text-generation model
llm = HuggingFaceEndpoint(
    repo_id="bigscience/bloomz-560m",
    temperature=0.5,
    max_new_tokens=200,
    huggingfacehub_api_token=token,
)

# Invoke the model with error handling
try:
    response = llm.invoke("Who was the first president of India?")
    print("Response:", response)
except Exception as e:
    import traceback
    print("Inference error:", str(e))
    traceback.print_exc()
