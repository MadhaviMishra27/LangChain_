from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="conversational",
    huggingfacehub_api_token=hf_token,
    temperature=0.5,
    max_new_tokens=100
)

response = llm.invoke("prompt = <|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is the capital of India?\n<|assistant|>")
print(response)
