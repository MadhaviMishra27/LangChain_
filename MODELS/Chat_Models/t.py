from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Print to confirm token is loaded
print("TOKEN CHECK:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Set up the Hugging Face inference endpoint
llm = HuggingFaceEndpoint(
    repo_id="bigscience/bloomz-560m",
    temperature=0.5,
    task="text-generation",  # ✅ correct task format
    max_new_tokens=200,
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")  # ✅ required for cloud API
)

# Wrap it in Chat interface
#model =HuggingFaceEndpoint(llm=llm)

# Send the prompt
#response = model.invoke("Who was the first president of India?")
response = llm.invoke("Who was the first president of India?")
print("Response:", response)
