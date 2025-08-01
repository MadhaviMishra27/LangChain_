from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")  # ✅ required for cloud API
)

# Wrap it in Chat interface
model = ChatHuggingFace(llm=llm)

# Send the prompt
response = model.invoke("Who was the first president of India?")
print("Response:", response.content)
