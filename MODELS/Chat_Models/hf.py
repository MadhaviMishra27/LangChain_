from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
# Load the token from .env file
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# Initialize the Hugging Face endpoint with a chat model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion",  # Important: must be chat-compatible
    huggingfacehub_api_token=hf_token,
)
# Wrap with LangChain's ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)
# Ask a question
response = chat_model.invoke("What is the capital of India?")
print(response.content)
