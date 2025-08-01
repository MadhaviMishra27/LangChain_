from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
# static prompt: each time new prompt
# Load token from .env file
load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit UI
st.title("Research Tool")
user_input = st.text_input("Enter your prompt")

if st.button("Summarize") and user_input:
    try:
        # Initialize the endpoint
        hf_endpoint = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="text-generation",
            api_key=api_key,
        )

        # Wrap it in ChatHuggingFace
        model = ChatHuggingFace(llm=hf_endpoint)

        # Invoke the model
        result = model.invoke(user_input)
        st.write(result.content)

    except Exception as e:
        st.error(f"Error during model call: {e}")
