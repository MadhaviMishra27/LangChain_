#when it is not much useful to take prompt from the user, as output is sensitive to prompt very much
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
api_key=os.getenv("HUGGINGFACE_API_TOKEN")
# Create a new instance of the ChatHuggingFace class
#STREMLIT LAYOUT
st.title("Research Paper Summarization")
paper=st.selectbox(
    "which paper would you like to summarize?",
    ["Attention is all you need","BERT","seq2seq+attention"]
)
style=st.selectbox(
    "select the style of summary:",
    ["begginer-friendly","technical","mathematical","code-oriented"]             
)
length=st.selectbox(
    "select explanation length:",
    ["short(1-2 paragraph)","medium(3-5 paragraph)","long(detailed explanation)"]
)

template=load_prompt('template.json')

if st.button("Generate Summary"):
    try:
        prompt=template.format(
            paper=paper,
            style=style,
            length=length
        )
        hf_endpoint=HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="text-generation",
            model_kwargs={"api_key": api_key},  # ← move api_key inside model_kwargs
            temperature=0.7,
        )
        model=ChatHuggingFace(llm=hf_endpoint)
        result=model.invoke(prompt)
        st.subheader("Summary")
        st.write(result.content)
    except Exception as e:
        st.error(f"error during model call:{e}")
    
