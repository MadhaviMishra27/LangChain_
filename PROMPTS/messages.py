from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # or any other supported model
    task="text-generation",
    model_kwargs={"api_key": api_key},
    temperature=0.4,
    max_new_tokens=50
)
model = ChatHuggingFace(llm=llm)
messages=[
    SystemMessage(content='you are a  helpful assistent'),
    HumanMessage(content='tell me about LangChain')
]
result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)