from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion",
    huggingfacehub_api_token=hf_token,
    temperature=0.2,
    max_new_tokens=10
)

chat_model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="Answer in 1 short sentence only."),
    HumanMessage(content="Who invented the bulb for the very first time?")
]
response = chat_model.invoke(messages)
print(response.content.strip())
