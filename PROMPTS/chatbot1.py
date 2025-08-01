from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_TOKEN")
# Initialize the HuggingFace endpoint with your model
hf_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # or any other supported model
    task="text-generation",
    model_kwargs={"api_key": api_key},
    temperature=0.4,
    max_new_tokens=50
)

chat_history=[
    SystemMessage(content='you are a helpful ai assistent.')
]

# Initialize the chat model with system prompt to keep answers clean
# Pass it to ChatHuggingFace
model = ChatHuggingFace(
    llm=hf_endpoint,
    system_prompt="You are a concise and helpful assistant. Answer clearly and briefly without repeating the question.avoid unnecessarily text"
)

#chatbot will run infinitely until user types exit
# Chatbot loop
print("🤖 Chatbot ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == 'exit':
        break

    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(result)  # ✅ Append the model's response
    print("AI:", result.content.strip())