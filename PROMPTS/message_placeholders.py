# to keep history of previous days
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
# chat template
chat_template=ChatPromptTemplate([
    ('system','you are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])
chat_history=[]
#load chat history
with open(r'C:\Users\91739\OneDrive\Desktop\LangChain\PROMPTS\chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt=chat_template.invoke({'chat_history':chat_history,'query':'where is my refund?'})
print(prompt)