#to send list of messages in dynamic prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
 # this will not print the values of place holders
'''chat_template=ChatPromptTemplate([
    SystemMessage(content='you are a helpful {domain} expert'),
    HumanMessage(content='explain in simple terms, what is {topic}') 
])'''

chat_template=ChatPromptTemplate([
    ('system','you are a helpful {domain} expert'),
    ('human','explain in simple terms, what is {topic}')
])
prompt=chat_template.invoke({'domain':'cricket','topic':'dusra'})
print(prompt)