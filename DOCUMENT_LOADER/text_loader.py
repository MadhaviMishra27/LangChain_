from langchain_community.document_loaders import TextLoader
loader=TextLoader(r'C:\Users\91739\OneDrive\Desktop\LangChain\DOCUMENT_LOADER\indianEco.txt',encoding='utf-8')
docs=loader.load()
#print(docs)
print(type(docs))
print(len(docs))
print(docs[0])