from langchain_community.document_loaders import PyPDFLoader
loader= PyPDFLoader(r'C:\Users\91739\OneDrive\Desktop\LangChain\DOCUMENT_LOADER\Gen_AI_&_Agentic_AI.pdf')
docs=loader.load()
print(len(docs))
print(docs[1].page_content)
print(docs[0].metadata)