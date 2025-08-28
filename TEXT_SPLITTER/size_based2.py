from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
loader=PyPDFLoader(r'C:\Users\91739\OneDrive\Desktop\LangChain\DOCUMENT_LOADER\Gen_AI_&_Agentic_AI.pdf')
docs=loader.load()
#because docs is a list of Document objects, but split_text() expects a single string input.
#Extract text content from the documents first, then combine or process each separately.
docs= "\n".join([doc.page_content for doc in docs])
splitter= CharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separator=''
)
result=splitter.split_text(docs)
print(result[0])