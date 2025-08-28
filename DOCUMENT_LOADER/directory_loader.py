from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
loader=DirectoryLoader(
    path=r'C:\Users\91739\OneDrive\Desktop\LangChain\DOCUMENT_LOADER\doc',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
docs=loader.lazy_load()
#print(len(docs))
for document in docs:
    print(document.metadata)