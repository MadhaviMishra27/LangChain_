from langchain_community.document_loaders import CSVLoader
loader=CSVLoader(file_path=r'C:\Users\91739\OneDrive\Desktop\LangChain\DOCUMENT_LOADER\product_reviews_small.csv')
docs=loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0])