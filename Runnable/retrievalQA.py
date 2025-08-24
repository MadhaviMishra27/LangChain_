# PDF reader using retrievalQA chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
# Load the document
loader = TextLoader("exm.txt")  # Your input text file
documents = loader.load()
# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
# Initialize embeddings with HuggingFaceEmbeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
# Store embeddings in FAISS vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)
# Create retriever from vectorstore
retriever = vectorstore.as_retriever()
# Initialize Hugging Face causal LM pipeline
model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    device=-1  # Use 0 if GPU available
)
llm = HuggingFacePipeline(pipeline=pipe)
# Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever,return_source_documents=False  # Set True if you want to see source docs
)
# Query the chain directly
query = "What are the key takeaways from the document?"
answer = qa_chain.invoke({"query": query})
print("Answer:", answer)
