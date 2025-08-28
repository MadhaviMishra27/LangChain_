from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter


# Then call your chain on each chunk

# Load environment variables from .env file
load_dotenv()
hf_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Setup Hugging Face text generation pipeline with authentication
hf_pipe = pipeline(
    "text2text-generation",    # Use text2text-generation for T5 models
    model="google/flan-t5-base",
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    device=-1
)

# Wrap pipeline with LangChain
llm = HuggingFacePipeline(pipeline=hf_pipe)

prompt=PromptTemplate(
    template='answer the following que \n {question} from the following taxt- {text}',
    input_variables=['question','text']
)

parser=StrOutputParser()
# Define URL(s) to load documents from
url = r'https://www.w3schools.com/gen_ai/gen_ai_intro.asp'
loader = WebBaseLoader(url)

# Load documents (texts) from the web page(s)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(docs[0].page_content)

chain= prompt | llm | parser
result= chain.invoke({'question':'define gen ai?','text':docs[0].page_content})
print(result)
