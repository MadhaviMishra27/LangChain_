from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load environment variables from .env file
load_dotenv()

# Set USER_AGENT to identify your requests and avoid warnings
os.environ['USER_AGENT'] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
)

hf_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not hf_api_token:
    raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN in your environment or .env file")

# Setup Hugging Face text2text generation pipeline (no use_auth_token arg here)
hf_pipe = pipeline(
    "text2text-generation",    # T5 and similar models use this task
    model="google/flan-t5-base",
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    device=-1  # CPU; set to 0 for GPU if available
)

# Wrap pipeline with LangChain
llm = HuggingFacePipeline(pipeline=hf_pipe)

# Define prompt template
prompt = PromptTemplate(
    template='Answer the following question:\n{question}\n\nBased on this text:\n{text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()
chain = prompt | llm | parser

# Define URL to load content from
url = r'https://www.w3schools.com/gen_ai/gen_ai_intro.asp'
loader = WebBaseLoader(url)

# Load documents from URL
docs = loader.load()

# Split large text into manageable chunks for the model
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(docs[0].page_content)

# Process each chunk with the chain and collect answers
question = "Define generative AI."
answers = []
for chunk in chunks:
    response = chain.invoke({
        'question': question,
        'text': chunk
    })
    answers.append(response)

# Combine and print the answers
final_answer = "\n".join(answers)
print(final_answer)
