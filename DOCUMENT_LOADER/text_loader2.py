from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
loader=TextLoader(r'C:\Users\91739\OneDrive\Desktop\LangChain\DOCUMENT_LOADER\indianEco.txt',encoding='utf-8')
docs=loader.load()
# Choose model name
model_name = "t5-small"  # Can be changed to other Hugging Face text2text models

# Create Hugging Face pipeline for text2text generation
hf_pipe = pipeline("text2text-generation",model=model_name,device=-1)  # Use 0 if you want GPU

# Wrap Hugging Face pipeline with LangChain
llm = HuggingFacePipeline(pipeline=hf_pipe)
prompt=PromptTemplate(
    template='write a summary for following report: \n {text}',
    input_variable=['text']
)
parser = StrOutputParser()
chain=prompt | llm | parser

print(chain.invoke({'text':docs[0].page_content}))