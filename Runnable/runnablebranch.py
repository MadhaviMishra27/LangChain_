from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableBranch, RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM 
from dotenv import load_dotenv

# Load environment variables (for API tokens etc)
load_dotenv()

# Load tokenizer and model (you can switch to any Hugging Face model here)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM .from_pretrained(model_name)

# Setup Hugging Face text-generation pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    do_sample=True,
    temperature=0.5,
    device=-1  # use GPU device=0 if available
)

# Wrap pipeline in LangChain LLM interface
llm = HuggingFacePipeline(pipeline=pipe)

# Define prompts
prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n{text}",
    input_variables=['text']
)

# Output parser to clean response text
parser = StrOutputParser()

# Create runnable chain to generate detailed report
report_gen_chain = RunnableSequence(prompt1, llm, parser)
report_gen_chain = prompt1 | llm | parser
# Branching logic: if report exceeds 500 words, summarize it, else pass through
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, llm, parser)),
    RunnablePassthrough()
)

# Final runnable chain: generate report, then run branch chain (summarize or passthrough)
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Run chain with your topic, e.g., "Russia vs Ukraine"
result = final_chain.invoke({'topic': 'Russia vs Ukraine'})

print(result)
