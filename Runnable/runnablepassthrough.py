from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables for API keys (if needed)
load_dotenv()

# Load Hugging Face model and tokenizer
model_name = "bigscience/bloomz-560m"  # Choose your model here
#model_name = "t5-small" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Setup Hugging Face pipeline for text-generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    device=-1  # Use 0 if using GPU
)

# Wrap the Hugging Face pipeline in LangChain's LLM wrapper
llm = HuggingFacePipeline(pipeline=pipe) 
'''passthrough= RunnablePassthrough()
print(passthrough.invoke({'name':'madhavi'}))'''




# Define the first prompt template
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Define output parser to process LLM output
parser = StrOutputParser()

# Define the second prompt template for explaining the joke
prompt2 = PromptTemplate(
    template="Explain the following joke in 10 sentences- {text}",
    input_variables=["text"]
)

# Assemble the runnable sequence
chain = RunnableSequence(prompt1,llm,parser,prompt2,llm,parser)

joke_gen_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,llm, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({'topic':'samosa'}))
