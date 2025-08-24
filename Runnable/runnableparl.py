from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
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

# Define your prompt templates
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

# Output parser for extracting plain text responses
parser = StrOutputParser()

# Setup runnable sequences for each prompt-model-parser trio
tweet_sequence = RunnableSequence(prompt1, llm, parser)
linkedin_sequence = RunnableSequence(prompt2, llm, parser)

# Combine both sequences into a parallel runnable
parallel_chain = RunnableParallel({
    'tweet': tweet_sequence,
    'linkedin': linkedin_sequence
})

# Invoke the parallel runnable with input
result = parallel_chain.invoke({'topic': 'AI'})

print(result['tweet'])
print(result['linkedin'])