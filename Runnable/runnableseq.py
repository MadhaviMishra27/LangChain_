from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load your Hugging Face model and tokenizer
model_name = "bigscience/bloomz-560m"  # or any open-source chat/model you prefer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    device=-1,  # set 0 if you have GPU
)

llm = HuggingFacePipeline(pipeline=pipe)

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

# Run the chain with an input topic
result = chain.invoke({"topic": "summer"})

print(result)
