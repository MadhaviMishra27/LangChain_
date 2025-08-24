# prompt(topic)--> llm --> report --> llm --> 5 importnant key pointer(summary)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load local model and tokenizer
local_dir = "./models/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir)

device = 0 if torch.cuda.is_available() else -1

# Create Hugging Face pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=200,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    return_full_text=False
)

# Wrap pipeline as LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Define output parser
parser = StrOutputParser()

# Define prompt 1: detailed report
prompt1 = PromptTemplate(
    template="Write a detailed report about {topic}.",
    input_variables=["topic"],
)

# Define prompt 2: summary generation
prompt2 = PromptTemplate(
    template="Summarize the following report concisely in 5 sentences:\n{report}",
    input_variables=["report"],
)

# Compose the chain: topic -> prompt1 -> llm -> parser -> prompt2 -> llm -> parser
chain = prompt1 | llm | parser | prompt2 | llm | parser

# Run the chained prompts
topic_input = "Artificial Intelligence"

summary = chain.invoke({"topic": topic_input})

print("Final Summary Output:\n", summary)
chain.get_graph().print_ascii()