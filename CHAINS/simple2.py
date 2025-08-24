from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

local_dir = "./models/bloomz-560m"

if not os.path.exists(local_dir):
    print("Downloading model...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
else:
    print("Loading model from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(local_dir)

device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=150,         # Increased to allow longer generation
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
    return_full_text=False      # Only new generated text, not prompt
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    template="You are a helpful assistant.generate me 10 lines about {topic}.",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "cricket"})

print("Result:", result)
chain.get_graph().print_ascii()