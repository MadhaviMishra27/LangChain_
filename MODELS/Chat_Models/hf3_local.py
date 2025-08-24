from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
import torch
import os

# Define local model directory
local_dir = "./models/bloomz-560m"

# Step 1: Download and save model locally (only once)
if not os.path.exists(local_dir):
    print("Downloading model...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
else:
    print("Loading model from local directory...")

# Step 2: Load from local directory
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir)

# Step 3: Set up HuggingFace pipeline
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=75,
    temperature=0.5
)

# Step 4: Wrap it in LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Step 5: Use the model
prompt = "Answer the question correctly. Question: Who was the first president of India? Answer:"

response = llm.invoke(prompt)
print("Response:", response.strip())
