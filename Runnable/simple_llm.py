from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import torch

# Load Hugging Face text-generation pipeline with a small model suitable for CPU
pipe = pipeline(
    "text-generation",
    model="bigscience/bloomz-560m",
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)
parser = StrOutputParser()

# Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="tell me two interesting facts about {topic}."
)

# Get input topic from user
topic = input("Enter a topic: ")

# Format the prompt
formatted_prompt = prompt.format(topic=topic)

# Generate blog title using the open-source model
blog_title = llm.predict(formatted_prompt)

# Parse and print the output
blog_title_clean = parser.parse(blog_title)
print("Generated Blog Title:", blog_title_clean)
