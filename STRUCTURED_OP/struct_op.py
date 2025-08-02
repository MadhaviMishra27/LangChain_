import json
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_TOKEN")

# Set up the LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    model_kwargs={"api_key": api_key},
    temperature=0.4,
    max_new_tokens=256
)
model = ChatHuggingFace(llm=llm)

# Define the prompt explicitly
prompt = """
Given the following product review, return a valid JSON with two fields:

1. "summary": a concise 1-2 sentence summary of the review
2. "sentiment": either "positive", "negative", or "neutral"

Respond ONLY with valid JSON. No extra explanation.

Review:
"I bought this vacuum cleaner last month and it's honestly a game-changer.
The suction power is incredible, it's lightweight, and the battery lasts surprisingly long.
However, the dustbin is a bit small and needs frequent emptying."
"""

# Get raw response from model
response = model.invoke(prompt)
print("🔍 Raw output:\n", response.content)

# Try to parse JSON manually
try:
    cleaned_output = response.content.strip()

    # If output is surrounded by text, extract JSON substring
    start = cleaned_output.find("{")
    end = cleaned_output.rfind("}") + 1
    json_str = cleaned_output[start:end]

    parsed = json.loads(json_str)
    print("\n✅ Parsed JSON:")
    print("Summary:", parsed.get("summary"))
    print("Sentiment:", parsed.get("sentiment"))
except Exception as e:
    print("\n❌ Failed to parse JSON:", e)
