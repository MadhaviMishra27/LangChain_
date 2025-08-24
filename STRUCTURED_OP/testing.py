from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing_extensions import Annotated
from typing import TypedDict
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_TOKEN")

# Set up the HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    model_kwargs={"api_key": api_key},
    temperature=0.3,
    max_new_tokens=512
)
model = ChatHuggingFace(llm=llm)

# Define TypedDict schema for structured output
class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Sentiment of the review: positive, negative, or neutral"]

# Enable structured output based on the TypedDict
structured_model = model.with_structured_output(Review)

# Clear prompt to guide the model
review_text = """
You are an intelligent assistant. Carefully read the following product review and extract:
1. A short summary.
2. The sentiment (positive, negative, or neutral).

Return JSON ONLY in this format:
{
  "summary": "...",
  "sentiment": "..."
}

Review:
"I bought this vacuum cleaner last month and it's honestly a game-changer.
The suction power is incredible, it's lightweight, and the battery lasts surprisingly long.
However, the dustbin is a bit small and needs frequent emptying."
"""

# Invoke the structured model
result = structured_model.invoke(review_text)

# Handle result
if result:
    print("✅ Summary:", result["summary"])
    print("✅ Sentiment:", result["sentiment"])
else:
    print("❌ Structured output failed. Try using stronger prompt formatting or check model compatibility.")
