from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model & tokenizer locally
model_id = "bigscience/bloomz-560m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prompt
prompt = "who was the first president of indian? "
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.25
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:", response.replace(prompt, "").strip())
