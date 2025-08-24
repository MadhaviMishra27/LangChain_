# take text form user--> generate notes and quiz--> give combined output to user
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
#from langchain.chains.parallel import Parallel
from langchain.schema.runnable import RunnableParallel

# Load local model and tokenizer for model1 (notes and merge)
local_dir1 = "./models/bloomz-560m"
tokenizer1 = AutoTokenizer.from_pretrained(local_dir1)
model1 = AutoModelForCausalLM.from_pretrained(local_dir1)
device = 0 if torch.cuda.is_available() else -1

pipe1 = pipeline(
    "text-generation",
    model=model1,
    tokenizer=tokenizer1,
    device=device,
    max_new_tokens=200,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    return_full_text=False,
)
model1 = HuggingFacePipeline(pipeline=pipe1)

# Load local model and tokenizer for model2 (quiz)
local_dir2 = "./models/bloomz-560m"  # can reuse same or different model
tokenizer2 = AutoTokenizer.from_pretrained(local_dir2)
model2 = AutoModelForCausalLM.from_pretrained(local_dir2)

pipe2 = pipeline(
    "text-generation",
    model=model2,
    tokenizer=tokenizer2,
    device=device,
    max_new_tokens=150,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    return_full_text=False,
)
model2 = HuggingFacePipeline(pipeline=pipe2)

# Define parser
parser = StrOutputParser()

# Define prompts
prompt1 = PromptTemplate(
    template="Generate simple and short notes from the following text:\n{text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question-answer pairs from the following text:\n{text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template=(
        "Combine the following notes and quiz into a neat document:\n\n"
        "Notes:\n{notes}\n\n"
        "Quiz:\n{quiz}\n"
    ),
    input_variables=["notes", "quiz"],
)

# Define chains in parallel and merge
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser,
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain
text=''' Generative AI is a branch of artificial intelligence that focuses on creating new and original content, such as text, images, videos, music, and even computer code. Unlike traditional AI systems that only classify or predict based on existing data, generative AI learns the underlying patterns of data and then produces outputs that look similar but are not copied.

The most common type of generative AI is based on large language models (LLMs), like GPT or LLaMA, which are trained on billions of words. These models can generate essays, summaries, stories, and even engage in human-like conversations. For images, models such as DALL·E and Stable Diffusion can create artwork, designs, and realistic photos just from text descriptions.

Generative AI works through techniques such as deep learning and transformer architectures. It takes an input prompt (a few words, an image, or some code) and predicts what should come next, step by step, until a full response is generated.

Applications of generative AI include:

Content creation: writing blogs, reports, or advertisements.

Art and design: producing illustrations, logos, and visual effects.

Education: creating study materials, quizzes, or personalized learning.

Healthcare: assisting in drug discovery and medical image analysis.

Business: generating product descriptions, chatbots, and customer support.

Despite its benefits, generative AI also brings challenges. It can produce inaccurate or misleading information (hallucinations), raise concerns about copyright and ownership, and sometimes reflect biases present in the training data. Additionally, training and running large AI models requires high computing power and energy.

In summary, generative AI is transforming industries by moving AI beyond analysis into creativity. It combines the ability to learn from massive datasets with the power to generate new, useful, and sometimes surprising results.'''

result=chain.invoke({'text':text})
print(result)

