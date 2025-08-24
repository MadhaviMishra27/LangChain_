from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough
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
def word_count(text):
    return len(text.split())

# Define the first prompt template
prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Define output parser to process LLM output
parser = StrOutputParser()

joke_gen_chain= RunnableSequence(prompt, llm,parser)
parallel_chain=RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic': 'AI'})

final_result = """{}
word count - {}""".format(result['joke'], result['word_count'])

print(final_result)
