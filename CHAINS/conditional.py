from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import torch
from langchain.schema.runnable import RunnableBranch, RunnableLambda
#runnable lambda changes a lambda fun to runnable , so what we can use it as a chain 
pipe = pipeline(
    "text-generation",
    model="bigscience/bloomz-560m",
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=50,
    do_sample=False,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=pipe)
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following product review as Positive or Negative.\n"
        "Review: \"{review}\"\n"
        "Sentiment:"
    ),
    input_variable=["review"],
)
chain = prompt1 | llm | parser
prompt2=PromptTemplate(
    template='write an appropriate response to this positive feedback \n {review}',
    input_variable=['review']
)
prompt3=PromptTemplate(
    template='write an appropriate response to this ngative feedback \n {review}',
    input_variable=['review']
)

branch_chain = RunnableBranch(
    (lambda x: "positive" in x.lower(), prompt2 | llm | parser),
    (lambda x: "negative" in x.lower(), prompt3 | llm | parser),
    RunnableLambda(lambda x: "could not find a sentiment")
)


'''branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2 | llm | parser),    #  condition and corresponding chain
    (lambda x:x.sentiment=='negative', prompt3 | llm | parser),
    RunnableLambda(lambda x: "could not find a sentiment")
)'''
final_chain= chain | branch_chain
print(final_chain.invoke({'review':'this phone is amazing'}))
final_chain.get_graph().print_ascii()


'''branch_chain=RunnableBranch(
    (con1,chain),    #  condition and corresponding chain
    (con2,chain),
    (default chain)
)
# Direct invocation example
review_text = "This product is amazing and works great!"
result = chain.invoke({"review": review_text}).strip()

print(f"Sentiment output:\n{result}") '''
