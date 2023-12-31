from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

import os
load_dotenv()

huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")

hub_llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={'temperature':0}
)

template = """
Question: {question},
Answer:
"""

prompt = PromptTemplate(
    input_variables=['question'],
    template=template
)


llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# question = input("Ask your question: ")
# print(llm_chain.run(question))

qa = [
    {'question': "What is the capital city of France?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]
res = llm_chain.generate(qa)
print( res )

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=hub_llm
)

qs_str = (
    "What is the capital city of France?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
		"What color is a ripe banana?\n"
)
llm_chain.run(qs_str)

