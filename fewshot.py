from langchain.llms.google_palm import GooglePalm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from dotenv import load_dotenv

import os
load_dotenv()

llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"))

#create our examples
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]

#create an example template
example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)
response = chain.run("What is the meaning of life?")
print(response)