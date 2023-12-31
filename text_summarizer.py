from langchain.llms.google_palm import GooglePalm
from langchain.agents.load_tools import Tool
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

import os
google_cse_id = os.getenv("GOOGLE_CSE_ID")
load_dotenv()

llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write the summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
        name="Summarizer",
        func=summarize_chain.run,
        description="useful for summarizing texts"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent("What's the latest news about the langchain framework? Then please summarize the results.")
print(response['output'])