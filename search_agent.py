from langchain.llms.google_palm import GooglePalm
from langchain.agents import initialize_agent, Tool
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from dotenv import load_dotenv

import os
google_cse_id = os.getenv("GOOGLE_CSE_ID")
load_dotenv()

llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"))

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    )
]

agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True,
                         max_iterations=6)

response = agent("What's the latest news about the Mars rover?")
print(response['output'])