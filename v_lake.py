from langchain.llms.google_palm import GooglePalm
from langchain.embeddings.google_palm import GooglePalmEmbeddings
from langchain.vectorstores.deeplake import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from dotenv import load_dotenv

import os
activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
load_dotenv()



llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GooglePalmEmbeddings()
'''
texts = [
        "Napoleon Bonaparte was born in 15 August 1769",
        "Louis XIV was born in 5 September 1638"
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)
'''

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "angkul58" 
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# loading data and updating a  existing data set
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)
# add documents to our Deep Lake dataset
db.add_documents(docs)

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions"
    ),
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run("When was Louis XIV born?")
print(response)
