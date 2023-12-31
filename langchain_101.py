from langchain.llms.google_palm import GooglePalm
# from langchain.chains import LLMChain # The chains
# from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain  #the memory
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

import os
load_dotenv()

llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"))
'''
prompt = PromptTemplate(
    input_variables=["Product"],
    template="What are five good name for a company that makes {product}?"
)
'''
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
# print(llm(text))
#chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("eco-friendly shoes"))

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation
print(conversation)