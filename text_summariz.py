from langchain.llms.google_palm import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

import os
load_dotenv()

def text_summarize():
    llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"))

    summarization_template = "Summarize the following text: {texted}"

    prompt = PromptTemplate(input_variables=["texted"], template=summarization_template)

    chain = LLMChain(llm=llm, prompt=prompt)

    texted = input("Enter your text: ")

    summarized_text = chain.run(texted)
    print("\n")
    output = print(summarized_text)
    return output

text_summarize()