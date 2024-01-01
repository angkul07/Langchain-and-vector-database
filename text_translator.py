from langchain.llms.google_palm import GooglePalm
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

import os

load_dotenv()
huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")


# def text_translator():
# llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"))
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={'temperature':0.1}
)
translator_template = "You are a translator AI. You are well trained on my different languages and you have the ability to translate many languagues. Translate the following text from {source_language} to {target_language}: {text}"
translator_prompt = PromptTemplate(
    input_variables=["source_language", "target_language", "text"],
    template=translator_template,
)
text = input("Enter the text that you want to translate: \n")
source_language = input("What is the language of text?: \n")
target_language = input("Language in which text to be translated?: \n")
translator_chain = LLMChain(llm=llm, prompt=translator_prompt)
output = translator_chain.predict(
    text=text, source_language=source_language, target_language=target_language
)
print(output)


