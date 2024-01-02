from langchain.chat_models.google_palm import ChatGooglePalm
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from dotenv import load_dotenv

import os
load_dotenv()

chat = ChatGooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

system_template = "You are an assistant that helps user to find information about movies without giving any spoiler and create exciting about movie."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message = "Find the information about the movie {movie_title}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

movie = input("Movie title: ")

response = chat(chat_prompt.format_prompt(movie_title=movie).to_messages())

print(response.content)