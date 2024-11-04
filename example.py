from config import *

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model=GPT_MODEL, temperature=0.0)

result = model.invoke("What is 2 🦜 9?")

print(result.content)