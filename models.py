from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


#----------------------Models----------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
)