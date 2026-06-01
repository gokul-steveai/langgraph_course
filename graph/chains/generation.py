import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langsmith.client import Client

load_dotenv()

llm = ChatGroq(
    model_name=os.getenv("MODEL_NAME"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
)

prompt = Client().pull_prompt(
    "rlm/rag-prompt", dangerously_pull_public_prompt=True
)

generation_chain: RunnableSequence = prompt | llm | StrOutputParser()
