import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()


class RouterQuery(BaseModel):
    datasource: Literal["vector_store", "web_search"] = Field(
        ..., description="The datasource to use for the query."
    )


llm = ChatGroq(
    model_name=os.getenv("MODEL_NAME"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
)

structured_llm_router = llm.with_structured_output(RouterQuery)

system = """You are an expert at routing a user question to vector store or web search.
The vector store contains documents relevant to the agents, prompt engineering and adversarial attacks.
Use the vector store for questions on these topics and the web search for other questions.
Output 'vector_store' or 'web_search' to indicate the datasource to use for the query."""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User Question: {question}",
        ),
    ]
)

question_router: RunnableSequence = router_prompt | structured_llm_router
