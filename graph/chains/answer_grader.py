import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_core.runnables import (
    RunnableSequence,
)
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()


class AnswerGrader(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses and resolves the question, output 'yes' or 'no'"
    )


llm = ChatGroq(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

structured_llm_grader = llm.with_structured_output(AnswerGrader)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User Question: {question} \n\n LLM generation: {generation}",
        ),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
