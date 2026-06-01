import os

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

llm = ChatGroq(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


class HallucinationGrader(BaseModel):
    """Binary score for hallucination present in generated answer"""

    binary_score: bool = Field(
        description="Output True if the answer is completely grounded in the facts, or False if there are hallucinations."
    )


structured_llm_grader = llm.with_structured_output(HallucinationGrader)


system = """You are a grader assessing whether an answer addresses / resolves a question.
Provide a strict boolean score: True if the answer resolves the question, or False if it does not."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Set of facts: \n\n {documents} \n\n generation: {generation}",
        ),
    ]
)

hallucination_grader: RunnableSequence = (
    hallucination_prompt | structured_llm_grader
)
