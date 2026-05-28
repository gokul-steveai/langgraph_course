from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(..., example="Critique of what is missing")
    superfluous: str = Field(..., example="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question"""

    answer: str = Field(..., example="~250 words detailed answer to the question")
    reflection: Reflection = Field(..., example="Your reflection on the initial answer")
    search_queries: List[str] = Field(
        ...,
        example=["Search query 1", "Search query 2"],
        description="1-3 search queries for researching improvements to address the critique of your answer",
    )

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )