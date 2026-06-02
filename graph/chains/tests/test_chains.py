from pprint import pprint

from dotenv import load_dotenv

from graph.chains.generation import (
    generation_chain,
)

load_dotenv()

from graph.chains.hallucination_grader import (
    HallucinationGrader,
    hallucination_grader,
)
from graph.chains.retrieval_grader import (
    GradeDocuments,
    retrieval_grader,
)
from graph.chains.router import RouterQuery, question_router
from ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"

    docs = retriever.invoke(question)

    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {
            "document": doc_txt,
            "question": question,
        }
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"

    docs = retriever.invoke(question)

    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {
            "document": doc_txt,
            "question": "How to make a ice cream?",
        }
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"

    docs = retriever.invoke(question)

    doc_txt = docs[0].page_content

    generation = generation_chain.invoke(
        {
            "context": doc_txt,
            "question": question,
        }
    )

    pprint(generation)


def test_hallucination_grader_yes() -> None:
    question = "Agent memory"

    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {
            "context": docs,
            "question": question,
        }
    )
    res: HallucinationGrader = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": generation,
        }
    )

    assert res.binary_score


def test_hallucination_grader_no() -> None:
    question = "Agent memory"

    docs = retriever.invoke(question)
    res: HallucinationGrader = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "This is a test generation",
        }
    )

    assert not res.binary_score


def test_router_to_vector_store() -> None:
    question = "Agent memory"

    res: RouterQuery = question_router.invoke({"question": question})

    assert res.datasource == "vector_store"


def test_router_to_web_store() -> None:

    question = "How to make a ice cream?"

    res: RouterQuery = question_router.invoke({"question": question})

    assert res.datasource == "web_search"
