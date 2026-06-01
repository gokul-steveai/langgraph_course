from typing import Any, Dict

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.state import GraphState


async def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("-------- GRADE DOCUMENTS -------")

    documents = state["documents"]
    question = state["question"]

    filtered_documents = []
    web_search = False

    for doc in documents:
        res: GradeDocuments = await retrieval_grader.ainvoke(
            {"document": doc.page_content, "question": state["question"]}
        )

        score = res.binary_score
        if score == "yes":
            print("------ GRADE: DOCUMENT RELEVANT ------")
            filtered_documents.append(doc)
        else:
            print("------ GRADE: DOCUMENT NOT RELEVANT ------")
            web_search = True
            continue

    return {
        "documents": filtered_documents,
        "question": question,
        "web_search": web_search,
    }
