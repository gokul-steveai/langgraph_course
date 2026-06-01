from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves documents from the vector store.

    Args:
        state (GraphState): The state of the graph.

    Returns:
        Dict[str, Any]: A dictionary containing the retrieved documents and the question.
    """
    print("-------- RETRIEVE -------")

    question = state["question"]

    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}
