from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


async def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves documents from the vector store.

    Args:
        state (GraphState): The state of the graph.

    Returns:
        Dict[str, Any]: A dictionary containing the retrieved documents and the question.
    """
    print("-------- RETRIEVE -------")

    question = state["question"]

    documents = await retriever.ainvoke(question)

    return {"documents": documents, "question": question}
