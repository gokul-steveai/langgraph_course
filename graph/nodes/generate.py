from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


async def generate(state: GraphState) -> Dict[str, Any]:
    """Generate a response to a question."""
    print("---- GENERATE ----")
    question = state["question"]
    documents = state["documents"]

    generation = await generation_chain.ainvoke(
        {"context": documents, "question": question}
    )

    return {
        "documents": documents,
        "generation": generation,
        "question": question,
    }
