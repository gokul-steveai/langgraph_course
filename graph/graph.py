from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import AnswerGrader, answer_grader
from graph.chains.hallucination_grader import (
    HallucinationGrader,
    hallucination_grader,
)
from graph.chains.router import RouterQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


async def route_question(state: GraphState) -> str:
    print("--- ROUTE QUESTION ----")
    question = state["question"]

    res: RouterQuery = await question_router.ainvoke({"question": question})

    if res.datasource == "vector_store":
        print("--- ROUTE QUESTION TO VECTOR SEARCH ---")
        return RETRIEVE
    else:
        print("--- ROUTE QUESTION TO WEB SEARCH ---")
        return WEB_SEARCH


def decide_to_generate(state: GraphState) -> str:
    """Decide whether to generate a response to a question."""
    if state["web_search"]:
        print(
            "--- DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION ----"
        )
        return WEB_SEARCH

    else:
        print("---- DECISION: GENERATE ----")
        return GENERATE


async def grade_generation_grounded_in_documents_and_question(
    state: GraphState,
) -> str:
    print("---- CHECK HALLUCINATION ----")

    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]

    score: HallucinationGrader = await hallucination_grader.ainvoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("--- DECISION: GENERATION IS GROUNDED IN DOCUMENTS ---")
        print("--- GRADE GENERATION vs QUESTION ---")
        score: AnswerGrader = await answer_grader.ainvoke(
            {"question": question, "generation": generation}
        )

        if answer_grade := score.binary_score == "yes":
            print("--- DECISION: GENERATION ADDRESSES QUESTION ---")
            return "useful"
        else:
            print("--- DECISION: GENERATION DOES NOT ADDRESS QUESTION ---")
            return "not useful"
    else:
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE: RETRIEVE,
        WEB_SEARCH: WEB_SEARCH,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": WEB_SEARCH,
        "not supported": GENERATE,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE},
)

workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
