from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflection_chain


class MessageGraph(TypedDict):
    """
    A graph of messages.
    """

    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph) -> MessageGraph:
    """
    Generate a new node in the graph.
    """
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph) -> MessageGraph:
    """
    Reflect on a node in the graph.
    """
    response = reflection_chain.invoke({"messages": state["messages"]})

    return {"messages": [HumanMessage(content=response.content)]}


def should_continue(state: MessageGraph) -> str:
    """
    Determine if the graph should continue.

    Returns:
        str: The next node in the graph.
    """
    if len(state["messages"]) > 4:
        return END

    return REFLECT


# Define the graph
builder = StateGraph(
    state_schema=MessageGraph,
)

builder.add_node(
    GENERATE,
    generation_node,
)


builder.add_node(
    REFLECT,
    reflection_node,
)

builder.set_entry_point(GENERATE)


builder.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)

builder.add_edge(
    REFLECT,
    GENERATE,
)

graph = builder.compile()

# graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

# graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {"messages": [HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)]}
    response = graph.invoke(inputs)
    print(response["messages"][-1].content)
