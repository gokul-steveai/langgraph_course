from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run queries on Tavily"""
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool(
            name=AnswerQuestion.__name__,
            description="Answer the question",
            schema=AnswerQuestion,
            func=run_queries,
        ),
        StructuredTool(
            name=ReviseAnswer.__name__,
            description="Revise your original answer to your question.",
            schema=ReviseAnswer,
            func=run_queries,
        ),
    ]
)
