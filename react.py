import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """
    Triple a number.

    Args:
        num (float): The raw float or integer number to triple. Do NOT wrap this in string quotes.

    Returns:
        float: The triple of the input number.
    """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]


llm = ChatGroq(model=os.environ["MODEL_NAME"], temperature=0).bind_tools(tools)


def main():
    user_prompt = "What is the triple value of 3?"
    response = llm.invoke(user_prompt)

    print(response)


if __name__ == "__main__":
    main()
