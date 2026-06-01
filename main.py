from asyncio import run

from dotenv import load_dotenv

load_dotenv()
from graph.graph import app

if __name__ == "__main__":
    print("--- Starting application ---")
    response = run(app.ainvoke(input={"question": "What is Agent memory?"}))
    print(response)
