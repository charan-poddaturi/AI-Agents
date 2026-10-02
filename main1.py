import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()


class State(TypedDict):
    message: str


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


def call_llm(state: State):
    response = llm.invoke(state["message"])

    return {
        "message": response.content
    }


graph = StateGraph(State)

graph.add_node("llm", call_llm)

graph.add_edge(START, "llm")
graph.add_edge("llm", END)

app = graph.compile()


result = app.invoke({
    "message": "how can a human be more powerful that artificial intelligence which is an advanced stage ,answer in one line."
})

print(result["message"])