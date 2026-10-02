from typing import Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


# -------------------------
# 1. Create a tool
# -------------------------

@tool
def multiply(
    a: Annotated[int, "First number"],
    b: Annotated[int, "Second number"]
) -> int:
    """Multiply two numbers."""
    return a * b


tools = [multiply]


# -------------------------
# 2. Create the LLM
# -------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


# -------------------------
# 3. Define the LLM node
# -------------------------

def call_llm(state: MessagesState):

    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


# -------------------------
# 4. Build the graph
# -------------------------

graph = StateGraph(MessagesState)

graph.add_node("llm", call_llm)

graph.add_node(
    "tools",
    ToolNode(tools)
)


# -------------------------
# 5. Connect the graph
# -------------------------

graph.add_edge(START, "llm")

graph.add_conditional_edges(
    "llm",
    tools_condition
)

graph.add_edge("tools", "llm")

app = graph.compile()


# -------------------------
# 6. Run the agent
# -------------------------

result = app.invoke({
    "messages": [
        SystemMessage(
            content="You are a helpful assistant. "
                    "Use the calculator tool whenever arithmetic is required."
        ),
        ("user", "What is 347 multiplied by 829?")
    ]
})


print(result["messages"][-1].content)