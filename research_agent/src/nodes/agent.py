import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.state import AgentState
from src.tools.search import web_search
from src.tools.calculator import safe_calculator
from src.tools.rag_stub import rag_search

SYSTEM_INSTRUCTION = (
    "You are an expert autonomous Research Agent. Your goal is to thoroughly research the user's query "
    "by using available tools (web_search, safe_calculator, rag_search). "
    "Plan your research steps carefully. If you need information from the web, internal docs, or calculations, "
    "invoke the appropriate tool. Be precise and thorough."
)

def get_model():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Google API key not found. Ensure GOOGLE_API_KEY is set in your .env file.")
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        # Pass system instruction directly so Gemini handles role ordering cleanly
        system_instruction=SYSTEM_INSTRUCTION,
    )

tools = [web_search, safe_calculator, rag_search]
model_with_tools = get_model().bind_tools(tools)


def agent_node(state: AgentState):
    """Agent node that plans and invokes research tools."""
    query = state.get("query", "")
    messages = list(state.get("messages", []))
    iteration_count = state.get("iteration_count", 0)

    # Initialize messages list with user query if empty
    if not messages:
        messages = [HumanMessage(content=query)]

    # If the last message is already an AIMessage without a tool response, do not re-invoke directly
    response = model_with_tools.invoke(messages)

    collected_data = list(state.get("collected_data", []))
    if response.content and isinstance(response.content, str) and response.content.strip():
        collected_data.append(f"Iteration {iteration_count} Agent Thought/Output: {response.content}")

    return {
        "messages": [response],
        "collected_data": collected_data,
        "iteration_count": iteration_count + 1
    }