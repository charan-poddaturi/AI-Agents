from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema for the iterative research agent."""
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    collected_data: List[str]
    iteration_count: int
    max_iterations: int
    is_sufficient: bool
    analysis_notes: str
    final_report: str
