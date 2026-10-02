from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from src.state import AgentState
from src.nodes.agent import agent_node, tools
from src.nodes.analyze import analyze_node
from src.nodes.reporter import reporter_node


def route_agent(state: AgentState):
    """Route from agent to tools if tool calls exist, otherwise to analyze."""
    messages = state["messages"]
    if not messages:
        return "analyze"
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "analyze"


def should_continue(state: AgentState):
    """Conditional routing based on sufficiency and iteration count."""
    is_sufficient = state.get("is_sufficient", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if not is_sufficient and iteration_count < max_iterations:
        return "continue"
    else:
        return "finish"


def create_research_graph():
    """Build and compile the iterative research agent state graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("reporter", reporter_node)

    # Set entry point
    workflow.add_edge(START, "agent")

    # Agent conditional routing (to tools or analyze)
    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "analyze": "analyze"
        }
    )

    # Tool execution loops back to agent
    workflow.add_edge("tools", "agent")

    # Analyze conditional routing (continue research or finish to reporter)
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "continue": "agent",
            "finish": "reporter"
        }
    )

    # Reporter leads to END
    workflow.add_edge("reporter", END)

    return workflow.compile()


# Compiled app instance ready for invocation/streaming
app = create_research_graph()
