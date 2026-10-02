import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.state import AgentState


def reporter_node(state: AgentState):
    """Synthesizer node generating a comprehensive structured markdown final report."""
    query = state["query"]
    collected_data = state.get("collected_data", [])
    analysis_notes = state.get("analysis_notes", "")
    iteration_count = state.get("iteration_count", 0)

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    reporter_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )

    context_summary = "\n".join(collected_data) if collected_data else "No specific data collected."

    prompt = f"""You are an expert research report writer. Synthesize a professional, comprehensive, and well-structured Markdown research report addressing the user's query based on the collected research data.

User Query: {query}

Research Iterations Completed: {iteration_count}
Analysis Notes: {analysis_notes}

Collected Research Data:
{context_summary}

Structure the final report professionally with:
1. # Executive Summary / Title
2. ## Introduction / Background
3. ## Key Findings & Analysis
4. ## Calculations or Technical Insights (if applicable)
5. ## Conclusion / Recommendations
6. ## Sources & References

Ensure the report is thorough, objective, polished, and directly answers the query.
"""

    response = reporter_model.invoke([HumanMessage(content=prompt)])
    final_report = response.content

    return {
        "final_report": final_report,
        "messages": [response]
    }
