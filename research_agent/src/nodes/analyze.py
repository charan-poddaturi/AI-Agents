import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.state import AgentState
import json


def analyze_node(state: AgentState):
    """Evaluator node assessing research sufficiency against query requirements."""
    query = state["query"]
    messages = state["messages"]
    collected_data = list(state.get("collected_data", []))
    iteration_count = state.get("iteration_count", 0)

    # Extract any recent tool outputs or messages to add to collected_data
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_content = f"Tool ({msg.name}) Output: {msg.content}"
            if tool_content not in collected_data:
                collected_data.append(tool_content)

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    evaluator_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0
    )

    context_summary = "\n".join(collected_data) if collected_data else "No data collected yet."

    eval_prompt = f"""You are a strict research evaluation judge.
User Query: {query}

Gathered Research Data & Context:
{context_summary}

Current Iteration: {iteration_count} of Max {state.get('max_iterations', 3)}

Evaluate whether the gathered information is sufficient and complete to fully answer the user query.
Respond in strict JSON format with two fields:
1. "is_sufficient": boolean (true if information is comprehensive enough to answer the query, false otherwise)
2. "analysis_notes": string (detailed explanation of what is known, what is missing, and why sufficiency is true or false)

Return ONLY valid JSON. No markdown code blocks around it, just raw JSON.
"""

    try:
        response = evaluator_model.invoke([HumanMessage(content=eval_prompt)])
        content = response.content.strip()
        # Clean markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        parsed = json.loads(content)
        is_sufficient = bool(parsed.get("is_sufficient", False))
        analysis_notes = str(parsed.get("analysis_notes", "Evaluated progress."))
    except Exception as e:
        # Fallback heuristic if JSON parsing fails
        is_sufficient = iteration_count >= state.get("max_iterations", 3)
        analysis_notes = f"Analysis evaluation defaulted due to parsing error: {e}. Collected items count: {len(collected_data)}"

    return {
        "collected_data": collected_data,
        "is_sufficient": is_sufficient,
        "analysis_notes": analysis_notes,
    }
