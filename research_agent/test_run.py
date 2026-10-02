import os
from dotenv import load_dotenv
from src.graph import app

load_dotenv()


def test_research_graph():
    query = "What is LangGraph and how does it compare to LangChain agents? Also calculate 15 * 25."
    print(f"Running test verification with query: '{query}'")

    initial_state = {
        "query": query,
        "messages": [],
        "collected_data": [],
        "iteration_count": 0,
        "max_iterations": 3,
        "is_sufficient": False,
        "analysis_notes": "",
        "final_report": ""
    }

    result = app.invoke(initial_state)

    assert "final_report" in result, "Result must contain final_report"
    report = result["final_report"]
    print("\n--- Test Run Successful ---")
    print(f"Report Length: {len(report)} characters")
    print(f"Iteration Count: {result.get('iteration_count')}")
    print(f"Collected Data Pieces: {len(result.get('collected_data', []))}")
    print("\nSample Report Preview:")
    print(report[:500] + "...\n")


if __name__ == "__main__":
    test_research_graph()
