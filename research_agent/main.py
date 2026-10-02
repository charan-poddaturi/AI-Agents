import os
import sys
from dotenv import load_dotenv
from src.graph import app

# Load environment variables from .env if present
load_dotenv()


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please copy .env.example to .env and configure your GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)

    print("==================================================")
    print("      Modular LangGraph Research Agent CLI        ")
    print("==================================================")
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        try:
            query = input("Enter your research query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not query:
        print("Query cannot be empty.")
        sys.exit(1)

    print(f"\n[Initiating Research]: {query}\n")

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

    config = {"recursion_limit": 25}

    try:
        for event in app.stream(initial_state, config=config):
            for node_name, state_update in event.items():
                print(f"--- Node Executed: [{node_name.upper()}] ---")
                if node_name == "agent":
                    msgs = state_update.get("messages", [])
                    if msgs:
                        last_msg = msgs[-1]
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                print(f"  -> Planning Tool Call: {tc.get('name')} with args {tc.get('args')}")
                        elif last_msg.content:
                            print(f"  -> Agent Output: {last_msg.content[:200]}...")
                elif node_name == "tools":
                    print("  -> Tool executed successfully.")
                elif node_name == "analyze":
                    print(f"  -> Sufficiency Check: {state_update.get('is_sufficient')}")
                    print(f"  -> Analysis Notes: {state_update.get('analysis_notes')}")
                elif node_name == "reporter":
                    print("  -> Final Report generated.")
                print()

        # Retrieve final state
        final_state = app.get_state(config)
        # Wait, app.get_state(config).values gives state dict in langgraph
        if hasattr(final_state, "values"):
            values = final_state.values
        else:
            values = final_state

        report = values.get("final_report", "No report generated.")
        print("==================================================")
        print("                  FINAL REPORT                    ")
        print("==================================================")
        print(report)
        print("==================================================")

    except Exception as e:
        print(f"\nAn error occurred during graph execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
