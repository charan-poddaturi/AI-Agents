# Modular LangGraph Research Agent

An iterative, stateful LangGraph Research Agent built with Python, LangGraph, LangChain, and Google Gemini (`gemini-3.5-flash-lite`).

## Architecture & Core Loop

1. **User Input -> Agent Node**: The agent analyzes the user's research query, plans research steps, and invokes tools.
2. **Tool Execution**: Supports three core tools:
   - **Web Search**: DuckDuckGo search for up-to-date web intelligence (free, no API key required).
   - **Safe Calculator**: AST-based secure mathematical expression evaluator.
   - **RAG Stub**: Modular interface ready for internal vector database integration.
3. **Analyze Node**: Evaluates gathered information against query requirements, determining if the information is sufficient (`is_sufficient`) and updating analysis notes.
4. **Conditional Routing**:
   - If information is insufficient and `iteration_count < max_iterations` (max 3), routes back to search again.
   - If information is sufficient or max_iterations is reached, routes to Final Report generation.
5. **Reporter Node**: Synthesizes all collected research data into a structured Markdown research report.

## Project Structure

```
research_agent/
├── src/
│   ├── nodes/
│   │   ├── agent.py       # Agent planning & tool invocation node
│   │   ├── analyze.py     # Sufficiency evaluation node
│   │   └── reporter.py    # Markdown report synthesis node
│   ├── tools/
│   │   ├── search.py      # DuckDuckGo search tool
│   │   ├── calculator.py  # Safe AST math calculator tool
│   │   └── rag_stub.py    # RAG knowledge base stub
│   ├── graph.py           # StateGraph definition & conditional edges
│   └── state.py           # AgentState TypedDict schema
├── .env.example           # Environment variables template
├── requirements.txt       # Python dependencies
├── main.py                # CLI runner with streaming
├── test_run.py            # End-to-end verification script
└── README.md              # Project documentation
```

## Setup & Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /mnt/c/Users/chara/OneDrive/Desktop/code/langgraph/research_agent
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and set your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GOOGLE_API_KEY=your_actual_google_api_key_here
   GEMINI_MODEL=gemini-3.5-flash-lite
   ```

## Usage

### Run via CLI
```bash
python main.py "Explain quantum computing and calculate 2^10"
```

### Run Verification Test
```bash
python test_run.py
```
