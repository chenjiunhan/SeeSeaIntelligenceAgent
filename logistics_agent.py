"""
Logistics Agent - Query vessel arrivals data and answer questions
"""
import os
from typing import TypedDict, Annotated, Literal
from pathlib import Path
import pandas as pd
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# ============================================================================
# Data Query Tools
# ============================================================================

@tool
def query_vessel_data(
    start_date: str = None,
    end_date: str = None,
    vessel_type: str = None,
    aggregate: str = "none"
) -> str:
    """
    Query vessel arrivals data from the Bab el-Mandeb chokepoint.

    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        vessel_type: Type of vessel to filter: container, dry_bulk, tanker, general_cargo, roro (optional)
        aggregate: Aggregation method: none, daily_avg, monthly_avg, total (default: none)

    Returns:
        String containing the queried data in a readable format
    """
    try:
        # Load data from SeeSeaIntelligence
        project_root = Path(__file__).parent.parent / 'SeeSeaIntelligence'
        data_path = project_root / 'processed' / 'logistics' / 'chokepoints' / 'bab-el-mandeb' / 'vessel_arrivals' / 'vessel_arrivals.csv'

        if not data_path.exists():
            return f"Error: Data file not found at {data_path}"

        df = pd.read_csv(data_path, parse_dates=['date'])
        df.set_index('date', inplace=True)

        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Filter by vessel type
        if vessel_type:
            valid_types = ['container', 'dry_bulk', 'tanker', 'general_cargo', 'roro']
            if vessel_type not in valid_types:
                return f"Error: Invalid vessel_type. Must be one of {valid_types}"
            data = df[vessel_type]
        else:
            data = df

        # Apply aggregation
        if aggregate == "daily_avg":
            result = data.mean()
            return f"Daily average: {result}"
        elif aggregate == "monthly_avg":
            result = data.resample('M').mean()
            return f"Monthly averages:\n{result.to_string()}"
        elif aggregate == "total":
            if vessel_type:
                result = data.sum()
                return f"Total {vessel_type} vessels: {result}"
            else:
                result = df[['container', 'dry_bulk', 'tanker', 'general_cargo', 'roro']].sum()
                return f"Total vessels by type:\n{result.to_string()}"
        else:  # none
            # Return sample of data
            if len(data) > 10:
                return f"Data (showing first 10 rows):\n{data.head(10).to_string()}\n\nTotal rows: {len(data)}"
            else:
                return f"Data:\n{data.to_string()}"

    except Exception as e:
        return f"Error querying data: {str(e)}"


@tool
def get_data_summary() -> str:
    """
    Get a summary of the vessel arrivals dataset including date range,
    total records, and basic statistics.

    Returns:
        String containing dataset summary
    """
    try:
        # Load data from SeeSeaIntelligence
        project_root = Path(__file__).parent.parent / 'SeeSeaIntelligence'
        data_path = project_root / 'processed' / 'logistics' / 'chokepoints' / 'bab-el-mandeb' / 'vessel_arrivals' / 'vessel_arrivals.csv'

        if not data_path.exists():
            return f"Error: Data file not found at {data_path}"

        df = pd.read_csv(data_path, parse_dates=['date'])
        df.set_index('date', inplace=True)

        summary = f"""
Dataset Summary - Bab el-Mandeb Vessel Arrivals
================================================
Total Records: {len(df)}
Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}

Vessel Types Available:
- Container ships
- Dry bulk carriers
- Tankers
- General cargo ships
- RoRo (Roll-on/Roll-off) vessels

Statistics (all data):
{df[['vessel_count', 'container', 'dry_bulk', 'tanker', 'general_cargo', 'roro']].describe().to_string()}
"""
        return summary

    except Exception as e:
        return f"Error getting summary: {str(e)}"


@tool
def compare_periods(
    period1_start: str,
    period1_end: str,
    period2_start: str,
    period2_end: str,
    vessel_type: str = None
) -> str:
    """
    Compare vessel arrivals between two time periods.

    Args:
        period1_start: Start date of first period (YYYY-MM-DD)
        period1_end: End date of first period (YYYY-MM-DD)
        period2_start: Start date of second period (YYYY-MM-DD)
        period2_end: End date of second period (YYYY-MM-DD)
        vessel_type: Type of vessel to compare (optional, compares all if not specified)

    Returns:
        String containing comparison results
    """
    try:
        # Load data from SeeSeaIntelligence
        project_root = Path(__file__).parent.parent / 'SeeSeaIntelligence'
        data_path = project_root / 'processed' / 'logistics' / 'chokepoints' / 'bab-el-mandeb' / 'vessel_arrivals' / 'vessel_arrivals.csv'

        df = pd.read_csv(data_path, parse_dates=['date'])
        df.set_index('date', inplace=True)

        # Get data for both periods
        period1 = df.loc[period1_start:period1_end]
        period2 = df.loc[period2_start:period2_end]

        if vessel_type:
            p1_avg = period1[vessel_type].mean()
            p2_avg = period2[vessel_type].mean()
            diff = p2_avg - p1_avg
            pct_change = (diff / p1_avg * 100) if p1_avg > 0 else 0

            result = f"""
Comparison for {vessel_type}:
Period 1 ({period1_start} to {period1_end}): {p1_avg:.2f} vessels/day
Period 2 ({period2_start} to {period2_end}): {p2_avg:.2f} vessels/day
Change: {diff:+.2f} vessels/day ({pct_change:+.1f}%)
"""
        else:
            p1_total = period1['vessel_count'].mean()
            p2_total = period2['vessel_count'].mean()
            diff = p2_total - p1_total
            pct_change = (diff / p1_total * 100) if p1_total > 0 else 0

            result = f"""
Comparison (Total Vessels):
Period 1 ({period1_start} to {period1_end}): {p1_total:.2f} vessels/day
Period 2 ({period2_start} to {period2_end}): {p2_total:.2f} vessels/day
Change: {diff:+.2f} vessels/day ({pct_change:+.1f}%)

Breakdown by type (Period 1 → Period 2):
"""
            for vtype in ['container', 'dry_bulk', 'tanker', 'general_cargo', 'roro']:
                p1 = period1[vtype].mean()
                p2 = period2[vtype].mean()
                d = p2 - p1
                result += f"  {vtype}: {p1:.1f} → {p2:.1f} ({d:+.1f})\n"

        return result

    except Exception as e:
        return f"Error comparing periods: {str(e)}"


# ============================================================================
# Agent State and Graph
# ============================================================================

# Bind tools to LLM
tools = [query_vessel_data, get_data_summary, compare_periods]
llm_with_tools = llm.bind_tools(tools)


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Determine if we should continue to tools or end"""
    messages = state["messages"]
    last_message = messages[-1]

    # If LLM makes a tool call, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # Otherwise end
    return "__end__"


def call_model(state: MessagesState):
    """Call the model with tools"""
    messages = state["messages"]

    # Check if we have any user messages
    if not messages:
        return {"messages": []}

    # Add system message if first user message
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        system_message = SystemMessage(content="""You are a logistics analyst assistant specialized in vessel traffic data at the Bab el-Mandeb strait chokepoint.

You have access to historical vessel arrivals data from 2019-01-01 to 2026-01-25 (including 2026 data).

IMPORTANT: You MUST use the available tools to query the actual data. DO NOT make assumptions about what data is available - always call the tools first.

Available tools:
- get_data_summary: Get overall dataset summary and statistics
- query_vessel_data: Query specific date ranges and vessel types
- compare_periods: Compare traffic between two time periods

When answering questions:
1. ALWAYS use the tools to fetch actual data from the database
2. Provide clear, data-driven answers based on the tool results
3. Include relevant context (dates, vessel types, trends)
4. Explain what the data means in practical terms

Vessel types available:
- container: Container ships
- dry_bulk: Dry bulk carriers
- tanker: Oil/chemical tankers
- general_cargo: General cargo ships
- roro: Roll-on/Roll-off vessels (vehicle carriers)
""")
        messages = [system_message] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Create the graph
def create_logistics_graph():
    """Create the logistics agent graph"""
    graph = StateGraph(MessagesState)

    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph


# Compile graph for export
graph = create_logistics_graph().compile()


# ============================================================================
# Main (for testing)
# ============================================================================

def main():
    """Test the logistics agent"""
    print("=" * 60)
    print("Logistics Agent - Vessel Arrivals Analysis")
    print("=" * 60)

    compiled_graph = create_logistics_graph().compile()

    # Test queries
    test_queries = [
        "What's the summary of the dataset?",
        "How many container ships passed through in January 2026?",
        "Compare vessel traffic between 2024 and 2025"
    ]

    for query in test_queries:
        print(f"\n\nQ: {query}")
        print("-" * 60)

        result = compiled_graph.invoke({
            "messages": [HumanMessage(content=query)]
        })

        # Print the final response
        final_message = result["messages"][-1]
        print(f"A: {final_message.content}")


if __name__ == "__main__":
    main()
