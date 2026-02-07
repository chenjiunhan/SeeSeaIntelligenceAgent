"""
Logistics Agent - Query vessel arrivals data and answer questions
"""
import os
from typing import TypedDict, Annotated, Literal
from pathlib import Path
from datetime import datetime
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
    chokepoint: str,
    start_date: str = None,
    end_date: str = None,
    vessel_type: str = None,
    aggregate: str = "none"
) -> str:
    """
    Query vessel arrivals data from various maritime chokepoints.

    Args:
        chokepoint: Chokepoint name (bab-el-mandeb, suez-canal, panama-canal, strait-of-hormuz, strait-of-malacca, bosporus-strait)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        vessel_type: Type of vessel to filter: container, dry_bulk, tanker, general_cargo, roro (optional)
        aggregate: Aggregation method: none, daily_avg, monthly_avg, total (default: none)

    Returns:
        String containing the queried data in a readable format
    """
    try:
        # Validate chokepoint
        valid_chokepoints = ['bab-el-mandeb', 'suez-canal', 'panama-canal', 'strait-of-hormuz', 'strait-of-malacca', 'bosporus-strait']
        if chokepoint not in valid_chokepoints:
            return f"Error: Invalid chokepoint. Must be one of {valid_chokepoints}"

        # Load data from SeeSeaIntelligence
        # Check if running in Docker (mounted at /data/processed) or local dev
        if Path('/data/processed').exists():
            data_path = Path(f'/data/processed/logistics/chokepoints/{chokepoint}/vessel_arrivals/vessel_arrivals.csv')
        else:
            project_root = Path(__file__).parent.parent / 'SeeSeaIntelligence'
            data_path = project_root / 'processed' / 'logistics' / 'chokepoints' / chokepoint / 'vessel_arrivals' / 'vessel_arrivals.csv'

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
def get_data_summary(chokepoint: str = None) -> str:
    """
    Get a summary of the vessel arrivals dataset including date range,
    total records, and basic statistics. If no chokepoint specified, lists all available chokepoints.

    Args:
        chokepoint: Chokepoint name (optional). If not specified, lists all available chokepoints.

    Returns:
        String containing dataset summary
    """
    try:
        # If no chokepoint specified, list all available
        if not chokepoint:
            return """Available Chokepoints:
1. bab-el-mandeb - Bab el-Mandeb Strait (Red Sea entrance)
2. suez-canal - Suez Canal (Egypt)
3. panama-canal - Panama Canal (Central America)
4. strait-of-hormuz - Strait of Hormuz (Persian Gulf)
5. strait-of-malacca - Strait of Malacca (Southeast Asia)
6. bosporus-strait - Bosporus Strait (Turkey)

Use get_data_summary with a specific chokepoint name to see details."""

        # Validate chokepoint
        valid_chokepoints = ['bab-el-mandeb', 'suez-canal', 'panama-canal', 'strait-of-hormuz', 'strait-of-malacca', 'bosporus-strait']
        if chokepoint not in valid_chokepoints:
            return f"Error: Invalid chokepoint. Must be one of {valid_chokepoints}"

        # Load data from SeeSeaIntelligence
        # Check if running in Docker (mounted at /data/processed) or local dev
        if Path('/data/processed').exists():
            data_path = Path(f'/data/processed/logistics/chokepoints/{chokepoint}/vessel_arrivals/vessel_arrivals.csv')
        else:
            project_root = Path(__file__).parent.parent / 'SeeSeaIntelligence'
            data_path = project_root / 'processed' / 'logistics' / 'chokepoints' / chokepoint / 'vessel_arrivals' / 'vessel_arrivals.csv'

        if not data_path.exists():
            return f"Error: Data file not found at {data_path}"

        df = pd.read_csv(data_path, parse_dates=['date'])
        df.set_index('date', inplace=True)

        # Format chokepoint name for display
        chokepoint_display = chokepoint.replace('-', ' ').title()

        summary = f"""
Dataset Summary - {chokepoint_display} Vessel Arrivals
================================================
Chokepoint: {chokepoint}
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
    chokepoint: str,
    period1_start: str,
    period1_end: str,
    period2_start: str,
    period2_end: str,
    vessel_type: str = None
) -> str:
    """
    Compare vessel arrivals between two time periods for a specific chokepoint.

    Args:
        chokepoint: Chokepoint name (bab-el-mandeb, suez-canal, panama-canal, strait-of-hormuz, strait-of-malacca, bosporus-strait)
        period1_start: Start date of first period (YYYY-MM-DD)
        period1_end: End date of first period (YYYY-MM-DD)
        period2_start: Start date of second period (YYYY-MM-DD)
        period2_end: End date of second period (YYYY-MM-DD)
        vessel_type: Type of vessel to compare (optional, compares all if not specified)

    Returns:
        String containing comparison results
    """
    try:
        # Validate chokepoint
        valid_chokepoints = ['bab-el-mandeb', 'suez-canal', 'panama-canal', 'strait-of-hormuz', 'strait-of-malacca', 'bosporus-strait']
        if chokepoint not in valid_chokepoints:
            return f"Error: Invalid chokepoint. Must be one of {valid_chokepoints}"

        # Load data from SeeSeaIntelligence
        if Path('/data/processed').exists():
            data_path = Path(f'/data/processed/logistics/chokepoints/{chokepoint}/vessel_arrivals/vessel_arrivals.csv')
        else:
            project_root = Path(__file__).parent.parent / 'SeeSeaIntelligence'
            data_path = project_root / 'processed' / 'logistics' / 'chokepoints' / chokepoint / 'vessel_arrivals' / 'vessel_arrivals.csv'

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
        # Get current date/time for context
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        system_message = SystemMessage(content=f"""You are a logistics analyst assistant specialized in vessel traffic data at major maritime chokepoints worldwide.

CURRENT DATE AND TIME: {current_datetime} (UTC+8 Taipei Time)

You have access to historical vessel arrivals data for multiple chokepoints:
1. Bab el-Mandeb Strait (bab-el-mandeb) - Red Sea entrance
2. Suez Canal (suez-canal) - Egypt
3. Panama Canal (panama-canal) - Central America
4. Strait of Hormuz (strait-of-hormuz) - Persian Gulf
5. Strait of Malacca (strait-of-malacca) - Southeast Asia
6. Bosporus Strait (bosporus-strait) - Turkey

CRITICAL RULES:
1. You MUST IMMEDIATELY use tools to query data - DO NOT ask users what they want to see
2. When users ask about "recent" traffic, automatically query the last 30 days from today's date
3. When users ask general questions like "how is the traffic", provide actual numbers using query_vessel_data
4. NEVER say "I can query" or "Would you like me to check" - JUST DO IT
5. If a query is vague, make reasonable assumptions and query the data

Available tools:
- get_data_summary: Get overall dataset summary and statistics for a chokepoint (or list all if no chokepoint specified)
- query_vessel_data: Query specific date ranges and vessel types for a chokepoint (REQUIRED for all traffic questions)
- compare_periods: Compare traffic between two time periods for a chokepoint

Response Guidelines:
1. For "recent" or "latest" → query last 30 days automatically
2. For general questions → query last 30 days and compare with same period last year
3. ALWAYS provide actual numbers, never just say "I can check"
4. Include trends, comparisons, and insights automatically

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
