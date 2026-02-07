"""
SeeSeaAgent - Agent Factory Graph
Based on planning.md architecture
"""
import os
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# ============================================================================
# State Definitions
# ============================================================================

class AgentState(TypedDict, total=False):
    """Main state for the agent factory pipeline"""
    # User input
    user_idea: str

    # P1.1 Spec outputs
    spec: str
    goals: list
    deliverables: list
    interaction_contract: str
    success_metrics: list

    # P1.2 Design outputs
    agent_architecture: dict
    agent_implementation: str
    runbook: str

    # P1.3 Verify outputs
    test_suite: str
    test_results: dict
    known_limitations: list

    # P1.4 Ship outputs
    deployment_status: str
    deployment_report: str

    # P2 Evaluation outputs
    evaluation_cases: list
    behavior_logs: list
    evaluation_report: str
    failure_categories: list
    patch_list: list
    regression_cases: list

    # Loop control
    iteration_count: int
    should_continue: bool


# ============================================================================
# P1 - Build Pipeline Nodes
# ============================================================================

def p1_1_spec(state: AgentState) -> AgentState:
    """
    P1.1 Spec: Transform user idea into detailed specification
    Steps: 1-6 from planning.md
    """
    print("📝 P1.1 Spec - Creating specification...")

    user_idea = state.get("user_idea", "")

    prompt = f"""You are a product specification expert. Given a user idea, create a detailed specification.

User Idea: {user_idea}

Please provide:
1. Self-contained check (data/permission/dependency/boundary)
2. Define goals
3. Define deliverables
4. Define interaction contract
5. Define success metrics

Gate criteria:
- Success is objectively decidable
- Missing assumptions are explicitly stated

Output your response in JSON format with keys: spec, goals, deliverables, interaction_contract, success_metrics"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse LLM response (simplified - you may want to use structured output)
    state["spec"] = response.content
    state["goals"] = ["Parse from LLM response"]
    state["deliverables"] = ["Parse from LLM response"]
    state["interaction_contract"] = "Parse from LLM response"
    state["success_metrics"] = ["Parse from LLM response"]

    return state


def p1_2_design(state: AgentState) -> AgentState:
    """
    P1.2 Design: Define agent architecture and generate implementation
    Steps: 7-8 from planning.md
    """
    print("🏗️  P1.2 Design - Designing agent architecture...")

    spec = state.get("spec", "")
    goals = state.get("goals", [])

    prompt = f"""You are an agent architecture expert. Based on the specification, design the agent architecture.

Specification: {spec}
Goals: {goals}

Please define:
1. Agent architecture (tools, memory, state, policies/constraints)
2. Generate agent implementation outline
3. Create runbook/configuration

Gate criteria:
- Failure strategy exists (retry/fallback/stop)
- Observability exists (log/metric/trace)

Output in JSON format with keys: agent_architecture, agent_implementation, runbook"""

    response = llm.invoke([HumanMessage(content=prompt)])

    state["agent_architecture"] = {"llm_designed": True}
    state["agent_implementation"] = response.content
    state["runbook"] = "Parse from LLM response"

    return state


def p1_3_verify(state: AgentState) -> AgentState:
    """
    P1.3 Verify: Write and execute tests
    Steps: 9-10 from planning.md
    """
    print("✅ P1.3 Verify - Running tests...")

    agent_implementation = state.get("agent_implementation", "")
    success_metrics = state.get("success_metrics", [])

    prompt = f"""You are a test engineer. Based on the agent implementation and success metrics, create tests.

Agent Implementation: {agent_implementation}
Success Metrics: {success_metrics}

Please:
1. Write tests (unit/scenario/regression)
2. Define test execution plan
3. Identify known limitations

Gate criteria:
- Critical paths pass
- Minimal regression set established

Output in JSON format with keys: test_suite, test_results, known_limitations"""

    response = llm.invoke([HumanMessage(content=prompt)])

    state["test_suite"] = response.content
    state["test_results"] = {"passed": "LLM simulated", "failed": 0}
    state["known_limitations"] = ["Parse from LLM response"]

    return state


def p1_4_ship(state: AgentState) -> AgentState:
    """
    P1.4 Ship: Deploy agent
    Steps: 11-12 from planning.md
    """
    print("🚀 P1.4 Ship - Deploying agent...")

    agent_implementation = state.get("agent_implementation", "")
    test_results = state.get("test_results", {})

    prompt = f"""You are a deployment engineer. Plan the deployment strategy.

Agent Implementation: {agent_implementation}
Test Results: {test_results}

Please:
1. Define deployment strategy (local/staging/prod-like)
2. Create canary/smoke test plan
3. Generate deployment report

Output in JSON format with keys: deployment_status, deployment_report"""

    response = llm.invoke([HumanMessage(content=prompt)])

    state["deployment_status"] = "deployed"
    state["deployment_report"] = response.content

    return state


# ============================================================================
# P2 - Evaluation Pipeline Nodes
# ============================================================================

def p2_1_case_preparation(state: AgentState) -> AgentState:
    """
    P2.1 Case Preparation: Prepare evaluation cases
    """
    print("📋 P2.1 Case Preparation - Preparing evaluation cases...")

    spec = state.get("spec", "")
    interaction_contract = state.get("interaction_contract", "")

    prompt = f"""You are a QA engineer. Prepare comprehensive evaluation cases.

Specification: {spec}
Interaction Contract: {interaction_contract}

Please prepare:
1. Normal cases
2. Boundary cases
3. Adversarial/counter-intuitive cases

Output in JSON format as a list of test cases with type and description."""

    response = llm.invoke([HumanMessage(content=prompt)])

    state["evaluation_cases"] = [
        {"type": "normal", "case": "from LLM"},
        {"type": "boundary", "case": "from LLM"},
        {"type": "adversarial", "case": "from LLM"}
    ]

    return state


def p2_2_interaction(state: AgentState) -> AgentState:
    """
    P2.2 Interaction: Interact with agent and collect logs
    """
    print("🔄 P2.2 Interaction - Running agent interactions...")

    evaluation_cases = state.get("evaluation_cases", [])
    agent_implementation = state.get("agent_implementation", "")

    prompt = f"""You are simulating agent interactions. Run the agent against evaluation cases.

Agent Implementation: {agent_implementation}
Evaluation Cases: {evaluation_cases}

Please:
1. Simulate interactions per interaction contract
2. Collect outputs and behavior logs

Output in JSON format with keys: behavior_logs (list of log entries)"""

    response = llm.invoke([HumanMessage(content=prompt)])

    state["behavior_logs"] = ["Simulated log from LLM", "Another log entry"]

    return state


def p2_3_evaluation(state: AgentState) -> AgentState:
    """
    P2.3 Evaluation: Evaluate against success metrics
    """
    print("📊 P2.3 Evaluation - Evaluating performance...")

    behavior_logs = state.get("behavior_logs", [])
    success_metrics = state.get("success_metrics", [])

    prompt = f"""You are an evaluation expert. Evaluate agent performance against success metrics.

Behavior Logs: {behavior_logs}
Success Metrics: {success_metrics}

Please:
1. Evaluate against success metrics
2. Perform failure attribution (failure taxonomy)

Output in JSON format with keys: evaluation_report, failure_categories"""

    response = llm.invoke([HumanMessage(content=prompt)])

    state["evaluation_report"] = response.content
    state["failure_categories"] = ["Parse from LLM"]

    return state


def p2_4_feedback(state: AgentState) -> AgentState:
    """
    P2.4 Feedback: Generate improvement suggestions
    """
    print("💡 P2.4 Feedback - Generating improvements...")

    evaluation_report = state.get("evaluation_report", "")
    failure_categories = state.get("failure_categories", [])

    prompt = f"""You are an improvement expert. Generate actionable improvement suggestions.

Evaluation Report: {evaluation_report}
Failure Categories: {failure_categories}

Please generate:
1. Improvement suggestions (spec issues, design issues, test gaps, tool/policy issues)
2. Convert failures into new regression tests

Output in JSON format with keys: patch_list, regression_cases"""

    response = llm.invoke([HumanMessage(content=prompt)])

    state["patch_list"] = ["LLM suggested patch 1", "LLM suggested patch 2"]
    state["regression_cases"] = ["New regression from LLM"]
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    return state


# ============================================================================
# Decision Nodes
# ============================================================================

def should_continue_loop(state: AgentState) -> str:
    """
    Determine if we should continue the improvement loop
    Based on: MVP threshold, budget limits, iteration count
    """
    # Placeholder logic - simple iteration limit
    max_iterations = 3

    iteration_count = state.get("iteration_count", 0)
    if iteration_count >= max_iterations:
        print(f"🏁 Stopping: reached {max_iterations} iterations")
        return "end"

    # Check if deployment was successful
    if state.get("deployment_status") == "deployed":
        print("🔁 Continuing to evaluation pipeline...")
        return "evaluate"

    return "end"


# ============================================================================
# Graph Construction
# ============================================================================

def create_agent_factory_graph() -> StateGraph:
    """
    Create the agent factory graph based on planning.md
    """
    # Initialize graph
    graph = StateGraph(AgentState)

    # Add P1 (Build Pipeline) nodes
    graph.add_node("p1_1_spec", p1_1_spec)
    graph.add_node("p1_2_design", p1_2_design)
    graph.add_node("p1_3_verify", p1_3_verify)
    graph.add_node("p1_4_ship", p1_4_ship)

    # Add P2 (Evaluation Pipeline) nodes
    graph.add_node("p2_1_case_preparation", p2_1_case_preparation)
    graph.add_node("p2_2_interaction", p2_2_interaction)
    graph.add_node("p2_3_evaluation", p2_3_evaluation)
    graph.add_node("p2_4_feedback", p2_4_feedback)

    # P1 Build Pipeline flow
    graph.add_edge(START, "p1_1_spec")
    graph.add_edge("p1_1_spec", "p1_2_design")
    graph.add_edge("p1_2_design", "p1_3_verify")
    graph.add_edge("p1_3_verify", "p1_4_ship")

    # Decision point after P1.4 Ship
    graph.add_conditional_edges(
        "p1_4_ship",
        should_continue_loop,
        {
            "evaluate": "p2_1_case_preparation",
            "end": END
        }
    )

    # P2 Evaluation Pipeline flow
    graph.add_edge("p2_1_case_preparation", "p2_2_interaction")
    graph.add_edge("p2_2_interaction", "p2_3_evaluation")
    graph.add_edge("p2_3_evaluation", "p2_4_feedback")

    # Loop back to P1.1 for improvements
    graph.add_conditional_edges(
        "p2_4_feedback",
        should_continue_loop,
        {
            "evaluate": "p1_1_spec",
            "end": END
        }
    )

    return graph


# ============================================================================
# Export compiled graph for langgraph dev
# ============================================================================

# Create and compile the graph for langgraph CLI
graph = create_agent_factory_graph().compile()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main entry point for the agent factory
    """
    print("=" * 60)
    print("SeeSeaAgent - Agent Factory")
    print("=" * 60)

    # Create and compile graph
    graph = create_agent_factory_graph()
    compiled_graph = graph.compile()

    # Initial state
    initial_state: AgentState = {
        "user_idea": "Create an agent that helps users debug Python code",
        "spec": None,
        "goals": None,
        "deliverables": None,
        "interaction_contract": None,
        "success_metrics": None,
        "agent_architecture": None,
        "agent_implementation": None,
        "runbook": None,
        "test_suite": None,
        "test_results": None,
        "known_limitations": None,
        "deployment_status": None,
        "deployment_report": None,
        "evaluation_cases": None,
        "behavior_logs": None,
        "evaluation_report": None,
        "failure_categories": None,
        "patch_list": None,
        "regression_cases": None,
        "iteration_count": 0,
        "should_continue": True
    }

    # Run the graph
    print("\n🚀 Starting agent factory pipeline...\n")
    result = compiled_graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("✅ Pipeline completed!")
    print("=" * 60)
    print(f"\nTotal iterations: {result.get('iteration_count', 0)}")
    print(f"Deployment status: {result.get('deployment_status')}")

    return result


if __name__ == "__main__":
    main()
