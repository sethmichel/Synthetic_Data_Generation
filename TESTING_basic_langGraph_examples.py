from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# 1. Define the State (The shared memory)
class PipelineState(TypedDict):
    volatility_constraint: float
    grounding_seeds: List[dict]
    generated_row: dict
    judge_critique: str
    retry_count: int

# 2. Define the Nodes (Your standard Python functions)
def sampler_node(state: PipelineState):
    # Call NeMo/LLM 1 here to get seeds
    seeds = [{"price": 100}, {"price": 105}] # Mock data
    return {"grounding_seeds": seeds}

def generator_node(state: PipelineState):
    # Call NeMo/LLM 2 here using the seeds
    new_row = {"price": 102, "volatility": state["volatility_constraint"]}
    return {"generated_row": new_row}

def judge_node(state: PipelineState):
    # Call NeMo/LLM 3 to evaluate generated_row
    # Mocking a failure for the example
    critique = "FAIL: Volume missing" 
    return {"judge_critique": critique}

def refiner_node(state: PipelineState):
    # Call NeMo/LLM 4 to fix the row based on critique
    fixed_row = state["generated_row"]
    fixed_row["volume"] = 500 # Mock fix
    return {
        "generated_row": fixed_row, 
        "retry_count": state.get("retry_count", 0) + 1
    }

# 3. Define the Router (The Conditional Edge)
def route_after_judge(state: PipelineState):
    if state["judge_critique"] == "PASS":
        return END # Save the data
    elif state.get("retry_count", 0) >= 2:
        return END # Too many retries, discard
    else:
        return "refiner" # Send to the Refiner node

# 4. Build the Graph
workflow = StateGraph(PipelineState)

# Add the nodes to the graph
workflow.add_node("sampler", sampler_node)
workflow.add_node("generator", generator_node)
workflow.add_node("judge", judge_node)
workflow.add_node("refiner", refiner_node)

# Define the standard edges (the strict manufacturing line)
workflow.add_edge(START, "sampler")
workflow.add_edge("sampler", "generator")
workflow.add_edge("generator", "judge")
workflow.add_edge("refiner", "judge") # The loop back

# Define the conditional edge (the routing logic)
workflow.add_conditional_edges(
    "judge",
    route_after_judge,
)

# Compile the graph into an executable app
app = workflow.compile()

# Run it
initial_state = {
    "volatility_constraint": 0.5,
    "retry_count": 0
}
result = app.invoke(initial_state)
print(result)