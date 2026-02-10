This is a component for a stock trading platform to generate usable quality synthetic data (AI-generated) trade logs based on human trade logs.

This is an infrastructure component. ML models are pre-trained Nvidia ones; what I've done is programmed this for my use case, selected the models, and created a data science method to ensure quality results as opposed to statistical noise.

My results can be verified via my backtesting system and Hyperband system. This reveals over/under fitted results.

Note: LLMs are terrible at technical indicators. This only generates the core data, and then I have to manually compute the technicals myself based on that data. If I generated these technicals, it would pollute everything with hallucinations.

### Architecture
**Main Tools**
- Nvidia NeMo Data Designer: This is a data science platform specifically made for synthetic data generation. It was released alongside tons of datasets and trained NeMo models for different purposes.
- Nvidia NIM: Using an API, run ML models on Nvidia's servers instead of a local computer.

**Design**
Data is processed by 3 LLMs + a sampler.
    - The Sampler (not an llm): Giving the generator all my data (one shot) would make it average everything out (bad results). This python code feeds the generator slowly using a "few-shot" approach. Basically, it selects something from my volatility percent diversity constraint, and it selects 3 gold data points satisfying that condition and gives them to the generator LLM.
        - you can improve results by using an llm but idk the real gain this would bring and it would raise complexity.

    - Generator: Take in a few gold data points at a time from the sampler and generate a JSON result of a new data row. This creates a new column for each row containing a JSON of the new row. This only generates core data; I have to manually compute the technical indicators myself since LLMs are weak at this. Because it's only doing 3 data points at a time, its variance is much higher than if it saw everything at once. This prevents something called "mode collapse."
    
    - Judge: Evaluates the Generator results based on 3 criteria - realism, consistency, and completeness. If it passes, it's saved; if it fails, it writes a critique string explaining why it failed. That critique is fed into the refiner loop.

    - Refiner: In a loop with the judge LLM. Results that the judge fails go here. It receives the judge critique + original few-shot gold examples + failed generated JSON row, and makes minimal edits to fix only the criticized issue while preserving valid fields. Output is forced back into strict JSON form. This is capped at 2 retries and then thrown away.

**LangGraph**
- state: contains volatility_constraint, grounding_seeds (the few-shot data), generated_row (the JSON result), judge_critique (Judge LLM feedback), retry_count
- nodes: Each LLM gets a function which is a node.
    - sampler_node: Updates few shot examples
    - generator_node: Uses few shot examples to call generator, updates generated_row
    - judge_node: Evaluates generated_row, updates judge_critique
    - refiner_node: Reads judge_critique, updates generated_row, updates retry_count
- edges: Loops
    - Start at sampler_node
    - sampler -> generator
    - generator -> judge
    - Use a conditional edge to choose if judge goes to refiner or concludes (saves generated_row)
        - refiner -> judge


Examples / vocab:
- Diversity Control: You can select a column and assign value ranges and weights to it for the results.
```python
    volatility_sampler = SamplerColumnConfig(
        name="target_volatility", 
        column_type="sampler",
        sampler_type="category",
        params=CategorySamplerParams(
            values = [0.3,  0.4,  0.5, 0.6,  0.7, 0.8, 0.9,  1.0,  1.1],
            weights= [0.05, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.05, 0.05]
        )
    )
```

mode collapse: 

LangChain vs LangGraph
These are basically frameworks to use multiple LLMs to solve a problem.
- LangChain: Works in retrieve -> summarize -> answer. It basically has a few ways to do each step, but does summarize and answer in a "chain" of processes involving memory, prompts, and the LLMs.
    - This is a DAG and doesn't work with loops well or revisiting states.
- LangGraph: A subset of LangChain. It's made for state machines and complex non-linear workloads. You can add, complete, and summarize tasks. Each of these is 1 node (graph structure) with edges between them. These are fed by a process input (which is NOT an LLM) that routes to the correct node. There's also a state component that tracks each state. The idea is you keep context over long-term interactions.
    - This can have loops and revisiting states because it's not a DAG. 
    - Routing is done by conditional edges (if/else).
    - A node is actually any Python function (not an LLM and not locked to those 3 examples I said). In my case, each node will be code which calls the LLM.
    - state is a typed dictionary and is the shared memory passed between functions.
- summary: LangGraph is for multi-agent systems. LangChain is an abstraction layer to chain LLM operations into LLM applications.
