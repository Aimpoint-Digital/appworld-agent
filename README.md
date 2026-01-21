# appworld-agent

## Task list
- run AppWorld benchmark on target SLM, Qwen/Qwen3-8B
- use llm to recursively summarize main failure modes, and log example failures
- test local fastapi app with vllm, appworld to get inputs and outputs to ensure code is using structure correctly in pydantic models/signatures
- build out helper functions intelligent_truncation, agent_repl, code_extractor
- need to get task id for task at hand from appworld. this can be done with minimal edits to llm call that supports extra args, test
- finish and test main app
- run on single appworld task
- run benchmarks for truncation + repl, repl only, truncation only
- pull out successful runs, and compare to benchmark failures. track all failures and summarize % of categories reduced for each