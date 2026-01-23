# appworld-agent

## Task list
- run AppWorld benchmark on target SLM, Qwen/Qwen3-8B
- use llm to recursively summarize main failure modes (on test/dev set to avoid target leakage), and log example failures
- test local fastapi app with vllm, appworld to get inputs and outputs to ensure code is using structure correctly in pydantic models/signatures
- build out helper functions intelligent_truncation, agent_repl, code_extractor
- need to get task id for task at hand from appworld. this can be done with minimal edits to llm call that supports extra args, test
- finish and test main app
- run on single appworld task
- run benchmarks for truncation + repl, repl only, truncation only
- pull out successful runs, and compare to benchmark failures. track all failures and summarize % of categories reduced for each


ec2:
running test at (vllmenv) ubuntu@ip-172-31-39-1:~/vllm$ vllm serve Qwen/Qwen3-8B   --reasoning-parser qwen3   --max-model-len 6384   --gpu-memory-utilization 0.9 --max-num-seqs 16

appworld pip install -U "click==8.1.7" run prior to data download

tmux being used

repo forked and cloned - appworld

checked, incomign request looking like:
{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "user", "content": "goodbye"}
  ],
  "temperature": 0.0,
  "user": "TASK_123"
}
