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



instructions for running:
first made MODEL_INFOs file in model repo # appworld/experiments/code/models/vllm_local.py

set export OPENAI_API_KEY=EMPTY


then made jsonnet file

python experiments/configs/_generator/run.py \
  --model_names vllm-local-8000-qwen3-8b \
  --agent_names simplified_react_code_agent \
  --dataset_names test_normal


  then ran exp with:

appworld run auto \
  --agent-name simplified_react_code_agent \
  --model-name vllm-local-8000-qwen3-8b \
  --dataset-name test_normal

results stored like:
ubuntu@ip-172-31-39-1:~/appworld-source/appworld_source/appworld/experiments/outputs/simplified_react_code_agent/vllm_local/vllm-local-8000-qwen3-8b/test_normal/tasks$ ls
3d9a636_1  3d9a636_2  3d9a636_3  fd1f8fa_1


then do evaluation:
appworld evaluate simplified_react_code_agent/vllm_local/vllm-local-8000-qwen3-8b/test_normal test_normal

