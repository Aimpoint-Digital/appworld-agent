# appworld-agent

Utilities + workflows for running **AppWorld** evaluations on small/target models (starting with **Qwen/Qwen3-8B via vLLM**) and analyzing failures (logs → LLM categorization → JSON report). Also includes a **FastAPI proxy** that can inject “interventions” like replaying code in AppWorld and generating “fix suggestions”.

## Goals

* Run AppWorld benchmark on target SLM (**Qwen/Qwen3-8B**), with a local vllm session and fastapi proxy session, separate from running appworld session
* Generate evaluation reports (`evaluations/<dataset>.json`).
* For failed tasks: collect logs + transcripts and classify failure modes with an external LLM (e.g. GPT-4o).
* Track deltas across interventions:

  * truncation + repl
  * repl only
  * truncation only
* Compare successful runs vs failures and quantify category reduction.

---

## Repo Layout (expected)

This repo assumes:

* You run experiments in **AppWorld checkout** (Aimpoint fork). Our fork makes edits to appworld source code which are critical for repo to run. 
* Output ends up under:

  ```
  APPWORLD_ROOT/experiments/outputs/<experiment_name>/...
  ```

Common output paths:

```
experiments/outputs/<experiment_name>/tasks/<task_id>/logs/
experiments/outputs/<experiment_name>/evaluations/<dataset>.json
```

---

## Prerequisites

### EC2 / Server assumptions

* Ubuntu EC2
* `tmux` used for long-running vLLM / eval runs
* vLLM serving Qwen3-8B locally
* Python venv(s) (example: `appenv`, `vllmenv`)

### Environment variables

* For AppWorld + vLLM proxy runs:

  * `OPENAI_API_KEY=EMPTY` (AppWorld expects it; proxy/vLLM uses Bearer EMPTY)
* For failure-mode classification with OpenAI:

  * `OPENAI_API_KEY=<real key>` (in the environment where classifier runs)

---

## 1) Use the Aimpoint Digital AppWorld fork

You **must** run with the Aimpoint fork because we patched the agent plumbing:

* Fix reasoning extraction / parsing in the ReAct code agent
* Pass **task id** through the OpenAI-compatible request `user` field so the proxy can recover it

NOTE: do not install appworld directly as stated in their readme instructions. you must run pip install -e . to install the fork. then run appworld install --repo, and then download the data. 

### Clone + checkout

```bash
git clone <AIMPOINT_APPWORLD_FORK_URL> appworld_source
cd appworld_source
git checkout <branch-with-agent-fixes>
```

> If you already have upstream AppWorld cloned, add the fork as a remote and checkout the fork branch:

```bash
git remote add aimpoint <AIMPOINT_APPWORLD_FORK_URL>
git fetch aimpoint
git checkout -b aimpoint-fork aimpoint/<branch>
```

### What changed in the fork (high level)

* ReAct code agent was modified to:

  * correctly extract / handle “reasoning” artifacts
  * set `user="<task_id>"` in requests sent to the model server (OpenAI chat completions format, edit in next_execution_inputs_usage_and_status)

This is critical because our proxy extracts task id via:

```python
def extract_task_id(req: ChatCompletionRequest) -> Optional[str]:
    return req.user or None
```

---

## 2) Start vLLM server (Qwen/Qwen3-8B)

Example command used on EC2:

```bash
# In tmux
source ~/vllm_env/bin/activate  # or your env
vllm serve Qwen/Qwen3-8B \
  --reasoning-parser qwen3 \
  --max-model-len 6384 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 16
```

Notes:

* The reasoning parser matters for Qwen3-style outputs.
* Keep max length conservative to avoid OOM.

---

## 3) Install AppWorld + dataset

Inside your AppWorld venv:

```bash
source ~/appenv/bin/activate
cd ~/appworld_source  # APPWORLD_ROOT

pip install -U "click==8.1.7"   # required before data download in our setup
appworld download data
```

---

## 4) Configure model entry (MODEL_INFOs)

We created a model entry in the AppWorld experiments model registry:

* File: `appworld/experiments/code/models/vllm_local.py`
* Add a `MODEL_INFOs` entry for something like `vllm-local-8000-qwen3-8b`

This lets AppWorld resolve `--model-name vllm-local-8000-qwen3-8b`.

Also set:

```bash
export OPENAI_API_KEY=EMPTY
```

---

## 5) Generate config + run experiment

### Generate jsonnet config

```bash
python experiments/configs/_generator/run.py \
  --model_names vllm-local-8000-qwen3-8b \
  --agent_names simplified_react_code_agent \
  --dataset_names test_normal
```

### Run the benchmark

```bash
appworld run auto \
  --agent-name simplified_react_code_agent \
  --model-name vllm-local-8000-qwen3-8b \
  --dataset-name test_normal
```

Outputs should appear under something like:

```
experiments/outputs/simplified_react_code_agent/vllm_local/vllm-local-8000-qwen3-8b/test_normal/tasks/
```

Example:

```bash
ls experiments/outputs/simplified_react_code_agent/vllm_local/vllm-local-8000-qwen3-8b/test_normal/tasks
# 3d9a636_1  3d9a636_2  3d9a636_3  fd1f8fa_1 ...
```

---

## 6) Evaluate the run

Run evaluation from **APPWORLD_ROOT** (important):

```bash
cd ~/appworld_source/appworld_source/appworld  # APPWORLD_ROOT
appworld evaluate simplified_react_code_agent/vllm_local/vllm-local-8000-qwen3-8b/test_normal test_normal
```

Evaluation outputs:

```
experiments/outputs/<experiment_name>/evaluations/test_normal.json
experiments/outputs/<experiment_name>/evaluations/test_normal.txt
```

---

## 7) FastAPI proxy (interventions / replay / fix suggestion)

This repo contains a FastAPI app that proxies OpenAI-style `/v1/chat/completions` to vLLM, and can:

* extract `task_id` from request `user`
* replay executable code from conversation history into `AppWorld(task_id=..., experiment_name=...)`
* execute newest assistant code
* if output looks like an error, call vLLM again to generate a short “fix suggestion”
* log all interventions as JSON lines to a log file

### How it works

Incoming request resembles:

```json
{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "goodbye"}],
  "temperature": 0.0,
  "user": "TASK_123"
}
```

The proxy uses:

* `req.user` → task id
* `code_extractor()` → extract python blocks from assistant content
* `world.execute(code)` → replay + execute
* `looks_like_error()` → regex-based error detection
* calls vLLM to produce a better “fix suggestion” response (optional)

### Run the proxy

Set env vars:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8001
export OPENAI_API_KEY=EMPTY
export PROXY_LOG_PATH=proxy_interventions.log
```

Run:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then point AppWorld model endpoint to the proxy (in your MODEL_INFOs / model config). The proxy forwards to vLLM.

---

## 8) Failure-mode extraction + classification

After evaluation, we want:

1. Read `evaluations/<dataset>.json`
2. Collect failed tasks: `individual[task_id].success == false`
3. For each failed task, gather logs (typically):

   * `tasks/<task_id>/logs/lm_calls.jsonl`
   * optionally `tasks/<task_id>/logs/environment_io.md`
4. Send transcript + eval info to an LLM (e.g. GPT-4o)
5. Save structured classification results into:

   * `failure_analysis_<dataset>.json`

### Output format (example)

* `primary_category`: one main bucket (wrong API, wrong args, loop, etc.)
* `secondary_categories`: optional
* `root_cause`: short explanation
* `evidence`: snippets pointing to where it went wrong
* `suggested_fix`: concrete fix idea
* `confidence`: 0–1

---

## Common pitfalls

### Running `appworld evaluate` from the wrong directory

Run from `APPWORLD_ROOT`, otherwise it may not find `./data` or may look for outputs in the wrong relative location.

### Missing `user` field propagation

If the agent doesn’t set `user="<task_id>"`, the proxy can’t associate requests with a task and will skip interventions.

### vLLM endpoint mismatch

Make sure `VLLM_BASE_URL` matches where `vllm serve` is listening and the proxy points to `/v1/chat/completions`.

---

## Run the Experiment

## 0) One-time setup (per machine / repo)

1. **Activate env + be at AppWorld repo root**

```bash
cd ~/appworld-source/appworld_source/appworld
source ../../../../appenv/bin/activate  # adjust if different
```

2. **Confirm data exists**

```bash
ls -la ./data/tasks | head
```

If missing:

```bash
appworld download data
```

3. **Start vLLM (Qwen3-8B)**

```bash
tmux new -s vllm
vllm serve Qwen/Qwen3-8B \
  --reasoning-parser qwen3 \
  --max-model-len 6384 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 16 \
  --port 8001
```

Detach: `Ctrl-b d`

4. **(If needed) generate configs (you already did, but here’s the canonical)**

```bash
python experiments/configs/_generator/run.py \
  --model_names vllm-local-8000-qwen3-8b \
  --agent_names simplified_react_code_agent \
  --dataset_names test_normal
```

---

## Run A — Baseline (direct vLLM, no proxy)

1. **Point AppWorld model to vLLM directly**

* Ensure your `MODEL_INFO` (or model config for `vllm-local-8000-qwen3-8b`) uses, jsut make sure the JSONNET file is using the same port the FastaPI/vLLM is running on:

  * `base_url: http://127.0.0.1:8001`
  * and `Authorization: Bearer EMPTY` 

2. **Run the benchmark**

```bash
appworld run auto \
  --agent-name simplified_react_code_agent \
  --model-name vllm-local-8000-qwen3-8b \
  --dataset-name test_normal
```

3. **Evaluate (from AppWorld repo root)**

```bash
appworld evaluate vllm-local-8000-qwen3-8b test_normal
```

4. **Classify failures (your script)**
Make sure openAI token set in environment

```bash
python scripts/extract_failure_modes.py \
  --experiment "vllm-local-8000-qwen3-8b" \
  --dataset "test_normal" \
  --out "experiments/outputs/vllm-local-8000-qwen3-8b/analysis/failure_modes.json"
```

---

## Run B — Proxy ON (post-processing), summarization OFF

### What changes?

* AppWorld model base_url now points to **FastAPI proxy** at `http://127.0.0.1:8000`
* Proxy forwards to vLLM at `http://127.0.0.1:8001` -> make sure vLLM is running on that port using instructions above
* Proxy env vars:

  * post-processing enabled (your default behavior)
  * summarization disabled

1. **Start the proxy (port 8000)**

```bash
tmux new -s proxy_no_summary
export VLLM_BASE_URL="http://127.0.0.1:8001"
export ENABLE_CONTEXT_SUMMARY=0
export OPENAI_API_KEY=EMPTY   # keep if AppWorld expects it set; summarization is off anyway
export PROXY_LOG_PATH="proxy_no_summary.log"
uvicorn path.to.your_proxy_module:app --host 0.0.0.0 --port 8000
```

Detach: `Ctrl-b d`

2. **Point AppWorld model to the proxy**

* `base_url: http://127.0.0.1:8000`
* model name can stay `vllm-local-8000-qwen3-8b` (AppWorld just passes it through)

3. **Run benchmark (use a distinct experiment name)**
   Best practice: create a separate model name like `vllm-proxy-8000-qwen3-8b` so outputs don’t collide. If you don’t want to add a model name, you can still isolate by running into different output roots, but simplest is: add a model entry.

Then run:

```bash
appworld run auto \
  --agent-name simplified_react_code_agent \
  --model-name vllm-proxy-8000-qwen3-8b \
  --dataset-name test_normal
```

4. **Evaluate**

```bash
appworld evaluate vllm-proxy-8000-qwen3-8b test_normal
```

5. **Classify failures**

```bash
python scripts/classify_failures.py \
  --experiment "vllm-proxy-8000-qwen3-8b" \
  --dataset "test_normal" \
  --out "experiments/outputs/vllm-proxy-8000-qwen3-8b/analysis/failure_modes.json"
```

---

## Run C — Proxy ON (post-processing), summarization ON

### What changes?

Just proxy env vars (and you’ll need a real OpenAI key if summarizing via OpenAI).

1. **Stop prior proxy tmux session and start a new one**

```bash
tmux kill-session -t proxy_no_summary
tmux new -s proxy_with_summary
export VLLM_BASE_URL="http://127.0.0.1:8001"
export ENABLE_CONTEXT_SUMMARY=1
export SUMMARY_CHAR_THRESHOLD=24000
export SUMMARY_TOKEN_THRESHOLD=6000
export SUMMARY_KEEP_LAST_K=6
export SUMMARY_MODEL="gpt-4o"         # or gpt-4o-mini
export OPENAI_API_KEY="YOUR_REAL_KEY"
export PROXY_LOG_PATH="proxy_with_summary.log"
uvicorn path.to.your_proxy_module:app --host 0.0.0.0 --port 8000
```

Detach: `Ctrl-b d`

2. **Run benchmark (distinct experiment name/model entry)**

```bash
appworld run auto \
  --agent-name simplified_react_code_agent \
  --model-name vllm-proxy-sum-8000-qwen3-8b \
  --dataset-name test_normal
```

3. **Evaluate**

```bash
appworld evaluate vllm-proxy-sum-8000-qwen3-8b test_normal
```

4. **Classify failures**

```bash
python scripts/classify_failures.py \
  --experiment "vllm-proxy-sum-8000-qwen3-8b" \
  --dataset "test_normal" \
  --out "experiments/outputs/vllm-proxy-sum-8000-qwen3-8b/analysis/failure_modes.json"
```

---

## Two “gotchas” that will save you pain

1. **Evaluation must run from AppWorld repo root** (where `./data` exists).
   Otherwise you’ll get the “Did not find any ./data” error you hit earlier.

2. **Make experiment names distinct**
   AppWorld’s evaluator expects outputs in:
   `./experiments/outputs/{experiment_name}/tasks/{task_id}/dbs`

So don’t reuse the same `{experiment_name}` across runs unless you really intend to overwrite.


---------------------------------------
/home/ubuntu/appworld-source/appworld_source/appworld/experiments/outputs/simplified_react_code_agent/vllm_local/vllm-local-8000-qwen3-8b/test_normal


setup and run baseline:

clone appworld fork

clone appworld-agents

make 3 venvs, can you use tmux in signularity? no do via exec commands in container file

need to make sure env set up, reqs are set up. need separate venvs for appworl and vllm/fastapi

clone each and install

for appworld, need to make the jsonnet config, add to model registry, 

test quantization and remove diff 1 and 2

appworld run auto   --agent-name simplified_react_code_agent   --model-name vllm-local-8000-qwen3-8b   --dataset-name test_normal

changing models:
had to set openai var. for baseline make sure vllm running on 8000. for qwen awq serving via
vllm serve Qwen/Qwen3-8B-AWQ   --port 8000   --max-model-len 32768   --max-num-seqs 1   --gpu-memory-utilization 0.90   --enable-chunked-prefill

had to change name in config file of model in appworld env

reset appwrold root after reboot export APPWORLD_ROOT=/home/ubuntu/appworld-source/appworld_source/appworld

/home/ubuntu/appworld-agents/appworld-agent/failure_analysis_test_normal_basemodel_full_context_qwen8b.json

changed model name in intercept env file