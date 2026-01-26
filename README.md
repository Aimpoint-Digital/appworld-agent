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

## Task List / TODO

* [ ] Run AppWorld benchmark on target SLM (Qwen/Qwen3-8B)
* [ ] Use LLM to recursively summarize main failure modes (dev/test_normal; avoid target leakage)
* [ ] Validate FastAPI + vLLM payload structure matches AppWorld expectations (Pydantic/signatures)
* [ ] Build helper functions:

  * `intelligent_truncation`
  * `agent_repl`
  * `code_extractor`
* [ ] Ensure task id is propagated from AppWorld → model call (`user` field)
* [ ] Finish + test main app
* [ ] Run a single AppWorld task end-to-end
* [ ] Run benchmarks:

  * truncation + repl
  * repl only
  * truncation only
* [ ] Compare successful runs vs baseline failures and compute category deltas

---

## Common pitfalls

### Running `appworld evaluate` from the wrong directory

Run from `APPWORLD_ROOT`, otherwise it may not find `./data` or may look for outputs in the wrong relative location.

### Missing `user` field propagation

If the agent doesn’t set `user="<task_id>"`, the proxy can’t associate requests with a task and will skip interventions.

### vLLM endpoint mismatch

Make sure `VLLM_BASE_URL` matches where `vllm serve` is listening and the proxy points to `/v1/chat/completions`.

---

## Quick end-to-end checklist

1. Start vLLM in tmux
2. Activate AppWorld env
3. `appworld download data` (once)
4. Generate configs
5. `appworld run auto ...`
6. `appworld evaluate ...`
7. Run failure analyzer script to produce `failure_analysis_<dataset>.json`

---
