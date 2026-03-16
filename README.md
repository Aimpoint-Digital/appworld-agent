# appworld-agent

Code for the paper: **"Inference-Time Scaffolding for Small Language Model Agents: Doubling AppWorld Performance Without Additional Training"**

This repo implements a three-tier inference scaffolding pipeline that deploys the same frozen Qwen3-8B model in three roles to improve performance on the [AppWorld benchmark](https://github.com/stonybrooknlp/appworld) without any additional training.

---

## How the Code Maps to the Paper

| File | Paper Role |
|------|-----------|
| `main.py` | FastAPI proxy implementing the correction module (Tier 3) — intercepts agent actions, checks for errors, generates fixes using isolated re-invocation of the model |
| `summary_module.py` | Summarization module (Tier 2) — compresses conversation history when context exceeds thresholds, preserving credentials, API schemas, and error patterns |
| `extract_failure_modes.py` | Failure mode taxonomy pipeline — sends failed task transcripts to GPT-4o for structured classification into the categories reported in Table 3 |
| `failure_analysis_summary.py` | Aggregates classification results into summary statistics |
| `utils/helpers.py` | Shared utilities (code extraction, error detection, API doc retrieval) |

> **Note:** `state_registry.py` is an experimental credential manager that was not included in the final paper results. It can be ignored.

---

## Prerequisites: Two Repos Required

This repo works alongside a patched fork of AppWorld. You need both:

1. **This repo** — `Aimpoint-Digital/appworld-agent`
2. **Aimpoint AppWorld fork** — `Aimpoint-Digital/appworld`

> ⚠️ Do **not** install AppWorld from the upstream repo. The Aimpoint fork contains critical patches required for this pipeline to function.

### What the fork patches

- Fixes reasoning extraction and parsing in the `simplified_react_code_agent`
- Propagates `task_id` through the OpenAI-compatible `user` field in each request, so the proxy can associate model calls with tasks:

```python
def extract_task_id(req: ChatCompletionRequest) -> Optional[str]:
    return req.user or None
```

Without this patch, the correction proxy cannot identify which task is running and will skip all interventions.

---

## Environment Setup

You need **three separate virtual environments** to avoid dependency conflicts:

| Env | Purpose | Install from |
|-----|---------|-------------|
| `appenv` | AppWorld benchmark runner | Aimpoint AppWorld fork |
| `vllmenv` | vLLM model server | `pip install vllm` |
| `proxyenv` | This repo (FastAPI proxy + analysis scripts) | `requirements.txt` |

### 1. Set up AppWorld (appenv)

```bash
git clone <AIMPOINT_APPWORLD_FORK_URL> appworld_source
cd appworld_source

python -m venv ~/appenv
source ~/appenv/bin/activate

pip install -U "click==8.1.7"   # required before data download
pip install -e .                 # install fork, not upstream
appworld install --repo
appworld download data
```

Set the AppWorld root (required after every reboot):

```bash
export APPWORLD_ROOT=/path/to/appworld_source/appworld
```

### 2. Set up vLLM (vllmenv)

```bash
python -m venv ~/vllmenv
source ~/vllmenv/bin/activate
pip install vllm
```

### 3. Set up this repo (proxyenv)

```bash
git clone <AIMPOINT_APPWORLD_AGENT_URL> appworld-agent
cd appworld-agent

python -m venv ~/proxyenv
source ~/proxyenv/bin/activate
pip install -r requirements.txt
```

---

## Model Configuration

AppWorld resolves models by name from a registry file. Add the following entry to:

```
appworld_source/appworld/experiments/configs/_generator/models/vllm_local.py
```

```python
MODEL_INFOS = [
    {
        "model_name": "vllm-local-8000-qwen3-8b",
        "client_name": "openai",
        "model_id": "Qwen/Qwen3-8B",          # must match what vLLM is serving
        "model_kwargs": {
            "api_type": "chat_completions",
            "temperature": 0,
            "seed": 100,
            "api_key_env_name": "NO_API_KEY",
            "base_url": "http://127.0.0.1:8000/v1",   # point to proxy or vLLM directly
            "max_completion_tokens": 3000,
            "tool_parser_name": None,
            "parallel_tool_calls": True,
            "cost_per_token": {
                "input_cache_miss": 0.0,
                "input_cache_hit": 0.0,
                "input_cache_write": 0.0,
                "output": 0.0,
            },
        },
        "function_calling": True,
        "tool_choice": "auto",
        "function_calling_demos": False,
        # Required to handle vLLM tool schema quirks
        "remove_function_property_keys": [
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minimum",
            "maximum",
        ],
        "model_server_config": {
            "enabled": False,    # prevents AppWorld from trying to manage the server
        },
        "part_of": ["vn", "vllm"],
        "provider": "vllm",
    },
]
```

Then generate the jsonnet config:

```bash
source ~/appenv/bin/activate
cd $APPWORLD_ROOT

python experiments/configs/_generator/run.py \
  --model_names vllm-local-8000-qwen3-8b \
  --agent_names simplified_react_code_agent \
  --dataset_names test_normal
```

---

## Running Experiments

The three runs below correspond directly to the ablation study in the paper (Table 4). Each run uses a distinct experiment name to prevent output collisions — AppWorld stores outputs under `experiments/outputs/{experiment_name}/`.

---

### Run A — Baseline (direct vLLM, no proxy)

**1. Start vLLM (FP16)**

```bash
source ~/vllmenv/bin/activate
tmux new -s vllm

vllm serve Qwen/Qwen3-8B \
  --reasoning-parser qwen3 \
  --max-model-len 12000 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 4 \
  --port 8000
```
Detach: `Ctrl-b d`

Make sure `base_url` in your MODEL_INFOS points to `http://127.0.0.1:8000/v1`.

**2. Run benchmark**

```bash
source ~/appenv/bin/activate
cd $APPWORLD_ROOT
export OPENAI_API_KEY=EMPTY

appworld run auto \
  --agent-name simplified_react_code_agent \
  --model-name vllm-local-8000-qwen3-8b \
  --dataset-name test_normal
```

**3. Evaluate** (must run from APPWORLD_ROOT)

```bash
appworld evaluate vllm-local-8000-qwen3-8b test_normal
```

**4. Classify failures**

Requires a real OpenAI API key — uses `gpt-4o-mini` for classification.

```bash
source ~/proxyenv/bin/activate
export OPENAI_API_KEY=<your_real_key>
python extract_failure_modes.py
```

The script is interactive. When prompted:
- **Experiment directory**: full path to the experiment's output dir (contains `tasks/` and `evaluations/`)
- **Dataset name**: `test_normal`
- **Evaluation mode**: `full` (scores all 168 tasks) or `present` (only tasks on disk)
- **Run appworld evaluate now?**: `Y` to run evaluation as part of the script, `n` to skip if already done

Output is written to `failure_analysis_{dataset}_{mode}.json` inside the experiment directory.

---

### Run B — Correction Only (proxy ON, summarization OFF)

**1. Make sure vLLM is running on port 8001** (restart if needed with `--port 8001`)

**2. Start proxy with summarization disabled**

```bash
source ~/proxyenv/bin/activate
tmux new -s proxy

export VLLM_BASE_URL="http://127.0.0.1:8001"
export ENABLE_CONTEXT_SUMMARY=0
export OPENAI_API_KEY=EMPTY
export PROXY_LOG_PATH="proxy_correction_only.log"

uvicorn main:app --host 0.0.0.0 --port 8000
```
Detach: `Ctrl-b d`

Update `base_url` in MODEL_INFOS (or add a new model entry `vllm-proxy-8000-qwen3-8b`) to point to `http://127.0.0.1:8000/v1`.

**3. Run benchmark**

```bash
appworld run auto \
  --agent-name simplified_react_code_agent \
  --model-name vllm-proxy-8000-qwen3-8b \
  --dataset-name test_normal
```

**4. Evaluate**

```bash
appworld evaluate vllm-proxy-8000-qwen3-8b test_normal
```

**5. Classify failures**

```bash
export OPENAI_API_KEY=<your_real_key>
python extract_failure_modes.py
```

Follow the same interactive prompts as Run A, pointing to the proxy experiment directory.

---

### Run C — Full Scaffold (proxy ON, summarization ON)

**1. Stop prior proxy session and start new one**

```bash
tmux kill-session -t proxy
tmux new -s proxy_full

export VLLM_BASE_URL="http://127.0.0.1:8001"
export ENABLE_CONTEXT_SUMMARY=1
export SUMMARY_CHAR_THRESHOLD=24000
export SUMMARY_TOKEN_THRESHOLD=6000
export SUMMARY_KEEP_LAST_K=6
export SUMMARY_KEEP_FIRST_N=26                            # retain first 26 messages verbatim
export SUMMARY_MODEL="Qwen/Qwen3-8B"                     # same frozen model, routed via vLLM
export SUMMARY_VLLM_BASE_URL="http://127.0.0.1:8001"     # summarizer uses vLLM directly, not OpenAI
export VLLM_CONTEXT_LEN=32768                             # match AWQ max-model-len (12000 for FP16)
export OPENAI_API_KEY=EMPTY
export PROXY_LOG_PATH="proxy_full_scaffold.log"

uvicorn main:app --host 0.0.0.0 --port 8000
```
Detach: `Ctrl-b d`

**2. Run benchmark**

```bash
appworld run auto \
  --agent-name simplified_react_code_agent \
  --model-name vllm-proxy-sum-8000-qwen3-8b \
  --dataset-name test_normal
```

**3. Evaluate**

```bash
appworld evaluate vllm-proxy-sum-8000-qwen3-8b test_normal
```

**4. Classify failures**

```bash
export OPENAI_API_KEY=<your_real_key>
python extract_failure_modes.py
```

Follow the same interactive prompts as Run A, pointing to the full scaffold experiment directory.

---

## AWQ Configuration (4-bit Quantized, 32K Context)

To reproduce the AWQ results from the paper, replace the vLLM serve command with:

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --port 8001 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --reasoning-parser qwen3
```

Update `model_id` in MODEL_INFOS to `Qwen/Qwen3-8B-AWQ` and use a distinct model name (e.g. `vllm-local-8000-qwen3-8b-awq`) to keep outputs separate from FP16 runs.

Everything else (proxy setup, run commands, failure classification) is identical to the FP16 instructions above.

---

## Failure Mode Analysis

`extract_failure_modes.py` is an interactive script that:
1. Optionally runs `appworld evaluate` to score tasks
2. Reads `lm_calls.jsonl` and `environment_io.md` for each failed task
3. Sends transcripts to `gpt-4o-mini` for structured classification
4. Writes results to `failure_analysis_{dataset}_{mode}.json` in the experiment directory

Run it once per experiment directory you want to analyze:

```bash
source ~/proxyenv/bin/activate
export OPENAI_API_KEY=<your_real_key>
python extract_failure_modes.py
```

Each classified failure in the output includes:

- `primary_category` — one of: `auth_or_credentials_issue`, `reasoning_or_planning_error`, `wrong_api_parameters_or_schema_mismatch`, `missing_api_call_or_wrong_api_name`, `repetition_or_loop`, `pagination_or_incomplete_iteration`, `formatting_or_code_block_error`, `context_length_or_token_limit`, `other`
- `secondary_categories` — contributing factors
- `root_cause` — short explanation
- `evidence` — transcript snippets
- `suggested_fix` — concrete fix idea
- `confidence` — 0–1 classification confidence

To compare failure distributions before and after scaffolding (Table 6 in the paper), run `failure_analysis_summary.py` against two output files:

```bash
python failure_analysis_summary.py
```

---

## Common Pitfalls

**1. `appworld evaluate` must run from APPWORLD_ROOT**

AppWorld looks for `./data` relative to the working directory. Running from anywhere else produces a "Did not find any ./data" error.

**2. APPWORLD_ROOT resets on reboot**

Re-export after every server restart:

```bash
export APPWORLD_ROOT=/path/to/appworld_source/appworld
```

Consider adding this to your `~/.bashrc` or tmux session startup.

**3. Use distinct experiment names per run**

AppWorld stores outputs under `./experiments/outputs/{experiment_name}/tasks/{task_id}/dbs`. Reusing the same name across runs will overwrite prior results.

**4. Missing `user` field propagation**

If you are not using the Aimpoint fork, the agent will not set `user="<task_id>"` in model requests. The proxy uses this field to associate corrections with the right task — without it, all interventions are silently skipped.

**5. vLLM port mismatch**

The proxy (`port 8000`) forwards to vLLM (`port 8001`). AppWorld talks to the proxy. Make sure these match your MODEL_INFOS `base_url` and the `VLLM_BASE_URL` env var.

**6. Missing `--reasoning-parser qwen3` causes malformed outputs**

Qwen3 models emit thinking tags in their outputs. Without `--reasoning-parser qwen3` in the vLLM serve command, these tags are not stripped and the agent receives malformed responses. Always include this flag for both FP16 and AWQ configurations.

---

## Citation

If you use this code, please cite:

```bibtex
@article{mcclendon2025scaffolding,
  title={Inference-Time Scaffolding for Small Language Model Agents: 
         Doubling AppWorld Performance Without Additional Training},
  author={McClendon, Aaron and others},
  journal={arXiv preprint},
  year={2025}
}
```
