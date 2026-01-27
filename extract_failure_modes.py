#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# -------------------------
# Utilities
# -------------------------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_text(path: Path, max_chars: int = 200_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]

def read_jsonl(path: Path, max_lines: int = 50_000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines rather than exploding
                continue
    return rows

def safe_get(obj: Any, path: List[Any], default=None):
    cur = obj
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                return default
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
    return cur

def ensure_path(prompt: str) -> Path:
    while True:
        s = input(prompt).strip()
        p = Path(s).expanduser().resolve()
        if p.exists():
            return p
        print(f"Path not found: {p}")

def find_appworld_root_from_experiment_dir(experiment_dir: Path) -> Path:
    """
    Walk upward until we find a directory containing:
      - experiments/outputs
      - data
    That’s a good heuristic for APPWORLD_ROOT in your layout.
    """
    cur = experiment_dir.resolve()
    for _ in range(20):
        if (cur / "experiments" / "outputs").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(
        "Could not infer APPWORLD_ROOT. Expected a parent directory containing both "
        "'experiments/outputs' and 'data'."
    )

def infer_experiment_name(appworld_root: Path, experiment_dir: Path) -> str:
    """
    experiment_name is the path *relative* to:
      APPWORLD_ROOT/experiments/outputs/
    """
    outputs_root = appworld_root / "experiments" / "outputs"
    experiment_dir = experiment_dir.resolve()
    try:
        rel = experiment_dir.relative_to(outputs_root)
    except ValueError:
        raise RuntimeError(
            f"Experiment dir {experiment_dir} is not under {outputs_root}"
        )
    return str(rel).replace("\\", "/")

def run_appworld_evaluate_full(appworld_root: Path, experiment_name: str, dataset_name: str) -> None:
    cmd = ["appworld", "evaluate", experiment_name, dataset_name]
    print(f"\nRunning: {' '.join(cmd)} (cwd={appworld_root})\n")
    subprocess.run(cmd, cwd=str(appworld_root), check=True)

def run_appworld_evaluate_present_tasks(
    appworld_root: Path,
    experiment_dir: Path,
    experiment_name: str,
) -> List[str]:
    """
    Evaluate ONLY tasks that exist in experiment_dir/tasks/.
    Returns the list of task_ids evaluated.
    """
    tasks_dir = experiment_dir / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Expected tasks dir: {tasks_dir}")

    task_ids = sorted([p.name for p in tasks_dir.iterdir() if p.is_dir()])
    if not task_ids:
        print(f"No tasks found under: {tasks_dir}")
        return []

    for i, task_id in enumerate(task_ids, start=1):
        cmd = ["appworld", "evaluate", experiment_name, "--task-id", task_id]
        print(f"\n[{i}/{len(task_ids)}] Running: {' '.join(cmd)} (cwd={appworld_root})\n")
        subprocess.run(cmd, cwd=str(appworld_root), check=True)

    return task_ids

def load_evaluations_full(experiment_dir: Path, dataset_name: str) -> Dict[str, Any]:
    eval_json = experiment_dir / "evaluations" / f"{dataset_name}.json"
    if not eval_json.exists():
        raise FileNotFoundError(f"Expected evaluation json at: {eval_json}")
    return read_json(eval_json)

def load_evaluations_present(experiment_dir: Path, task_ids: List[str]) -> Dict[str, Any]:
    """
    Merge evaluations/on_only_<task_id>.json into a dataset-like dict:
      {"aggregate": {...best-effort...}, "individual": {task_id: {...}}}
    Aggregate is optional / best-effort since per-task aggregate is not meaningful.
    """
    evaluations_dir = experiment_dir / "evaluations"
    individual: Dict[str, Any] = {}
    missing: List[str] = []

    for task_id in task_ids:
        p = evaluations_dir / f"on_only_{task_id}.json"
        if not p.exists():
            missing.append(task_id)
            continue
        data = read_json(p)
        entry = safe_get(data, ["individual", task_id], default=None)
        if isinstance(entry, dict):
            individual[task_id] = entry
        else:
            missing.append(task_id)

    # “aggregate” is not meaningful across per-task runs; keep it minimal.
    merged = {
        "aggregate": {
            "note": "Merged from per-task evaluations (on_only_<task_id>.json). Aggregate metrics are not computed.",
            "num_tasks_evaluated": len(individual),
            "num_tasks_missing_eval": len(missing),
        },
        "individual": individual,
        "missing_eval_task_ids": missing,
    }
    return merged

def extract_transcript_from_lm_calls(lm_calls_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Produces a simple list of {role, content} messages.
    It uses the structure you showed: row["input"]["messages"] and row["output"]["choices"][0]["message"].
    Drops 'reasoning' and 'reasoning_content' by only keeping role/content.
    """
    transcript: List[Dict[str, str]] = []

    for row in lm_calls_rows:
        in_messages = safe_get(row, ["input", "messages"], default=[])
        if isinstance(in_messages, list):
            for m in in_messages:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role", "") or "")
                content = m.get("content", "")
                if content is None:
                    content = ""
                transcript.append({"role": role, "content": str(content)})

        out_msg = safe_get(row, ["output", "choices", 0, "message"], default=None)
        if isinstance(out_msg, dict):
            role = str(out_msg.get("role", "assistant") or "assistant")
            content = out_msg.get("content", "")
            if content is None:
                content = ""
            transcript.append({"role": role, "content": str(content)})

    return transcript

def summarize_transcript(transcript: List[Dict[str, str]], max_chars: int = 60_000) -> str:
    """
    Convert transcript to a readable string, truncated.
    """
    chunks: List[str] = []
    for m in transcript:
        role = m.get("role", "")
        content = m.get("content", "")
        chunks.append(f"{role.upper()}:\n{content}\n")
    s = "\n".join(chunks)
    return s[:max_chars]


# -------------------------
# LLM Classification (GPT-4o)
# -------------------------

FAILURE_SCHEMA: Dict[str, Any] = {
    "name": "failure_classification",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "task_id": {"type": "string"},
            "success": {"type": "boolean"},
            "primary_category": {
                "type": "string",
                "enum": [
                    "missing_api_call_or_wrong_api_name",
                    "wrong_api_parameters_or_schema_mismatch",
                    "pagination_or_incomplete_iteration",
                    "auth_or_credentials_issue",
                    "reasoning_or_planning_error",
                    "repetition_or_loop",
                    "tooling_runtime_error",
                    "context_length_or_token_limit",
                    "formatting_or_code_block_error",
                    "other",
                ],
            },
            "secondary_categories": {
                "type": "array",
                "items": {"type": "string"},
            },
            "root_cause": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
            },
            "suggested_fix": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "task_id",
            "success",
            "primary_category",
            "secondary_categories",
            "root_cause",
            "evidence",
            "suggested_fix",
            "confidence",
        ],
    },
}

def classify_failure_gpt4o(
    client: OpenAI,
    task_id: str,
    eval_entry: Dict[str, Any],
    transcript_text: str,
    env_io_text: str,
) -> Dict[str, Any]:
    """
    Uses GPT-4o with JSON schema output.
    """
    # Keep the prompt short-ish; include only the most helpful context.
    prompt = f"""
You are analyzing an AppWorld agent run failure.

Task ID: {task_id}

Evaluation entry (from evaluations JSON):
{json.dumps(eval_entry, indent=2)[:20_000]}

Conversation transcript (from lm_calls.jsonl):
{transcript_text}

Environment I/O (from environment_io.md, if present):
{env_io_text[:20_000]}

Return a classification following the provided JSON schema.
Focus on the primary failure reason and give concrete evidence lines/snippets.
"""

    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        response_format={
            "type": "json_schema",
            "json_schema": FAILURE_SCHEMA,
        },
    )
    # resp.output_text should be valid JSON per schema
    return json.loads(resp.output_text)


# -------------------------
# Main pipeline
# -------------------------

def analyze_experiment(
    experiment_dir: Path,
    dataset_name: str,
    run_eval: bool = True,
    eval_mode: str = "present",  # "present" | "full"
) -> Path:
    appworld_root = find_appworld_root_from_experiment_dir(experiment_dir)
    experiment_name = infer_experiment_name(appworld_root, experiment_dir)

    # Decide evaluation strategy
    task_ids_present: List[str] = []
    if run_eval:
        if eval_mode == "full":
            run_appworld_evaluate_full(appworld_root, experiment_name, dataset_name)
        elif eval_mode == "present":
            task_ids_present = run_appworld_evaluate_present_tasks(
                appworld_root=appworld_root,
                experiment_dir=experiment_dir,
                experiment_name=experiment_name,
            )
        else:
            raise ValueError("eval_mode must be 'full' or 'present'")

    # Load evaluation results
    if eval_mode == "full":
        evaluation = load_evaluations_full(experiment_dir, dataset_name)
    else:
        # If we didn't run eval now, still infer task_ids from disk
        if not task_ids_present:
            tasks_dir = experiment_dir / "tasks"
            task_ids_present = sorted([p.name for p in tasks_dir.iterdir() if p.is_dir()]) if tasks_dir.exists() else []
        evaluation = load_evaluations_present(experiment_dir, task_ids_present)

    individual = evaluation.get("individual", {})

    failed_task_ids: List[str] = []
    for tid, entry in individual.items():
        if entry.get("success") is False:
            failed_task_ids.append(tid)

    print(f"\nEval mode: {eval_mode}")
    print(f"Found {len(individual)} evaluated tasks.")
    print(f"Found {len(failed_task_ids)} failed tasks.\n")

    client = OpenAI()

    results: Dict[str, Any] = {
        "experiment_dir": str(experiment_dir),
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "eval_mode": eval_mode,
        "num_evaluated_tasks": len(individual),
        "num_failed_tasks": len(failed_task_ids),
        "failures": [],
        "evaluation_meta": evaluation.get("aggregate", {}),
        "missing_eval_task_ids": evaluation.get("missing_eval_task_ids", []),
    }

    for i, task_id in enumerate(failed_task_ids, start=1):
        print(f"[{i}/{len(failed_task_ids)}] Classifying {task_id} ...")

        task_entry = individual.get(task_id, {})
        logs_dir = experiment_dir / "tasks" / task_id / "logs"

        lm_calls_path = logs_dir / "lm_calls.jsonl"
        env_io_path = logs_dir / "environment_io.md"

        lm_calls_rows = read_jsonl(lm_calls_path)
        transcript = extract_transcript_from_lm_calls(lm_calls_rows)
        transcript_text = summarize_transcript(transcript, max_chars=60_000)

        env_io_text = read_text(env_io_path, max_chars=60_000)

        if not transcript_text.strip() and not env_io_text.strip():
            classification = {
                "task_id": task_id,
                "success": False,
                "primary_category": "other",
                "secondary_categories": ["missing_logs"],
                "root_cause": "Could not find usable lm_calls.jsonl/environment_io.md for this task.",
                "evidence": [f"Missing or empty: {lm_calls_path}", f"Missing or empty: {env_io_path}"],
                "suggested_fix": "Ensure the agent run writes lm_calls.jsonl and environment_io.md into tasks/<task_id>/logs/.",
                "confidence": 0.3,
            }
        else:
            classification = classify_failure_gpt4o(
                client=client,
                task_id=task_id,
                eval_entry=task_entry,
                transcript_text=transcript_text,
                env_io_text=env_io_text,
            )

        results["failures"].append({
            "task_id": task_id,
            "evaluation": task_entry,
            "classification": classification,
            "logs_dir": str(logs_dir),
        })

    out_path = experiment_dir / f"failure_analysis_{dataset_name}_{eval_mode}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}\n")
    return out_path

def main():
    print("\n=== AppWorld Failure Analyzer ===\n")
    exp_dir = ensure_path("Enter FULL path to the experiment directory (contains tasks/, evaluations/):\n> ")
    dataset = input("Enter dataset name (e.g., test_normal):\n> ").strip()
    if not dataset:
        raise ValueError("dataset_name is required.")

    mode = input("Evaluation mode: 'present' (only tasks on disk) or 'full' (entire dataset)? [present/full]:\n> ").strip().lower()
    if mode not in ("present", "full", ""):
        raise ValueError("mode must be 'present' or 'full'")
    if mode == "":
        mode = "present"

    run_eval_str = input("Run `appworld evaluate` now? [Y/n]:\n> ").strip().lower()
    run_eval = (run_eval_str != "n")

    analyze_experiment(exp_dir, dataset_name=dataset, run_eval=run_eval, eval_mode=mode)


if __name__ == "__main__":
    main()
