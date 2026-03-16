import os
import uuid
import json
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv, find_dotenv

from appworld import AppWorld
from summary_module import maybe_summarize_payload
from utils.helpers import (
    code_extractor,
    extract_api_calls,
    _strip_think_tags,
    _extract_patch_from_curated,
    _count_consecutive_no_code_assistant_msgs,
    _extract_failed_app_from_error,
    _build_api_docs_context,
)
from state_registry import StateRegistry, CAPTURE_SUFFIX

load_dotenv(find_dotenv())

# -------------------------
# Logging setup
# -------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_PATH = os.getenv("PROXY_LOG_PATH", "proxy_interventions.log")
ENABLE_STATE_REGISTRY = os.getenv("ENABLE_STATE_REGISTRY", "1") == "1"

logger = logging.getLogger("appworld_proxy")
logger.setLevel(LOG_LEVEL)

# Avoid double-handlers if reloaded by uvicorn
if not logger.handlers:
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)


def log_event(event: str, **fields: Any) -> None:
    """
    Emit a single JSON log line to make later parsing easy (Datadog, ELK, etc.)
    """
    payload = {
        "event": event,
        "ts": datetime.utcnow().isoformat() + "Z",
        **fields,
    }
    logger.info("\n\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


# -------------------------
# ENV settings, configs
# -------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001")
VLLM_CHAT_URL = f"{VLLM_BASE_URL}/v1/chat/completions"
VLLM_MODEL = os.getenv("VLLM_MODEL")

ERROR_PATTERNS = [
    r"^Execution failed",  # AppWorld standard
    r"Traceback \(most recent call last\):",
    r"\b(Exception|Error):",  # generic Python errors
]

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list] = None


class ChatCompletionRequest(BaseModel):
    """
    appworld may pass through some fields i dont need to edit here, so just pass through
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    user: Optional[str] = None


def extract_task_id(req: ChatCompletionRequest) -> Optional[str]:
    return req.user or None


def build_vllm_payload(req: ChatCompletionRequest) -> Dict[str, Any]:
    """
    this can be used to generate an intelligent summary if chat getting too long
    """
    return req.model_dump(exclude_none=True)


def _rand_experiment_name(task_id: str) -> str:
    return f"proxy-{task_id}-{uuid.uuid4().hex[:8]}"


def looks_like_error(world_out: Any) -> bool:
    if not isinstance(world_out, str):
        return False
    return any(re.search(p, world_out, flags=re.MULTILINE) for p in ERROR_PATTERNS)


# NOTE: could try to force a code response after pondering, rather than possibly only returning reasoning content
async def get_fix_suggestion_from_vllm(
    task_id: Optional[str],
    code: str,
    world_out: str,
    api_docs_text: str = "",
    model: str = VLLM_MODEL,
    state_context: str = "",
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are debugging Python code executed inside AppWorld.\n"
                "Your job is to produce a SHORT, ACTIONABLE intervention that the agent can use immediately.\n"
                "Do NOT ask questions. Do NOT use input(). Do NOT invent tool outputs or facts.\n"
                "You MUST base corrections on the provided API docs text when available.\n\n"
                "You MUST use the provided variable state when available — do NOT re-derive values that already exist.\n"
                "Classify the failure using ONE primary category from this list:\n"
                "- missing_api_call_or_wrong_api_name\n"
                "- wrong_api_parameters_or_schema_mismatch\n"
                "- pagination_or_incomplete_iteration\n"
                "- auth_or_credentials_issue\n"
                "- reasoning_or_planning_error\n"
                "- repetition_or_loop\n"
                "- tooling_runtime_error\n"
                "- formatting_or_code_block_error\n"
                "- other\n\n"
                "Output format (exact):\n"
                "PRIMARY_CATEGORY: <one from list>\n"
                "EVIDENCE: <1-3 short quotes from the execution output>\n"
                "DIAGNOSIS: <1-2 sentences>\n"
                "FIX_STEPS:\n"
                "- <2-6 concrete bullet steps>\n"
                "PATCH:\n"
                "```python\n"
                "<corrected code>\n"
                "```\n\n"
                "PATCH RULES (strict):\n"
                "- PATCH IS REQUIRED. Never output PATCH: (omitted).\n"
                "- PATCH must contain ONLY a single fenced python code block (no prose before/after).\n"
                "- The code MUST be directly executable in AppWorld.\n"
                "- The code MUST perform at least one AppWorld API call (e.g., apis.<app>.<api>(...)).\n"
                "- If the fix is uncertain, still output a minimal executable patch that gathers the missing info via API docs,\n"
                "  e.g. print(apis.api_docs.show_api_doc(app_name=..., api_name=...)) and then returns/prints what to do next.\n"
                # "- If the failure relates to task completion, the patch MUST call apis.supervisor.complete_task(...) when appropriate.\n"
                "- NEVER call apis.supervisor.complete_task() in your patch. "
                "- NEVER call apis.supervisor.complete_task() in your patch. "
                "  The patch should fix the immediate error, not complete the task. "
                "  Task completion happens only after all steps succeed.\n"
                "- When using API calls, match parameter names exactly as shown in the provided API docs.\n\n"
                "- REUSE existing variables from the VARIABLES section below — do NOT re-login or re-fetch values that are already available.\n"
                "Before responding, verify:\n"
                "- Output matches the exact format.\n"
                "- PATCH exists and is a single ```python fenced block.\n"
                "- PATCH includes at least one apis.* call.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task id: {task_id}\n\n"
                + (f"{state_context}\n\n" if state_context else "")
                + "The following code was executed and failed.\n\n"
                "Code:\n"
                f"```python\n{code}\n```\n\n"
                "Execution output:\n"
                f"```text\n{world_out}\n```\n\n"
                + (
                    "Relevant API documentation (authoritative):\n"
                    f"```json\n{api_docs_text}\n```\n\n"
                    if api_docs_text
                    else ""
                )
                + "IMPORTANT: Any fixes must match the API docs exactly "
                "(argument names, types, and required fields).\n"
            ),
        },
    ]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1000,
        "stream": False,
        **({"user": task_id} if task_id else {}),
    }

    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer EMPTY"},
        )

    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"].get("content", "")


async def post_process_assistant_message(
    assistant_message: Dict[str, Any],
    task_id: Optional[str],
    history: Sequence[Dict[str, Any]],
    request_id: str,
) -> Dict[str, Any]:
    """
    Rebuild AppWorld state by replaying executable code extracted from prior messages,
    then execute code from the newest assistant message.
    """
    if not task_id:
        log_event(
            "intervention.skipped_no_task_id",
            request_id=request_id,
        )
        return assistant_message

    content = assistant_message.get("content") or ""
    experiment_name = _rand_experiment_name(task_id)

    log_event(
        "intervention.start",
        request_id=request_id,
        task_id=task_id,
        experiment_name=experiment_name,
        assistant_content_preview=content[:300],
    )

    world_out: Optional[str] = None
    new_code: Optional[str] = None
    curated: Optional[str] = None

    try:
        with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
            # Registry for tracking key vars
            registry = StateRegistry()

            # Replay history
            for m in history:
                if m.get("role") != "assistant":
                    continue
                msg_content = m.get("content") or ""
                code, _text = code_extractor(msg_content)
                if code:
                    out = world.execute(code)
                    if ENABLE_STATE_REGISTRY and not looks_like_error(out):
                        state_out = world.execute(CAPTURE_SUFFIX)
                        registry.update_from_replay_output(state_out)

            log_event(
                "intervention.registry_state_captured",
                request_id=request_id,
                task_id=task_id,
                num_bindings=len(registry.bindings),
                binding_keys=list(registry.bindings.keys()),
            )

            new_code, _new_text = code_extractor(content)

            # --- NO-CODE LOOP BREAKER ---
            if not new_code:
                consecutive = _count_consecutive_no_code_assistant_msgs(history)

                if consecutive >= 2:
                    # Model is stuck in prose loop — inject API discovery
                    api_docs_text = _build_api_docs_context(world, new_code=None)

                    curated = await get_fix_suggestion_from_vllm(
                        task_id=task_id,
                        code="# (model produced no executable code)",
                        world_out=(
                            f"The agent has produced {consecutive + 1} consecutive "
                            f"messages with no code. Last message excerpt:\n"
                            f"{content[:500]}"
                        ),
                        api_docs_text=api_docs_text,
                        state_context=registry.format_for_prompt(),
                    )

                    patch_code = _extract_patch_from_curated(curated or "")
                    if patch_code:
                        assistant_message["content"] = f"```python\n{patch_code}\n```"
                    else:
                        # Deterministic fallback: just discover APIs
                        assistant_message["content"] = (
                            "```python\n"
                            "print(apis.api_docs.show_app_descriptions())\n"
                            "```"
                        )

                    log_event(
                        "intervention.no_code_loop_break",
                        request_id=request_id,
                        task_id=task_id,
                        consecutive_no_code=consecutive + 1,
                    )
                    return assistant_message

                # First time no code — let it pass, might be planning
                return assistant_message

            # --- NORMAL PATH: code was extracted, execute it ---
            world_out = world.execute(new_code)

            if world_out is not None and looks_like_error(world_out):
                api_docs_text = _build_api_docs_context(world, new_code, world_out)

                curated = await get_fix_suggestion_from_vllm(
                    task_id=task_id,
                    code=new_code,
                    world_out=world_out,
                    api_docs_text=api_docs_text,
                    state_context=registry.format_for_prompt(),
                )

                # Extract just the patch, not the full diagnostic
                patch_code = _extract_patch_from_curated(curated or "")
                if patch_code:
                    assistant_message["content"] = f"```python\n{patch_code}\n```"
                else:
                    # Fix model failed too — fall back to doc discovery for the apps involved
                    # calls = extract_api_calls(new_code)
                    # if calls:
                    #     app_name = calls[0][0]
                    #     assistant_message["content"] = (
                    #         f"```python\n"
                    #         f"print(apis.api_docs.show_api_descriptions(app_name='{app_name}'))\n"
                    #         f"```"
                    #     )
                    # else:
                    #     assistant_message["content"] = content
                    # Fix model failed too — fall back to doc discovery for the failing app
                    failed_app = _extract_failed_app_from_error(world_out, new_code)
                    if failed_app:
                        assistant_message["content"] = (
                            f"```python\n"
                            f"print(apis.api_docs.show_api_descriptions(app_name='{failed_app}'))\n"
                            f"```"
                        )
                    else:
                        # Can't determine failing app, show all
                        assistant_message["content"] = (
                            "```python\n"
                            "print(apis.api_docs.show_app_descriptions())\n"
                            "```"
                        )

                log_event(
                    "intervention.curated",
                    request_id=request_id,
                    task_id=task_id,
                    curated_preview=(curated or "")[:800],
                    patch_extracted=bool(patch_code),
                )
            else:
                assistant_message["content"] = content

    except Exception as e:
        # If replay/execution itself crashed outside AppWorld's string errors
        log_event(
            "intervention.exception",
            request_id=request_id,
            task_id=task_id,
            error=str(e),
        )
        # Let it fall through: we’ll just return original assistant message
        raise Exception(
            f"[post_process_assistant_message] Error during World execution: {e}"
        )

    log_event(
        "intervention.execution_result",
        request_id=request_id,
        task_id=task_id,
        looks_like_error=looks_like_error(world_out),
        raw_world_out=(world_out or "")[:4000],
        executed_code_preview=(new_code or "")[:800],
    )

    log_event(
        "intervention.final_response",
        request_id=request_id,
        task_id=task_id,
        returned_content_preview=(assistant_message.get("content") or "")[:800],
        was_curated=bool(curated),
    )

    return assistant_message


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    request_id = uuid.uuid4().hex[:12]

    if req.stream:
        raise HTTPException(
            status_code=400, detail="Streaming not supported by this proxy yet."
        )

    task_id = extract_task_id(req)
    payload = build_vllm_payload(req)
    payload = await maybe_summarize_payload(payload=payload, task_id=task_id)

    # Log incoming request (minimal but useful)
    log_event(
        "request.incoming",
        request_id=request_id,
        task_id=task_id,
        model=req.model,
        n_messages=len(req.messages),
        client_host=getattr(request.client, "host", None),
        last_user_preview=(
            req.messages[-1].content[:300]
            if req.messages and req.messages[-1].content
            else None
        ),
    )

    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            VLLM_CHAT_URL, json=payload, headers={"Authorization": "Bearer EMPTY"}
        )

    log_event(
        "vllm.raw_response",
        request_id=request_id,
        task_id=task_id,
        status_code=r.status_code,
        raw_preview=r.text[:4000],  # avoid blowing up logs
    )

    if r.status_code >= 400:
        log_event(
            "request.vllm_error",
            request_id=request_id,
            task_id=task_id,
            status_code=r.status_code,
            body_preview=r.text[:2000],
        )

        # IMPORTANT: propagate 4xx as 4xx
        if 400 <= r.status_code < 500:
            raise HTTPException(
                status_code=400,
                detail=f"vLLM rejected request: {r.text}",
            )

        # Only treat 5xx as Bad Gateway
        raise HTTPException(
            status_code=400,
            detail=f"vLLM internal error {r.status_code}: {r.text}",
        )

    vllm_resp = r.json()

    msg = vllm_resp["choices"][0].get("message")
    if not isinstance(msg, dict):
        log_event(
            "response.bad_vllm_format",
            request_id=request_id,
            task_id=task_id,
            body_preview=str(vllm_resp)[:2000],
        )
        raise HTTPException(status_code=502, detail="vLLM response missing message")

    history = [m.model_dump(exclude_none=True) for m in req.messages]

    vllm_resp["choices"][0]["message"] = await post_process_assistant_message(
        assistant_message=msg,
        task_id=task_id,
        history=history,
        request_id=request_id,
    )

    return vllm_resp


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": []}
