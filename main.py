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
from helpers import code_extractor, extract_api_calls

load_dotenv(find_dotenv())

# -------------------------
# Logging setup
# -------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_PATH = os.getenv("PROXY_LOG_PATH", "proxy_interventions.log")

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
    logger.info(json.dumps(payload, ensure_ascii=False))


# -------------------------
# ENV settings, configs
# -------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001")
VLLM_CHAT_URL = f"{VLLM_BASE_URL}/v1/chat/completions"

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


async def get_fix_suggestion_from_vllm(
    task_id: Optional[str],
    code: str,
    world_out: str,
    api_docs_text: str = "",
    model: str = "Qwen/Qwen3-8B",
) -> str:
    messages = [
        {
          "role": "system",
          "content": (
              "You are debugging Python code executed inside AppWorld.\n"
              "Your job is to produce a SHORT, ACTIONABLE intervention message that the agent can use immediately.\n"
              "Do NOT ask questions. Do NOT use input(). Do NOT invent tool outputs or facts.\n"
              "Prefer concrete edits to the code and specific API/tool call corrections.\n\n"
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
              "- <2-6 bullet steps, concrete>\n"
              "PATCH: <optional; include a corrected code snippet if it's small, else omit>\n"
          ),
        }
        {
            "role": "user",
            "content": (
                f"Task id: {task_id}\n\n"
                "The following code was executed and failed.\n\n"
                "Code:\n"
                f"```python\n{code}\n```\n\n"
                "Execution output:\n"
                f"```text\n{world_out}\n```\n\n"
                + (
                    "Relevant API documentation (authoritative):\n"
                    f"```json\n{api_docs_text}\n```\n\n"
                    if api_docs_text else ""
                )
                + "IMPORTANT: Any fixes must match the API docs exactly "
                  "(argument names, types, and required fields).\n"
            ),
        }
    ]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 400,
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

    # before testing the new code, we need to recreate DB state
    try:
        with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
            # Replay: all assistant messages from history (excluding latest; history includes user+assistant)
            for m in history:
                if m.get("role") != "assistant":
                    continue
                msg_content = m.get("content") or ""
                code, _text = code_extractor(msg_content)  # must exist in your project
                if code:
                    world.execute(code)

            # Execute newest assistant message
            new_code, _new_text = code_extractor(content)
            if new_code:
                world_out = world.execute(new_code)

    except Exception as e:
        # If replay/execution itself crashed outside AppWorld's string errors
        log_event(
            "intervention.exception",
            request_id=request_id,
            task_id=task_id,
            error=str(e),
        )
        # Let it fall through: we’ll just return original assistant message
        raise Exception(f"[post_process_assistant_message] Error during World execution: {e}")

    log_event(
        "intervention.execution_result",
        request_id=request_id,
        task_id=task_id,
        looks_like_error=looks_like_error(world_out),
        raw_world_out=(world_out or "")[:4000],
        executed_code_preview=(new_code or "")[:800],
    )

    curated: Optional[str] = None
    if world_out is not None and looks_like_error(world_out) and new_code:

        # get documentation for the api being called
        api_docs_text = ""
        if new_code:
            calls = extract_api_calls(new_code)
            if calls:
                docs_chunks = []
                for app_name, api_name in calls[:5]:  # cap to avoid huge payloads
                    try:
                        doc = world.apis.api_docs.show_api_doc(app_name=app_name, api_name=api_name)
                        docs_chunks.append(f"apis.{app_name}.{api_name} spec:\n{json.dumps(doc, indent=2)[:4000]}")
                    except Exception as e:
                        docs_chunks.append(f"apis.{app_name}.{api_name} spec: <failed to load: {e}>")
                api_docs_text = "\n\n".join(docs_chunks)

        curated = await get_fix_suggestion_from_vllm(
            task_id=task_id,
            code=new_code,
            world_out=world_out,
            api_docs_text=api_docs_text,
            model="Qwen/Qwen3-8B",
        )
        log_event(
            "intervention.curated",
            request_id=request_id,
            task_id=task_id,
            curated_preview=(curated or "")[:800],
        )

        # If you want to replace assistant content with the curated suggestion:
        assistant_message["content"] = curated
    else:
        # Leave original assistant content intact
        assistant_message["content"] = content

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
        raise HTTPException(status_code=400, detail="Streaming not supported by this proxy yet.")

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
        last_user_preview=(req.messages[-1].content[:300] if req.messages and req.messages[-1].content else None),
    )

    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(VLLM_CHAT_URL, json=payload, headers={"Authorization": "Bearer EMPTY"})

    if r.status_code >= 400:
        log_event(
            "request.vllm_error",
            request_id=request_id,
            task_id=task_id,
            status_code=r.status_code,
            body_preview=r.text[:2000],
        )
        raise HTTPException(status_code=502, detail=f"vLLM error {r.status_code}: {r.text}")

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
