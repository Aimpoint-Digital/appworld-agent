import os
import uuid
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001")
VLLM_CHAT_URL = f"{VLLM_BASE_URL}/v1/chat/completions"

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None

    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    stream: Optional[bool] = False

    # should be the task id:
    user: Optional[str] = None


def to_openai_messages(req: ChatCompletionRequest) -> List[Dict[str, Any]]:
    return [m.model_dump(exclude_none=True) for m in req.messages]


def extract_task_id(req: ChatCompletionRequest) -> Optional[str]:
    if req.user:
        return req.user
    if req.appworld:
        return req.appworld.task_id
    return None


def build_vllm_payload(req: ChatCompletionRequest) -> Dict[str, Any]:
    max_tokens = req.max_tokens if req.max_tokens is not None else req.max_completion_tokens

    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": to_openai_messages(req),
        "temperature": req.temperature,
        # Only include if not None (vLLM/OpenAI both accept max_tokens)
        **({ "max_tokens": max_tokens } if max_tokens is not None else {}),
        **({ "top_p": req.top_p } if req.top_p is not None else {}),
        **({ "stop": req.stop } if req.stop is not None else {}),
        **({ "tools": req.tools } if req.tools is not None else {}),
        **({ "tool_choice": req.tool_choice } if req.tool_choice is not None else {}),
        **({ "parallel_tool_calls": req.parallel_tool_calls } if req.parallel_tool_calls is not None else {}),
        "stream": False,  # keep it simple at first
    }
    # IMPORTANT: do not forward req.appworld; unknown fields can break downstream
    # You can forward req.user if you want vLLM logs to include it:
    if req.user is not None:
        payload["user"] = req.user
    return payload


def post_process_assistant_message(
    assistant_message: Dict[str, Any],
    task_id: Optional[str],
) -> Dict[str, Any]:
    """
    Your hook: take vLLM assistant message and rewrite it if desired.
    Keep tool_calls if present. Keep role.
    """
    content = assistant_message.get("content") or ""

    # Example hooks (plug in your real ones):
    # code_snippet = code_extractor(assistant_message)
    # content = agent_repl(code_snippet, task_id)  # or modify content

    assistant_message["content"] = content
    return assistant_message


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported by this proxy yet.")

    task_id = extract_task_id(req)
    payload = build_vllm_payload(req) # in here we can intelligent summarize if we want

    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(VLLM_CHAT_URL, json=payload, headers={"Authorization": "Bearer EMPTY"})
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"vLLM error {r.status_code}: {r.text}")

    vllm_resp = r.json()

    # Defensive shape checks
    if "choices" not in vllm_resp or not vllm_resp["choices"]:
        raise HTTPException(status_code=502, detail="vLLM response missing choices")

    # Patch assistant message (choice 0)
    msg = vllm_resp["choices"][0].get("message")
    if not isinstance(msg, dict):
        raise HTTPException(status_code=502, detail="vLLM response missing message")

    vllm_resp["choices"][0]["message"] = post_process_assistant_message(msg, task_id) # this can call repl env

    # Optionally attach metadata for your own client (AppWorld won’t care)
    # vllm_resp["_appworld_task_id"] = task_id

    return vllm_resp


@app.get("/v1/models")
async def models():
    # minimal stub; you can proxy vLLM's /v1/models if desired
    return {"object": "list", "data": []}
