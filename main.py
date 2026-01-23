import os
import uuid
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

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
    """
    appworld may pass through some fields i dont need to edit here, so just pass through
    """
    model_config = ConfigDict(extra="allow")

    # standard fields we do need to track
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    user: Optional[str] = None


def to_openai_messages(req: ChatCompletionRequest) -> List[Dict[str, Any]]:
    return [m.model_dump(exclude_none=True) for m in req.messages]


def extract_task_id(req: ChatCompletionRequest) -> Optional[str]:
    if req.user:
        return req.user
    return None


def build_vllm_payload(req: ChatCompletionRequest) -> Dict[str, Any]:
    data = req.model_dump(exclude_none=True)
    return data


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

    # replace in place
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

    # Patch assistant message (choice 0)
    msg = vllm_resp["choices"][0].get("message")
    if not isinstance(msg, dict):
        raise HTTPException(status_code=502, detail="vLLM response missing message")

    vllm_resp["choices"][0]["message"] = post_process_assistant_message(msg, task_id) # this can call repl env

    return vllm_resp


@app.get("/v1/models")
async def models():
    """
    mainly a dummy route, probably dont need
    """
    return {"object": "list", "data": []}
