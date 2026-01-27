import os
import math
from typing import Any, Dict, List, Optional, Tuple
import httpx
from dotenv import load_dotenv, find_dotenv

from appworld import AppWorld

load_dotenv(find_dotenv())
# -------------------------
# Summarization settings
# -------------------------

ENABLE_CONTEXT_SUMMARY = os.getenv("ENABLE_CONTEXT_SUMMARY", "0") == "1"
SUMMARY_CHAR_THRESHOLD = int(os.getenv("SUMMARY_CHAR_THRESHOLD", "24000"))  # raw chars across all message content
SUMMARY_TOKEN_THRESHOLD = int(os.getenv("SUMMARY_TOKEN_THRESHOLD", "6000"))  # approximate tokens
KEEP_LAST_K = int(os.getenv("SUMMARY_KEEP_LAST_K", "6"))  # keep last K messages verbatim
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o")  # or gpt-4o-mini, etc.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def approx_tokens_from_text(s: str) -> int:
    """
    Cheap token approximation. Typically ~4 chars/token in English-ish text,
    but code can skew. Good enough for gating.
    """
    if not s:
        return 0
    return math.ceil(len(s) / 4)

def payload_size(messages: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return (total_chars, approx_tokens) for message contents."""
    total_chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
    return total_chars, approx_tokens_from_text("".join([(m.get("content") or "") for m in messages if isinstance(m.get("content"), str)]))

def find_system_and_first_user(messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """
    Returns:
      - system_message (if exists)
      - index of the first 'user' message (task instruction) if exists
    """
    system_msg = None
    first_user_idx = None

    for i, m in enumerate(messages):
        if system_msg is None and m.get("role") == "system":
            system_msg = m
        if first_user_idx is None and m.get("role") == "user":
            first_user_idx = i
        if system_msg is not None and first_user_idx is not None:
            break

    return system_msg, first_user_idx

def build_vllm_payload(req) -> Dict[str, Any]:
    """
    Keep this simple. Summarization should happen in an async step before the POST.
    """
    return req.model_dump(exclude_none=True)

async def summarize_messages_with_openai(
    *,
    task_id: Optional[str],
    task_instruction: str,
    middle_messages: List[Dict[str, Any]],
    model: str = SUMMARY_MODEL,
) -> str:
    """
    Summarize the middle of the conversation for the agent, with emphasis on:
      - what has been attempted
      - what worked/failed
      - recurring failure modes
      - important intermediate state / constraints / credentials discovered
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set, but summarization was requested.")

    # Format middle messages compactly
    def render_msg(m: Dict[str, Any]) -> str:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        # clip per-message to avoid huge requests; the point is compression anyway
        if len(content) > 2500:
            content = content[:2500] + "\n...[truncated]..."
        return f"{role.upper()}:\n{content}"

    middle_blob = "\n\n".join(render_msg(m) for m in middle_messages)

    sys = (
        "You are compressing an agent conversation for continued execution in a tool-using benchmark.\n"
        "Produce a concise but action-oriented summary that helps the agent continue correctly.\n"
        "Do NOT invent tool outputs or facts.\n"
        "Emphasize: attempted actions, outcomes, discovered constraints, and recurring failure modes.\n"
    )

    user = (
        f"Task ID: {task_id or 'unknown'}\n\n"
        f"Original task instruction (keep exact intent):\n{task_instruction}\n\n"
        "Conversation segment to summarize (messages 2..N-1):\n"
        f"{middle_blob}\n\n"
        "Return ONLY the summary, in this structured format:\n"
        "1) Progress so far (3-6 bullets)\n"
        "2) Key facts / state discovered (bullets)\n"
        "3) Failure modes / repeated mistakes (bullets)\n"
        "4) Next best actions (bullets)\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

async def maybe_summarize_payload(
    *,
    payload: Dict[str, Any],
    task_id: Optional[str],
    force: bool = False,
) -> Dict[str, Any]:
    """
    If enabled and the message history is large, replace the middle chunk with a summary message.
    Preserves:
      - system message (if any)
      - the first user instruction (task)
      - last K messages verbatim
    Summarizes:
      - messages between first user instruction and the last K messages
    """
    if not ENABLE_CONTEXT_SUMMARY and not force:
        return payload

    messages = payload.get("messages") or []
    if not isinstance(messages, list) or len(messages) < 6: # could probably have this as an env setting
        return payload

    total_chars, approx_toks = payload_size(messages)
    if not force and total_chars < SUMMARY_CHAR_THRESHOLD and approx_toks < SUMMARY_TOKEN_THRESHOLD:
        return payload

    system_msg, first_user_idx = find_system_and_first_user(messages)
    if first_user_idx is None:
        # no clear task instruction; we can still summarize, but it's riskier
        first_user_idx = 0

    # Decide “tail” to keep
    k = max(2, KEEP_LAST_K)
    tail = messages[-k:]

    # Identify the middle region to summarize:
    # from (first_user_idx+1) up to start of tail (exclusive)
    middle_start = first_user_idx + 1
    middle_end = max(middle_start, len(messages) - k)
    middle = messages[middle_start:middle_end]

    # If there’s nothing meaningful to summarize, skip
    if len(middle) < 2:
        return payload

    task_instruction = messages[first_user_idx].get("content") or ""
    if not isinstance(task_instruction, str):
        task_instruction = str(task_instruction)

    summary_text = await summarize_messages_with_openai(
        task_id=task_id,
        task_instruction=system_msg + task_instruction,
        middle_messages=middle,
    )

    # Build new message list:
    new_messages: List[Dict[str, Any]] = []

    # Keep system message first (exactly one), if present
    if system_msg is not None:
        new_messages.append(system_msg)

    # Keep the task instruction user message verbatim
    new_messages.append(messages[first_user_idx])

    # Insert a synthetic “summary” message
    new_messages.append(
        {
            "role": "assistant",
            "content": (
                "CONTEXT SUMMARY (auto-generated). This replaces earlier conversation details.\n\n"
                f"{summary_text}"
            ),
        }
    )

    # Keep the tail verbatim
    new_messages.extend(tail)

    # Replace payload messages
    new_payload = dict(payload)
    new_payload["messages"] = new_messages
    return new_payload
