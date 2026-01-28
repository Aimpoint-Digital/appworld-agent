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

ENABLE_CONTEXT_SUMMARY = os.getenv("ENABLE_CONTEXT_SUMMARY", "1") == "1"
SUMMARY_CHAR_THRESHOLD = int(os.getenv("SUMMARY_CHAR_THRESHOLD", "24000"))  # raw chars across all message content
SUMMARY_TOKEN_THRESHOLD = int(os.getenv("SUMMARY_TOKEN_THRESHOLD", "6000"))  # approximate tokens
KEEP_LAST_K = int(os.getenv("SUMMARY_KEEP_LAST_K", "6"))  # keep last K messages verbatim
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "Qwen/Qwen3-8B")  
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUMMARY_VLLM_BASE_URL = os.getenv("SUMMARY_VLLM_BASE_URL", os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001"))
SUMMARY_VLLM_CHAT_URL = f"{SUMMARY_VLLM_BASE_URL}/v1/chat/completions"
SUMMARY_VLLM_API_KEY = os.getenv("SUMMARY_VLLM_API_KEY", "")  # optional; usually EMPTY in AppWorld proxy
VLLM_CONTEXT_LEN = int(os.getenv("VLLM_CONTEXT_LEN", "12000"))
MIN_COMPLETION_TOKENS = int(os.getenv("MIN_COMPLETION_TOKENS", "256"))
DEFAULT_COMPLETION_TOKENS = int(os.getenv("DEFAULT_COMPLETION_TOKENS", "1024"))


def approx_prompt_tokens_from_messages(messages: List[Dict[str, Any]]) -> int:
    # reuse your existing approximation
    total_chars, approx_toks = payload_size(messages)
    return approx_toks

def clamp_max_tokens_for_vllm(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures payload['max_tokens'] fits into context window given prompt size.
    Works for vLLM OpenAI-compatible /chat/completions.
    """
    messages = payload.get("messages") or []
    prompt_toks = approx_prompt_tokens_from_messages(messages)
    available = max(0, VLLM_CONTEXT_LEN - prompt_toks)

    # determine requested completion length
    requested = payload.get("max_tokens")
    if requested is None:
        requested = payload.get("max_completion_tokens")
    if requested is None:
        requested = DEFAULT_COMPLETION_TOKENS

    # clamp
    safe = max(1, min(int(requested), available))

    # if there is basically no room, still set something tiny to avoid vLLM 400
    # (callers can decide to force summarize or reduce tail when safe < MIN)
    payload = dict(payload)
    payload["max_tokens"] = safe
    payload.pop("max_completion_tokens", None)  # optional: avoid ambiguity
    return payload


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

async def summarize_messages_with_vllm(
    *,
    task_id: Optional[str],
    task_instruction: str,
    middle_messages: List[Dict[str, Any]],
    model: str = SUMMARY_MODEL,
) -> str:
    # Format middle messages compactly
    def render_msg(m: Dict[str, Any]) -> str:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if len(content) > 2500:
            content = content[:2500] + "\n...[truncated]..."
        return f"{role.upper()}:\n{content}"

    middle_blob = "\n\n".join(render_msg(m) for m in middle_messages)

    sys = (
        "You are compressing an agent conversation for continued execution in a tool-using benchmark.\n"
        "Produce a concise but action-oriented summary that helps the agent continue correctly.\n"
        "Do NOT invent tool outputs, API calls, credentials, or facts.\n\n"
        "When summarizing, actively look for and explicitly note ANY of the following IF THEY OCCURRED:\n"
        "- Authentication or credential problems (missing tokens, login required, 401/403, expired creds)\n"
        "- Tool/API misuse (wrong API name, missing required call, wrong parameters, schema mismatch)\n"
        "- No-op executions (tool call made but no state change; empty changed_records; task claims success without effects)\n"
        "- Pagination or incomplete iteration issues (only first page fetched, missing cursor/offset handling)\n"
        "- Repeated or looping actions that failed similarly\n\n"
        "If none of the above occurred, say so explicitly.\n"
        "Prefer concrete evidence over interpretation (tool names, error messages, observed outcomes).\n"
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
        "max_tokens": 600,   # summaries should be short
        "stream": False,
        **({"user": task_id} if task_id else {}),
    }

    headers = {
        "Content-Type": "application/json",
        # vLLM typically ignores auth; keep compatible with your proxy style
        "Authorization": f"Bearer {SUMMARY_VLLM_API_KEY or 'EMPTY'}",
    }

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(SUMMARY_VLLM_CHAT_URL, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"].get("content", "")


# NOTE: could work but arguably not "pure" since using different model
# async def summarize_messages_with_openai(
#     *,
#     task_id: Optional[str],
#     task_instruction: str,
#     middle_messages: List[Dict[str, Any]],
#     model: str = SUMMARY_MODEL,
# ) -> str:
#     """
#     Summarize the middle of the conversation for the agent, with emphasis on:
#       - what has been attempted
#       - what worked/failed
#       - recurring failure modes
#       - important intermediate state / constraints / credentials discovered
#     """
#     if not OPENAI_API_KEY:
#         raise RuntimeError("OPENAI_API_KEY is not set, but summarization was requested.")

#     # Format middle messages compactly
#     def render_msg(m: Dict[str, Any]) -> str:
#         role = m.get("role", "unknown")
#         content = m.get("content", "")
#         if not isinstance(content, str):
#             content = str(content)
#         # clip per-message to avoid huge requests; the point is compression anyway
#         if len(content) > 2500:
#             content = content[:2500] + "\n...[truncated]..."
#         return f"{role.upper()}:\n{content}"

#     middle_blob = "\n\n".join(render_msg(m) for m in middle_messages)

#     # targets main failure modes
#     sys = (
#         "You are compressing an agent conversation for continued execution in a tool-using benchmark.\n"
#         "Produce a concise but action-oriented summary that helps the agent continue correctly.\n"
#         "Do NOT invent tool outputs, API calls, credentials, or facts.\n\n"

#         "When summarizing, actively look for and explicitly note ANY of the following IF THEY OCCURRED:\n"
#         "- Authentication or credential problems (missing tokens, login required, 401/403, expired creds)\n"
#         "- Tool/API misuse (wrong API name, missing required call, wrong parameters, schema mismatch)\n"
#         "- No-op executions (tool call made but no state change; empty changed_records; task claims success without effects)\n"
#         "- Pagination or incomplete iteration issues (only first page fetched, missing cursor/offset handling)\n"
#         "- Repeated or looping actions that failed similarly\n\n"

#         "If none of the above occurred, say so explicitly.\n"
#         "Prefer concrete evidence over interpretation (e.g., mention tool names, error messages, or observed outcomes).\n"
#     )

#     user = (
#         f"Task ID: {task_id or 'unknown'}\n\n"
#         f"Original task instruction (keep exact intent):\n{task_instruction}\n\n"
#         "Conversation segment to summarize (messages 2..N-1):\n"
#         f"{middle_blob}\n\n"
#         "Return ONLY the summary, in this structured format:\n"
#         "1) Progress so far (3-6 bullets)\n"
#         "2) Key facts / state discovered (bullets)\n"
#         "3) Failure modes / repeated mistakes (bullets)\n"
#         "4) Next best actions (bullets)\n"
#     )

#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": sys},
#             {"role": "user", "content": user},
#         ],
#         "temperature": 0.0,
#     }

#     headers = {
#         "Authorization": f"Bearer {OPENAI_API_KEY}",
#         "Content-Type": "application/json",
#     }

#     timeout = httpx.Timeout(60.0, connect=10.0)
#     async with httpx.AsyncClient(timeout=timeout) as client:
#         r = await client.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=headers)
#         r.raise_for_status()
#         data = r.json()
#         return data["choices"][0]["message"]["content"]

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
        return clamp_max_tokens_for_vllm(payload)

    messages = payload.get("messages") or []
    if not isinstance(messages, list) or len(messages) < 6: # could probably have this as an env setting
        return clamp_max_tokens_for_vllm(payload)

    total_chars, approx_toks = payload_size(messages)
    if not force and total_chars < SUMMARY_CHAR_THRESHOLD and approx_toks < SUMMARY_TOKEN_THRESHOLD:
        return clamp_max_tokens_for_vllm(payload)

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

    # summary_text = await summarize_messages_with_openai(
    #     task_id=task_id,
    #     task_instruction=system_msg + task_instruction,
    #     middle_messages=middle,
    # )
    system_text = ""
    if system_msg and isinstance(system_msg.get("content"), str):
        system_text = system_msg["content"]

    summary_text = await summarize_messages_with_vllm(
        task_id=task_id,
        task_instruction=(system_text + "\n\n" + task_instruction).strip(),
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

    new_payload = dict(payload)
    new_payload["messages"] = new_messages

    # Now clamp based on the NEW messages
    new_payload = clamp_max_tokens_for_vllm(new_payload)
    return new_payload
