import os
import math
from typing import Any, Dict, List, Optional, Tuple
import httpx
from dotenv import load_dotenv, find_dotenv
import re
from appworld import AppWorld
from fastapi import HTTPException

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
KEEP_FIRST_N = int(os.getenv("SUMMARY_KEEP_FIRST_N", "4"))



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

_VLLM_CTX_RE = re.compile(
    r"maximum context length is (\d+) tokens and your request has (\d+) input tokens.*?max_tokens.*?:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)

def build_vllm_payload(req) -> Dict[str, Any]:
    """
    Keep this simple. Summarization should happen in an async step before the POST.
    """
    return req.model_dump(exclude_none=True)

def _parse_vllm_ctx_error(text: str) -> Optional[Tuple[int, int, int]]:
    if not text:
        return None
    m = _VLLM_CTX_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def _render_msg(m: Dict[str, Any], clip: int) -> str:
    role = m.get("role", "unknown")
    content = m.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    if clip is not None and len(content) > clip:
        content = content[:clip] + "\n...[truncated]..."
    return f"{role.upper()}:\n{content}"

async def summarize_messages_with_vllm(
    *,
    middle_messages: List[Dict[str, Any]],
    model: str = SUMMARY_MODEL,
) -> str:
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
        "- If the assistant has completed the task, but has not explicitly called the completion API, return the codeblock to call that API, like   ```python \n apis.supervisor.complete_task(answer=23)\n\n"
        "If none of the above occurred, say so explicitly.\n"
        "Prefer concrete evidence over interpretation (tool names, error messages, observed outcomes).\n"
        "IMPORTANT: ENSURE THAT PREVIOUSLY EXTRACTED RESULTS SUCH AS ACCESS TOKENS, CREDENTIALS, API OUTPUTS, ETC, ARE RETURNED VERBATIM AS PART OF YOUR RESPONSE.\n"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SUMMARY_VLLM_API_KEY or 'EMPTY'}",
    }

    clip_sizes = [2500, 1200, 600, 300, 150]
    msg_caps = [60, 30, 15, 8, 4, 2]
    max_tokens_try = [600, 300, 150, 80, 40]

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for clip in clip_sizes:
            for cap in msg_caps:
                mids = middle_messages[-cap:] if len(middle_messages) > cap else middle_messages
                middle_blob = "\n\n".join(_render_msg(m, clip) for m in mids)

                user = (
                    "Conversation segment to summarize:\n"
                    f"{middle_blob}\n"
                )

                for mt in max_tokens_try:
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": sys},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.0,
                        "max_tokens": mt,
                        "stream": False,
                    }

                    r = await client.post(SUMMARY_VLLM_CHAT_URL, json=payload, headers=headers)

                    if r.status_code == 200:
                        data = r.json()
                        return data["choices"][0]["message"].get("content", "")

                    # Only handle context-length 400s; otherwise bail out
                    if r.status_code != 400:
                        break

                    ctx = _parse_vllm_ctx_error(r.text)
                    if not ctx:
                        break  # some other 400

                    max_ctx, input_toks, _req = ctx
                    if (max_ctx - input_toks) <= 256:
                        break  # shrink prompt further (clip/cap) rather than only lowering max_tokens

        raise HTTPException(
            status_code=400,
            detail=(
                "Context overflow during summarization: "
                "prompt cannot be reduced enough to fit the model context window."
            ),
        )


async def maybe_summarize_payload(
    *,
    payload: Dict[str, Any],
    task_id: Optional[str],
    force: bool = False,
) -> Dict[str, Any]:
    if not ENABLE_CONTEXT_SUMMARY and not force:
        return clamp_max_tokens_for_vllm(payload)

    messages = payload.get("messages") or []
    if not isinstance(messages, list) or len(messages) < 6:
        return clamp_max_tokens_for_vllm(payload)

    total_chars, approx_toks = payload_size(messages)
    if not force and total_chars < SUMMARY_CHAR_THRESHOLD and approx_toks < SUMMARY_TOKEN_THRESHOLD:
        return clamp_max_tokens_for_vllm(payload)

    n = max(1, KEEP_FIRST_N)
    k = max(1, KEEP_LAST_K)

    # Optional: "pin" a system message at the front even if N doesn't include it.
    pinned_system: Optional[Dict[str, Any]] = None
    if messages and messages[0].get("role") == "system": # this wont exist for react tempalte in appworld
        pinned_system = messages[0]

    # Build first/last windows (excluding pinned system from window math if you want)
    start_idx = 1 if pinned_system else 0
    core = messages[start_idx:]  # messages without pinned system

    if len(core) <= (n + k):
        # Not enough to justify a summary; just clamp and return
        return clamp_max_tokens_for_vllm(payload)

    first = core[:n]
    last = core[-k:]

    middle = core[n:len(core) - k]
    if len(middle) < 2:
        return clamp_max_tokens_for_vllm(payload)

    summary_text = await summarize_messages_with_vllm(
        middle_messages=middle
    )

    new_messages: List[Dict[str, Any]] = []
    if pinned_system is not None:
        new_messages.append(pinned_system)

    # Keep first N verbatim
    new_messages.extend(first)

    # Insert summary
    new_messages.append(
        {
            "role": "assistant",
            "content": (
                "CONTEXT SUMMARY (auto-generated). This replaces earlier conversation details.\n\n"
                f"{summary_text}"
            ),
        }
    )

    # Keep last K verbatim
    new_messages.extend(last)

    new_payload = dict(payload)
    new_payload["messages"] = new_messages
    return clamp_max_tokens_for_vllm(new_payload)
