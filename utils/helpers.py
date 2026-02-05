import re
from typing import List, Tuple, Optional

def code_extractor(text: str, ignore_multiple_calls: bool = True) -> tuple[str, str]:
    """
    Exact code from appworld repo, taken from SimplifiedReActCodeAgent at
    appworld/experiments/code/simplified/react_code_agent.py
    """
    original_text = text
    output_code = ""
    match_end = 0
    full_code_regex = r"```python\n(.*?)```"
    partial_code_regex = r".*```python\n(.*)"
    # Handle multiple calls
    for re_match in re.finditer(full_code_regex, original_text, flags=re.DOTALL):
        code = re_match.group(1).strip()
        if ignore_multiple_calls:
            text = original_text[: re_match.end()]
            return code, text
        output_code += code + "\n"
        match_end = re_match.end()
    # check for partial code match at end (no terminating ```)  following the last match
    partial_match = re.match(
        partial_code_regex, original_text[match_end:], flags=re.DOTALL
    )
    if partial_match:
        output_code += partial_match.group(1).strip()
        # terminated due to stop condition. Add stop condition to output.
        if not text.endswith("\n"):
            text = text + "\n"
        text = text + "```"
    if len(output_code) == 0:
        return "", text
    else:
        return output_code, text

API_CALL_RE = re.compile(r"\bapis\.([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\s*\(")

def extract_api_calls(code: str) -> List[Tuple[str, str]]:
    """Return list of (app_name, api_name) calls found in code, de-duped preserving order."""
    seen = set()
    out = []
    for app, api in API_CALL_RE.findall(code or ""):
        key = (app, api)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks. Handle unclosed tags (context overflow)."""
    # Closed think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Unclosed think tag (model ran out of tokens mid-reasoning)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def _extract_patch_from_curated(curated: str) -> Optional[str]:
    """Pull just the executable code from the fix model's structured output."""
    curated = _strip_think_tags(curated)
    
    # Try PATCH section first
    match = re.search(r'PATCH:\s*```python\s*(.+?)```', curated, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: any python code block
    match = re.search(r'```python\s*(.+?)```', curated, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None

def _count_consecutive_no_code_assistant_msgs(history: Sequence[Dict[str, Any]]) -> int:
    """Count how many recent assistant messages in a row have no code blocks. helps prevent no code rep loops"""
    count = 0
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        msg_content = m.get("content") or ""
        code, _ = code_extractor(msg_content)
        if code:
            break
        count += 1
    return count

def _build_api_docs_context(world, new_code: Optional[str]) -> str:
    """Three-tier doc enrichment."""
    chunks = []
    
    # Tier 1: always include task-level app descriptions
    try:
        task_apps = world.task.app_descriptions
        chunks.append(
            "Available apps for this task:\n"
            + json.dumps(task_apps, indent=2)[:2000]
        )
    except Exception:
        pass

    if not new_code:
        return "\n\n".join(chunks)

    calls = extract_api_calls(new_code)
    if not calls:
        return "\n\n".join(chunks)

    # Tier 2 + 3: per-app methods, per-method params
    seen_apps = set()
    for app_name, api_name in calls[:5]:
        method_names = []
        
        if app_name not in seen_apps:
            seen_apps.add(app_name)
            try:
                methods = world.apis.api_docs.show_api_descriptions(app_name=app_name)
                method_names = [m['name'] for m in methods]
                chunks.append(
                    f"Available methods for '{app_name}':\n"
                    + json.dumps(methods, indent=2)[:3000]
                )
            except Exception as e:
                chunks.append(f"App '{app_name}': <not found: {e}>")

        # Tier 3: specific method params (only if it actually exists)
        if api_name in method_names:
            try:
                doc = world.apis.api_docs.show_api_doc(
                    app_name=app_name, api_name=api_name
                )
                chunks.append(
                    f"apis.{app_name}.{api_name} spec:\n"
                    + json.dumps(doc, indent=2)[:4000]
                )
            except Exception as e:
                chunks.append(f"apis.{app_name}.{api_name} spec: <error: {e}>")
        else:
            chunks.append(
                f"apis.{app_name}.{api_name}: METHOD DOES NOT EXIST. "
                f"See available methods for '{app_name}' above."
            )

    return "\n\n".join(chunks)