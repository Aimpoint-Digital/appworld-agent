import re
from typing import List, Tuple

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