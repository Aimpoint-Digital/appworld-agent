
def code_extractor(text: str) -> tuple[str, str]:
    """
    Exact code from appworld repo, taken from SimplifiedReActCodeAgent at
    appworld/experiments/code/simplified/react_code_agent.py
    """
    original_text = text
    output_code = ""
    match_end = 0
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