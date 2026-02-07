import json
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger("appworld_proxy")

STATE_MARKER = "__PROXY_STATE__"

# Runs inside AppWorld's persistent namespace after each successful step.
# Captures all user-defined scalars and small compound types.
CAPTURE_SUFFIX = """
import json as _json
_proxy_state = {}
for _k, _v in dict(locals()).items():
    if _k.startswith('_') or _k == 'apis':
        continue
    if isinstance(_v, (int, float, str, bool)):
        _s = repr(_v)
        if len(_s) <= 300:
            _proxy_state[_k] = _s
    elif isinstance(_v, dict):
        _s = repr(_v)
        if len(_s) <= 500:
            _proxy_state[_k] = _s
        else:
            _proxy_state[_k] = str({k2: repr(v2)[:80] for k2, v2 in list(_v.items())[:10]}) + '...'
    elif isinstance(_v, list):
        if len(_v) <= 5:
            _s = repr(_v)
            if len(_s) <= 500:
                _proxy_state[_k] = _s
        else:
            _proxy_state[_k] = f'list of {len(_v)} items, first={repr(_v[0])[:100]}'
print("__PROXY_STATE__" + _json.dumps(_proxy_state))
"""


class StateRegistry:
    """
    Lightweight execution-state tracker.

    Captures the agent's variable namespace during history replay
    (which the proxy already does). No extra LLM calls, no prompt
    changes to the agent — just piggybacks on the existing replay loop.

    The correction node receives this as flat context so it can
    reference actual variable names and values (tokens, IDs, etc.)
    without needing conversation history.
    """

    def __init__(self):
        self.bindings: Dict[str, str] = {}

    def update_from_replay_output(self, output: Optional[str]):
        """Parse state dump from CAPTURE_SUFFIX output."""
        if not output:
            return
        for line in output.splitlines():
            if line.startswith(STATE_MARKER):
                try:
                    raw = json.loads(line[len(STATE_MARKER):])
                    if isinstance(raw, dict):
                        for k, v in raw.items():
                            self.bindings[k] = str(v)[:300]
                except json.JSONDecodeError:
                    logger.warning("StateRegistry: failed to parse state dump")

    def format_for_prompt(self) -> str:
        """Compact representation for injection into correction node prompt."""
        if not self.bindings:
            return ""
        lines = ["VARIABLES FROM PRIOR SUCCESSFUL STEPS:"]
        for k, v in self.bindings.items():
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)
