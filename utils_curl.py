# utils_curl.py
import shlex
from typing import Dict, Optional


def build_curl_command(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    multiline: bool = True,
) -> str:
    """
    Build a curl command that reuses the same method/headers/body the user entered.

    - Uses POSIX quoting (good for WSL, Linux, macOS).
    - `multiline=True` -> nice multi-line form with backslashes.
    """
    method = method.upper().strip() or "GET"

    cmd = ["curl", "-X", method, shlex.quote(url)]

    headers = headers or {}
    for key, value in headers.items():
        # e.g. -H 'Authorization: Bearer abc'
        header_str = f"{key}: {value}"
        cmd.extend(["-H", shlex.quote(header_str)])

    if body is not None and body.strip():
        cmd.extend(["-d", shlex.quote(body)])

    if not multiline:
        return " ".join(cmd)

    # Multi-line with trailing "\" for all but last token
    lines = []
    for i, token in enumerate(cmd):
        if i == 0:
            lines.append(token + " \\")
        elif i < len(cmd) - 1:
            lines.append("  " + token + " \\")
        else:
            lines.append("  " + token)

    return "\n".join(lines)
