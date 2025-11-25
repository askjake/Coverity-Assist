# utils_kube.py
import json
import subprocess
from typing import Optional, Tuple, List


def _pick_ingress_host_and_path(ingress_obj: dict) -> Optional[Tuple[str, str]]:
    """
    Given one ingress (from kubectl -o json), choose a (host, path) pair.
    Prefers HTTPS and the first non-'/' path if present.
    """
    spec = ingress_obj.get("spec", {})
    rules: List[dict] = spec.get("rules", [])
    if not rules:
        return None

    # Take the first rule for now (good enough for our use case)
    rule = rules[0]
    host = rule.get("host", "").strip()
    http = rule.get("http", {})
    paths = http.get("paths", []) or []

    # Choose first path; fall back to "/"
    if paths:
        path = paths[0].get("path", "/") or "/"
    else:
        path = "/"

    # Normalize path to something like "" or "/chat"
    if path == "/":
        path = ""
    else:
        if not path.startswith("/"):
            path = "/" + path
        # strip trailing "/" to make joining easier
        if path != "/" and path.endswith("/"):
            path = path[:-1]

    return (host, path)


def kubectl_suggest_base_url(
    namespace: str = "chatbot-dev",
    ingress_name: Optional[str] = None,
    prefer_https: bool = True,
) -> Optional[str]:
    """
    Call `kubectl get ingress -n <namespace> -o json` and return a suggested base URL,
    e.g. "https://my-ingress.example.com/chatbot".

    Returns None if no ingress / namespace not found / error.
    """
    base_cmd = [
        "kubectl",
        "get",
        "ingress",
        "-n",
        namespace,
        "-o",
        "json",
    ]
    if ingress_name:
        base_cmd.insert(3, ingress_name)  # kubectl get ingress <name> -n <ns> -o json

    try:
        out = subprocess.check_output(
            base_cmd,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        # e.output will typically contain "namespaces \"chatbot-dev\" not found" in your current cluster
        print(f"[kubectl_suggest_base_url] kubectl failed: {e.output.strip()}")
        return None
    except FileNotFoundError:
        # kubectl not installed / not on PATH
        print("[kubectl_suggest_base_url] kubectl binary not found on PATH")
        return None

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        print("[kubectl_suggest_base_url] Failed to parse kubectl JSON output")
        return None

    items = data.get("items", [])
    if not items:
        print("[kubectl_suggest_base_url] No ingress resources in that namespace")
        return None

    # If a specific ingress_name was requested, data may not have "items" but be a single object.
    # Handle that too:
    if isinstance(items, list) and items:
        ingress = items[0]
    else:
        ingress = data

    picked = _pick_ingress_host_and_path(ingress)
    if not picked:
        return None

    host, path = picked

    if not host:
        return None

    scheme = "https" if prefer_https else "http"
    base_url = f"{scheme}://{host}{path}"

    # Normalize trailing slash so callers can do base_url + "/some-endpoint"
    if base_url.endswith("/"):
        base_url = base_url[:-1]

    return base_url
