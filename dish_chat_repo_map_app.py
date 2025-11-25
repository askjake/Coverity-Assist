import os
import subprocess
import json
import re
from pathlib import Path
from textwrap import dedent
from urllib.parse import urljoin
from typing import Optional, Dict, Any, List, Tuple

import streamlit as st

from utils_kube import kubectl_suggest_base_url
from utils_curl import build_curl_command

# ---------- Config ----------
REPOS: Dict[str, str] = {
    "dish-chat-fe": "https://gitlab.com/dish-cloud/dt/sse/datasolutions/dish-chat-fe.git",
    "dish-chat": "https://gitlab.com/dish-cloud/dt/sse/datasolutions/dish-chat.git",
    "chat-bot": "https://gitlab.com/dish-cloud/dt/sse/datasolutions/ds-kubernetes-configs/chat-bot.git",
}

ROLE_MAP: Dict[str, Dict[str, str]] = {
    "dish-chat-fe": {
        "src/pages": "Next.js routes (entry pages, error pages)",
        "src/components": "Atomic UI components (atoms → templates, layouts)",
        "src/services": "HTTP clients for each backend domain (chats, messages, vault, etc.)",
        "src/store": "Client-side state slices per domain",
        "src/utils/stream.utils.ts": "Streaming helpers for chat responses",
    },
    "dish-chat": {
        "app/chat": "Chat domain – models, repos, service, router",
        "app/message": "Message domain – per-message operations, streaming endpoints",
        "app/vault": "Vault domain – encrypted secrets & credentials",
        "app/embedding": "Embedding generation & storage",
        "app/agent": "Agents, vectorstore integration, beta-report workflows",
        "app/core/llm": "LLM abstraction layer shared across domains",
        "app/aws": "AWS clients (Bedrock, S3, DynamoDB, etc.) and utilities",
        "app/usage_tracking": "Usage telemetry & analytics domain",
    },
    "chat-bot": {
        "app/base/deployment.yaml": "Core Deployment for chatbot-app in EKS",
        "app/base/service.yaml": "Cluster Service exposing chatbot port 8501",
        "app/base/hpa.yaml": "HorizontalPodAutoscaler for chatbot",
        "app/base/strictmtls.yaml": "Strict mTLS / network policy rules",
        "app/overlays/dev": "Dev overlay – DDB tables, S3 buckets, IAM role, SA, namespace",
        "app/overlays/prod": "Prod overlay – production parameters and resources",
    },
}

# Mapping of FE services → BE routes → K8s resources (reference examples).
ENDPOINT_MAP: Dict[str, Dict[str, str]] = {
    # ---- Chat domain ----
    "POST /api/v1/chats": {
        "fe_service": "src/services/chat.service.ts",
        "be_route": "app/chat/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
    "GET /api/v1/chats": {
        "fe_service": "src/services/chat.service.ts",
        "be_route": "app/chat/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
    "GET /api/v1/chats/{chat_id}": {
        "fe_service": "src/services/chat.service.ts",
        "be_route": "app/chat/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
    "PATCH /api/v1/chats/{chat_id}": {
        "fe_service": "src/services/chat.service.ts",
        "be_route": "app/chat/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },

    # ---- Message domain ----
    "POST /api/v1/messages": {
        "fe_service": "src/services/message.service.ts",
        "be_route": "app/message/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
    "GET /api/v1/chats/{chat_id}/messages": {
        "fe_service": "src/services/message.service.ts",
        "be_route": "app/message/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
    "POST /api/v1/messages/{message_id}/retry": {
        "fe_service": "src/services/message.service.ts",
        "be_route": "app/message/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },

    # ---- Vault domain ----
    "POST /api/v1/vault/credentials": {
        "fe_service": "src/services/vault.service.ts",
        "be_route": "app/vault/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
    "GET /api/v1/vault/credentials": {
        "fe_service": "src/services/vault.service.ts",
        "be_route": "app/vault/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
    "GET /api/v1/vault/providers": {
        "fe_service": "src/services/vault.service.ts",
        "be_route": "app/vault/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },

    # ---- Embedding / agent domains ----
    "POST /api/v1/embeddings": {
        "fe_service": "src/services/embedding.service.ts",
        "be_route": "app/embedding/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
    "POST /api/v1/agents/{agent_id}/invoke": {
        "fe_service": "src/services/agent.service.ts",
        "be_route": "app/agent/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },

    # ---- Usage / telemetry ----
    "GET /api/v1/usage/daily": {
        "fe_service": "src/services/usage.service.ts",
        "be_route": "app/usage_tracking/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
    "GET /api/v1/usage/events": {
        "fe_service": "src/services/usage.service.ts",
        "be_route": "app/usage_tracking/router.py",
        "k8s_resource": "app/base/deployment.yaml",
    },

    # ---- Health / meta ----
    "GET /health": {
        "fe_service": "src/services/chat.service.ts",
        "be_route": "app/main.py",
        "k8s_resource": "app/base/deployment.yaml",
    },
}


# ---------- Helpers ----------
def run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    """Run a subprocess and capture output (stdout + stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout
    except FileNotFoundError as e:
        return f"Command not found: {cmd[0]}\n{e}"
    except Exception as e:
        return f"Error running command {' '.join(cmd)}:\n{e}"


def clone_repo(name: str, url: str, base_dir: Path, user: str, token: str) -> str:
    dest = base_dir / name
    if dest.exists():
        return f"[SKIP] {name} already exists at {dest}"
    if not user or not token:
        return f"[ERROR] Missing GitLab username or PAT for {name}"
    auth_url = url.replace("https://", f"https://{user}:{token}@")
    output = run(["git", "clone", auth_url, str(dest)])
    return f"[CLONE] {name}\n{output}"


def build_tree(path: Path, max_depth: int = 3, prefix: str = "") -> str:
    """Return a simple text tree up to max_depth."""
    lines: List[str] = []

    def _walk(p: Path, depth: int, pre: str) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted([e for e in p.iterdir() if not e.name.startswith(".")])
        except FileNotFoundError:
            return
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{pre}{connector}{entry.name}")
            if entry.is_dir():
                child_prefix = pre + ("    " if i == len(entries) - 1 else "│   ")
                _walk(entry, depth + 1, child_prefix)

    _walk(path, 0, prefix)
    return "\n".join(lines)


def graphviz_architecture() -> str:
    """Return a DOT graph connecting FE, BE, and chat-bot."""
    return dedent(
        """
        digraph G {
          rankdir=LR;
          node [shape=box, style=rounded];

          FE   [label="dish-chat-fe\\n(Next.js UI)"];
          BE   [label="dish-chat\\n(Python backend)"];
          CB   [label="chat-bot\\n(K8s/EKS configs)"];

          FE -> BE [label="HTTP / streaming APIs"];
          BE -> CB [label="Deployed via\\nDeploy/Service/HPA\\n+ IRSA"];
        }
        """
    )


def parse_deployment_images(namespace: str, deployment_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse image tags and environment variables from a deployment."""
    output = run(
        ["kubectl", "-n", namespace, "get", "deploy", deployment_name, "-o", "yaml"]
    )
    if not output:
        return None, "kubectl returned no output"

    lower = output.lower()
    if (
        "error from server" in lower
        or "notfound" in lower
        or "the connection to the server" in lower
        or "was refused" in lower
        or "connectex" in lower
        or "no such host" in lower
        or "unable to connect to the server" in lower
    ):
        return None, output.strip()

    try:
        import yaml  # type: ignore
    except ImportError:
        return None, "PyYAML is not installed. Run `pip install pyyaml`"
    try:
        deploy_yaml = yaml.safe_load(output)
    except Exception as e:
        return None, f"Failed to parse deployment YAML: {e}"

    result: Dict[str, Any] = {
        "images": [],
        "env_vars": {},
        "overlay": "unknown",
    }

    containers = (
        deploy_yaml.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    for container in containers:
        image = container.get("image", "")
        result["images"].append(
            {
                "name": container.get("name", ""),
                "image": image,
                "tag": image.split(":")[-1] if ":" in image else "latest",
            }
        )
        env = container.get("env", [])
        for e in env:
            name = e.get("name", "")
            value = e.get("value", "")
            if name:
                result["env_vars"][name] = value

    metadata = deploy_yaml.get("metadata", {})
    labels = metadata.get("labels", {}) or {}

    ns_lower = namespace.lower()
    if "dev" in ns_lower or labels.get("environment") == "dev":
        result["overlay"] = "dev"
    elif "prod" in ns_lower or labels.get("environment") == "prod":
        result["overlay"] = "prod"
    elif "stage" in ns_lower or "staging" in ns_lower:
        result["overlay"] = "staging"

    return result, None


def load_openapi_spec(base_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load dish-chat OpenAPI spec from docs/dish-chat-backendapi.yaml, if present."""
    spec_path = base_dir / "dish-chat" / "docs" / "dish-chat-backendapi.yaml"
    if not spec_path.exists():
        return None, f"Spec file not found at {spec_path}"
    try:
        import yaml  # type: ignore
    except ImportError:
        return None, "PyYAML is not installed. Run `pip install pyyaml` in this environment."
    try:
        with spec_path.open("r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        return spec, ""
    except Exception as e:
        return None, f"Failed to load OpenAPI spec: {e}"


def build_endpoint_index(spec: Dict[str, Any]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """Flatten OpenAPI paths into a list of (label, path, method, meta)."""
    endpoints: List[Tuple[str, str, str, Dict[str, Any]]] = []
    paths = spec.get("paths", {}) or {}
    for path, methods in paths.items():
        for method, meta in methods.items():
            method_upper = method.upper()
            if method_upper not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                continue
            summary = meta.get("summary") or meta.get("operationId") or ""
            label = f"{method_upper} {path}"
            if summary:
                label += f" — {summary}"
            endpoints.append((label, path, method_upper, meta))
    endpoints.sort(key=lambda x: x[0])
    return endpoints


def extract_example_body(meta: Dict[str, Any]) -> Optional[Any]:
    """Try to pull an example JSON body from requestBody in the OpenAPI spec."""
    request_body = meta.get("requestBody", {}) or {}
    content = request_body.get("content", {}) or {}
    app_json = content.get("application/json", {}) or {}

    if "example" in app_json:
        return app_json["example"]

    examples = app_json.get("examples")
    if isinstance(examples, dict) and examples:
        first = next(iter(examples.values()))
        if isinstance(first, dict) and "value" in first:
            return first["value"]

    schema = app_json.get("schema", {}) or {}
    if "example" in schema:
        return schema["example"]

    return None


def try_requests_request(method: str, url: str, headers: Optional[Dict[str, str]], body_text: str) -> Tuple[Optional[Any], str]:
    """Send an HTTP request using the requests library, if available."""
    try:
        import requests  # type: ignore
    except ImportError:
        return None, "The 'requests' library is not installed. Run `pip install requests` in this environment."

    headers = headers or {}
    data: Optional[str] = None
    json_body: Optional[Any] = None

    if body_text.strip():
        try:
            json_body = json.loads(body_text)
            headers.setdefault("Content-Type", "application/json")
        except json.JSONDecodeError:
            data = body_text

    try:
        resp = requests.request(method, url, headers=headers, json=json_body, data=data, timeout=30)
    except Exception as e:
        return None, f"Request failed: {e}"

    return resp, ""


# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="Dish-Chat Repo Map",
    layout="wide",
)

st.title("Dish-Chat Reference – Repo Map & Structure")
st.caption("DATASO2-6818 · Internal Data Solutions tooling")

# --- Session state defaults ---
if "base_dir" not in st.session_state:
    st.session_state["base_dir"] = os.path.expanduser("~/DATASO2-6818")
if "base_url" not in st.session_state:
    st.session_state["base_url"] = "http://localhost:8000"
if "headers_json" not in st.session_state:
    st.session_state["headers_json"] = "{}"
if "body_text" not in st.session_state:
    st.session_state["body_text"] = ""

base_dir = Path(st.session_state["base_dir"])
base_dir.mkdir(parents=True, exist_ok=True)

# ---------- Sidebar: Git + K8s / Base URL ----------
st.sidebar.header("GitLab Access")
base_dir_input = st.sidebar.text_input(
    "Base directory on disk",
    key="base_dir_input",
    value=str(base_dir),
)
st.session_state["base_dir"] = base_dir_input
base_dir = Path(base_dir_input)
base_dir.mkdir(parents=True, exist_ok=True)

gitlab_user = st.sidebar.text_input("GitLab username", value="jacob.montgomery", key="gitlab_user")
st.sidebar.markdown("**Personal Access Tokens (PATs)** – not stored, only used for this session.")
pat_fe = st.sidebar.text_input("PAT – dish-chat-fe", type="password", key="pat_fe")
pat_be = st.sidebar.text_input("PAT – dish-chat backend", type="password", key="pat_be")
pat_cb = st.sidebar.text_input("PAT – chat-bot configs", type="password", key="pat_cb")

if st.sidebar.button("Clone / Refresh repos", key="btn_clone_repos"):
    st.sidebar.write("Cloning repositories…")
    outputs: List[str] = []
    outputs.append(clone_repo("dish-chat-fe", REPOS["dish-chat-fe"], base_dir, gitlab_user, pat_fe))
    outputs.append(clone_repo("dish-chat", REPOS["dish-chat"], base_dir, gitlab_user, pat_be))
    outputs.append(clone_repo("chat-bot", REPOS["chat-bot"], base_dir, gitlab_user, pat_cb))
    st.sidebar.code("\n\n".join(outputs))

st.sidebar.markdown("---")
st.sidebar.header("Kubernetes / Base URL")

# Global toggle for all kubectl-based features
st.sidebar.checkbox(
    "Enable kubectl-based features",
    value=False,
    key="enable_kubectl",
    help="Uses your local kubeconfig. Disable this if kubectl isn't configured or you don't want it to run.",
)
kubectl_enabled = st.session_state.get("enable_kubectl", False)

ns_default = "chatbot-dev"
namespace = st.sidebar.text_input("Ingress namespace", value=ns_default, key="ingress_ns")
ingress_name = st.sidebar.text_input("Ingress name (optional)", value="", key="ingress_name")

if st.sidebar.button("Detect base URL from ingress", key="btn_detect_ingress"):
    if not kubectl_enabled:
        st.sidebar.warning("Kubectl-based detection is disabled. Enable it above first.")
    else:
        suggested = kubectl_suggest_base_url(namespace=namespace, ingress_name=ingress_name or None)
        if suggested:
            st.session_state["base_url"] = suggested
            st.sidebar.success(f"Detected: {suggested}")
        else:
            st.sidebar.warning(
                "Could not auto-detect. Check namespace / kubectl context / ingress."
            )

if st.sidebar.button("Detect from chatbot-prod", key="btn_detect_prod"):
    if not kubectl_enabled:
        st.sidebar.warning("Kubectl-based detection is disabled. Enable it above first.")
    else:
        suggested = kubectl_suggest_base_url(namespace="chatbot-prod", ingress_name=ingress_name or None)
        if suggested:
            st.session_state["base_url"] = suggested
            st.sidebar.success(f"[chatbot-prod] Detected: {suggested}")
        else:
            st.sidebar.warning(
                "Could not auto-detect for chatbot-prod. Verify the namespace and ingress exist."
            )

st.sidebar.text_input(
    "Backend Base URL",
    key="base_url",
    value=st.session_state["base_url"],
    help="Used by the API explorer & curl generator.",
)

# ---------- High-Level Architecture ----------
st.markdown("### High-Level Architecture")
st.graphviz_chart(graphviz_architecture())

# ---------- Repo structures ----------
st.markdown("---")
cols = st.columns(3)
for col, repo_name in zip(cols, REPOS.keys()):
    repo_path = base_dir / repo_name
    with col:
        st.subheader(repo_name)
        if repo_path.exists():
            st.caption(str(repo_path))
            st.markdown("**Structure (first 3 levels)**")
            st.code(build_tree(repo_path, max_depth=3))

            st.markdown("**Key Directories & Purpose**")
            entries = ROLE_MAP.get(repo_name, {})
            if entries:
                for path, purpose in entries.items():
                    st.markdown(f"- `/{path}` – {purpose}")
            else:
                st.info("No role map defined yet.")
        else:
            st.warning("Repo not cloned yet. Use the sidebar to clone.")

# ---------- Which Overlay is Live? ----------
st.markdown("---")
st.markdown("### 🔍 Which Overlay is Live?")
st.caption(
    "Parse deployment YAML to detect container images, important env vars, and an inferred overlay (dev/prod/staging)."
)

overlay_col1, overlay_col2 = st.columns([2, 1])

with overlay_col1:
    overlay_ns = st.text_input(
        "Namespace for overlay check",
        value="chatbot-dev",
        key="overlay_ns",
    )
    overlay_deploy = st.text_input(
        "Deployment name for overlay check",
        value="chatbot-app",
        key="overlay_deploy",
    )

with overlay_col2:
    if st.button("Detect Live Overlay", key="btn_detect_overlay"):
        if not kubectl_enabled:
            st.error("Kubectl-based features are disabled. Enable them in the sidebar first.")
        else:
            result, error = parse_deployment_images(overlay_ns, overlay_deploy)
            if error:
                st.error(error)
            elif result is None:
                st.warning("No deployment data found.")
            else:
                st.success(f"**Detected Overlay:** `{result['overlay']}`")

                st.markdown("**Container Images:**")
                for img in result["images"]:
                    st.code(f"{img['name']}: {img['image']}")

                if result["env_vars"]:
                    st.markdown("**Key Environment Variables:**")
                    important_vars = ["ENVIRONMENT", "AWS_REGION", "LOG_LEVEL", "BEDROCK_MODEL"]
                    for var in important_vars:
                        if var in result["env_vars"]:
                            st.code(f"{var}={result['env_vars'][var]}")

# ---------- Kubernetes Live Checks ----------
st.markdown("---")
st.markdown("### Kubernetes Live Checks")
st.caption(
    "Runs `kubectl` commands using your current kubeconfig context. "
    "Useful for checking the chatbot deployment in dev/prod clusters."
)

k_ns = st.text_input("Namespace (live checks)", value="chatbot-dev", key="live_ns")
k_app_label = st.text_input("App label selector (key=value)", value="app=chatbot-app", key="live_label")
k_deploy_name = st.text_input("Deployment name", value="chatbot-app", key="live_deploy")

if kubectl_enabled:
    kc1, kc2, kc3 = st.columns(3)

    with kc1:
        if st.button("Get deploy/svc/hpa/pods", key="btn_get_resources"):
            output = run(
                ["kubectl", "-n", k_ns, "get", "deploy,svc,hpa,pod", "-l", k_app_label]
            )
            st.code(output or "(no output)")

    with kc2:
        if st.button("Describe deployment", key="btn_describe_deploy"):
            output = run(
                ["kubectl", "-n", k_ns, "describe", "deploy", k_deploy_name]
            )
            st.code(output or "(no output)")

    with kc3:
        if st.button("Tail logs from first pod (last 100 lines)", key="btn_tail_logs"):
            pods_output = run(
                ["kubectl", "-n", k_ns, "get", "pod", "-l", k_app_label, "-o", "name"]
            )
            pod_name = ""
            for line in pods_output.splitlines():
                line = line.strip()
                if line.startswith("pod/"):
                    pod_name = line.split("/", 1)[1]
                    break
            if not pod_name:
                st.code("No pod found with selector " + k_app_label)
            else:
                logs = run(
                    [
                        "kubectl",
                        "-n",
                        k_ns,
                        "logs",
                        pod_name,
                        "--tail=100",
                    ]
                )
                st.code(f"# logs for {pod_name}\n" + (logs or "(no logs)"))
else:
    st.info("Kubectl commands are disabled. Use the sidebar toggle to enable live checks.")

# ---------- Backend API Explorer ----------
st.markdown("---")
st.markdown("### Backend API Explorer (OpenAPI)")
st.caption(
    "💡 Uses docs/dish-chat-backendapi.yaml from the backend repo. "
    "Base URL comes from the sidebar (Kubernetes / Base URL section)."
)

spec, spec_err = load_openapi_spec(base_dir)
if spec is None:
    st.warning(spec_err)
else:
    base_url = st.session_state.get("base_url", "")
    st.text(f"Backend base URL: {base_url or '<set in the sidebar>'}")

    endpoints = build_endpoint_index(spec)
    if not endpoints:
        st.warning("No endpoints found in OpenAPI spec.")
    else:
        labels = [e[0] for e in endpoints]
        idx = st.selectbox(
            "Select endpoint",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            key="endpoint_select",
        )
        label, path, method, meta = endpoints[idx]

        st.markdown(f"**Selected:** `{method} {path}`")
        summary = meta.get("summary") or meta.get("description") or ""
        if summary:
            st.caption(summary)

        # Path parameters
        path_params = re.findall(r"{([^}]+)}", path)
        path_values: Dict[str, str] = {}
        if path_params:
            st.markdown("**Path parameters**")
            for p in path_params:
                path_values[p] = st.text_input(
                    f"`{p}` value",
                    value="",
                    key=f"path_param_{p}",
                )

        # Build final path with substituted params
        final_path = path
        for name, val in path_values.items():
            final_path = final_path.replace("{" + name + "}", val)

        st.markdown("**Headers (JSON)**")
        headers_text = st.text_area(
            "Headers JSON",
            key="headers_json",
            value=st.session_state.get("headers_json", "{}"),
            help='Example: {"Authorization": "Bearer <token>"}',
            height=80,
        )

        example_body = extract_example_body(meta)
        default_body = (
            json.dumps(example_body, indent=2)
            if example_body is not None
            else st.session_state.get("body_text", "")
        )
        st.markdown("**Request body (JSON or raw)**")
        body_text = st.text_area(
            "Body",
            key="body_text",
            value=default_body,
            help="If valid JSON, will be sent as JSON. Otherwise sent as raw body.",
            height=160,
        )

        # Parse headers JSON (without writing back to session_state)
        headers: Optional[Dict[str, str]] = {}
        if headers_text.strip():
            try:
                parsed = json.loads(headers_text)
                if not isinstance(parsed, dict):
                    st.error("Headers JSON must decode to an object/dict.")
                    headers = None
                else:
                    headers = {str(k): str(v) for k, v in parsed.items()}
            except Exception as e:
                st.error(f"Failed to parse headers JSON: {e}")
                headers = None

        # Build full URL
        if base_url:
            url = urljoin(base_url.rstrip("/") + "/", final_path.lstrip("/"))
        else:
            url = final_path

        col_req, col_curl = st.columns(2)

        with col_req:
            if st.button("Send request", key="btn_send_request"):
                if not base_url:
                    st.error("Set Backend Base URL in the sidebar before sending a request.")
                elif headers is None:
                    st.error("Fix headers JSON before sending a request.")
                else:
                    resp, err = try_requests_request(method, url, headers, body_text)
                    if err:
                        st.error(err)
                    else:
                        st.markdown(f"**Status:** `{resp.status_code}`")
                        st.markdown("**Response headers:**")
                        st.code(json.dumps(dict(resp.headers), indent=2))
                        text_resp = resp.text
                        try:
                            parsed_json = resp.json()
                            st.markdown("**Response body (JSON):**")
                            st.code(json.dumps(parsed_json, indent=2))
                        except Exception:
                            st.markdown("**Response body (raw):**")
                            st.code(text_resp)

        with col_curl:
            if st.button("Generate curl", key="btn_generate_curl"):
                if not base_url:
                    st.error("Set Backend Base URL in the sidebar before generating curl.")
                elif headers is None:
                    st.error("Fix headers JSON before generating curl.")
                else:
                    curl_cmd = build_curl_command(
                        method=method,
                        url=url,
                        headers=headers,
                        body=body_text if body_text.strip() else None,
                        multiline=True,
                    )
                    st.markdown("**Copy/paste-ready curl:**")
                    st.code(curl_cmd, language="bash")

# ---------- FE → BE → K8s Correlation Panel ----------
st.markdown("---")
st.markdown("### 🔗 Endpoint Correlation: FE → BE → K8s")
st.caption(
    "Click an endpoint to see where it lives across all three layers "
    "(frontend → backend → K8s deployment). If a file path is wrong, you'll see a warning "
    "and can adjust ENDPOINT_MAP at the top of this file."
)

endpoint_keys = list(ENDPOINT_MAP.keys())
selected_endpoint = st.selectbox(
    "Select an endpoint to trace",
    options=endpoint_keys,
    format_func=lambda x: x,
    key="endpoint_correlation",
)

if selected_endpoint:
    mapping = ENDPOINT_MAP[selected_endpoint]
    col_fe, col_be, col_k8s = st.columns(3)

    with col_fe:
        st.markdown("**Frontend Service**")
        fe_path = base_dir / "dish-chat-fe" / mapping["fe_service"]
        if fe_path.exists():
            st.success(f"✓ `{mapping['fe_service']}`")
            with st.expander("View file"):
                try:
                    st.code(fe_path.read_text(encoding="utf-8")[:2000] + "\n...")
                except Exception as e:
                    st.error(f"Cannot read file: {e}")
        else:
            st.warning(f"⚠ File not found: `{mapping['fe_service']}`")

    with col_be:
        st.markdown("**Backend Route**")
        be_path = base_dir / "dish-chat" / mapping["be_route"]
        if be_path.exists():
            st.success(f"✓ `{mapping['be_route']}`")
            with st.expander("View file"):
                try:
                    st.code(be_path.read_text(encoding="utf-8")[:2000] + "\n...")
                except Exception as e:
                    st.error(f"Cannot read file: {e}")
        else:
            st.warning(f"⚠ File not found: `{mapping['be_route']}`")

    with col_k8s:
        st.markdown("**K8s Resource**")
        k8s_path = base_dir / "chat-bot" / mapping["k8s_resource"]
        if k8s_path.exists():
            st.success(f"✓ `{mapping['k8s_resource']}`")
            with st.expander("View file"):
                try:
                    st.code(k8s_path.read_text(encoding="utf-8")[:2000] + "\n...")
                except Exception as e:
                    st.error(f"Cannot read file: {e}")
        else:
            st.warning(f"⚠ File not found: `{mapping['k8s_resource']}`")
