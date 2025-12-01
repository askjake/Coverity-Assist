import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ExecutionAttempt:
    timestamp: str
    command: Optional[str] = None
    url: Optional[str] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def is_error(self) -> bool:
        """Determine if this attempt represents an error."""
        if self.command:
            return self.exit_code != 0
        if self.url:
            # Check for common error patterns in response
            error_patterns = [
                r"error from server",
                r"notfound",
                r"connection refused",
                r"unable to connect",
                r"ssl certificate problem",
                r"<title>.*error.*</title>",
                r"403 forbidden",
                r"404 not found",
                r"500 internal server error",
            ]
            content = (self.stdout + self.stderr).lower()
            return any(re.search(pattern, content, re.IGNORECASE) for pattern in error_patterns)
        return False

    def extract_error_info(self) -> Dict[str, str]:
        """Extract structured error information."""
        info = {}

        if "ssl certificate problem" in self.stderr.lower():
            info["error_type"] = "SSL_CERTIFICATE_ERROR"
            info["suggestion"] = "Add -k or --insecure flag to curl, or fix certificate chain"

        elif "command not found" in self.stderr.lower():
            info["error_type"] = "COMMAND_NOT_FOUND"
            match = re.search(r"(\w+): not found", self.stderr)
            if match:
                info["missing_command"] = match.group(1)
            info["suggestion"] = "Install missing command or use full path"

        elif "connection refused" in self.stderr.lower():
            info["error_type"] = "CONNECTION_REFUSED"
            info["suggestion"] = "Check if service is running and port is correct"

        elif self.exit_code == 127:
            info["error_type"] = "COMMAND_NOT_FOUND"
            info["suggestion"] = "Command not in PATH or doesn't exist"

        elif "permission denied" in self.stderr.lower():
            info["error_type"] = "PERMISSION_DENIED"
            info["suggestion"] = "Check file permissions or use sudo"

        return info


@dataclass
class CommandHistory:
    original_command: str
    attempts: List[ExecutionAttempt]
    llm_suggestions: List[str]
    final_success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_command": self.original_command,
            "attempts": [asdict(a) for a in self.attempts],
            "llm_suggestions": self.llm_suggestions,
            "final_success": self.final_success,
        }
