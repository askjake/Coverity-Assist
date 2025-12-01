class Self_Correcting_Executor:
    def __init__(self, llm_chat_function, max_retries: int = 3):
        self.llm_chat = llm_chat_function
        self.max_retries = max_retries
        self.history: Dict[str, CommandHistory] = {}

    def execute_with_correction(
            self,
            command: Optional[str] = None,
            url: Optional[str] = None,
            context: str = "",
    ) -> Tuple[bool, str, CommandHistory]:
        """
        Execute command/URL with automatic error correction.

        Returns:
            (success, output, history)
        """
        original = command or url
        history = CommandHistory(
            original_command=original,
            attempts=[],
            llm_suggestions=[],
        )

        current_cmd = command
        current_url = url

        for attempt_num in range(self.max_retries):
            # Execute
            if current_cmd:
                result = self._execute_command(current_cmd)
            else:
                result = self._fetch_url(current_url)

            history.attempts.append(result)

            # Check if successful
            if not result.is_error():
                history.final_success = True
                self.history[original] = history
                return True, result.stdout, history

            # Extract error info
            error_info = result.extract_error_info()

            # Ask LLM for correction
            if attempt_num < self.max_retries - 1:
                corrected = self._ask_llm_for_correction(
                    original_command=original,
                    attempts=history.attempts,
                    context=context,
                    error_info=error_info,
                )

                if corrected:
                    history.llm_suggestions.append(corrected)
                    if command:
                        current_cmd = corrected
                    else:
                        # For URLs, might suggest different approach
                        if corrected.startswith("curl"):
                            current_cmd = corrected
                            current_url = None
                        else:
                            current_url = corrected
                else:
                    break  # LLM couldn't suggest fix

        # Failed after all retries
        self.history[original] = history
        return False, f"Failed after {len(history.attempts)} attempts", history

    def _execute_command(self, cmd: str) -> ExecutionAttempt:
        """Execute shell command and capture result."""
        import subprocess

        attempt = ExecutionAttempt(
            timestamp=datetime.utcnow().isoformat(),
            command=cmd,
        )

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            attempt.exit_code = result.returncode
            attempt.stdout = result.stdout
            attempt.stderr = result.stderr
        except subprocess.TimeoutExpired:
            attempt.exit_code = 124
            attempt.stderr = "Command timed out after 30 seconds"
        except Exception as e:
            attempt.exit_code = 1
            attempt.stderr = str(e)

        return attempt

    def _fetch_url(self, url: str) -> ExecutionAttempt:
        """Fetch URL and capture result."""
        attempt = ExecutionAttempt(
            timestamp=datetime.utcnow().isoformat(),
            url=url,
        )

        try:
            import requests
            resp = requests.get(url, timeout=30, verify=False)
            attempt.exit_code = 0 if resp.status_code < 400 else resp.status_code
            attempt.stdout = resp.text[:5000]  # Limit size
        except Exception as e:
            attempt.exit_code = 1
            attempt.stderr = str(e)

        return attempt

    def _ask_llm_for_correction(
            self,
            original_command: str,
            attempts: List[ExecutionAttempt],
            context: str,
            error_info: Dict[str, str],
    ) -> Optional[str]:
        """Ask LLM to suggest corrected command."""

        # Build prompt with error history
        prompt = f"""I tried to execute this command/URL but encountered errors:

**Original:** `{original_command}`

**Context:** {context}

**Attempt History:**
"""

        for i, attempt in enumerate(attempts, 1):
            prompt += f"\n**Attempt {i}:**\n"
            if attempt.command:
                prompt += f"Command: `{attempt.command}`\n"
            if attempt.url:
                prompt += f"URL: `{attempt.url}`\n"
            prompt += f"Exit code: {attempt.exit_code}\n"
            if attempt.stderr:
                prompt += f"Error: {attempt.stderr[:500]}\n"

        if error_info:
            prompt += f"\n**Error Analysis:**\n"
            for k, v in error_info.items():
                prompt += f"- {k}: {v}\n"

        prompt += """
Please provide a corrected command or URL that will work. Consider:
1. SSL certificate issues → add -k flag to curl
2. Command not found → use full path or alternative command
3. Connection issues → check URL format, port, protocol
4. Permission issues → suggest sudo or permission fixes

Respond with ONLY the corrected command/URL, nothing else. If you cannot fix it, respond with "CANNOT_FIX".
"""

        # Call LLM
        ok, response = self.llm_chat(prompt)

        if ok and response.strip() and response.strip() != "CANNOT_FIX":
            return response.strip()

        return None

    def save_history_to_file(self, filepath: str):
        """Save execution history for long-term memory."""
        data = {
            "saved_at": datetime.utcnow().isoformat(),
            "history": {k: v.to_dict() for k, v in self.history.items()},
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_history_from_file(self, filepath: str):
        """Load execution history from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Reconstruct history objects
            for orig_cmd, hist_data in data.get("history", {}).items():
                attempts = [
                    ExecutionAttempt(**attempt_data)
                    for attempt_data in hist_data["attempts"]
                ]
                self.history[orig_cmd] = CommandHistory(
                    original_command=hist_data["original_command"],
                    attempts=attempts,
                    llm_suggestions=hist_data["llm_suggestions"],
                    final_success=hist_data["final_success"],
                )
        except Exception as e:
            st.warning(f"Could not load history: {e}")
