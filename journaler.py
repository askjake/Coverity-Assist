
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

JOURNAL_DIR = Path(os.environ.get("JOURNALS_DIR", str(Path.cwd() / "journals")))
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = JOURNAL_DIR / "code_cleanup.log"
JOURNAL_FILE = JOURNAL_DIR / "code_cleanup.journal"
ALEX_JOURNAL = JOURNAL_DIR / "alex.journal"
GEMMA_JOURNAL = JOURNAL_DIR / "gemma.journal"
GABRIEL_JOURNAL = JOURNAL_DIR / "gabriel.journal"

# ~100 KB default cap before we summarize/rotate
LOG_SIZE_LIMIT = int(float(os.environ.get("JOURNAL_LOG_LIMIT_MB", "0.1")) * 1024 * 1024)

def log_activity(message: str, log_file: Path = LOG_FILE) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{ts}] {message}\n")

def append_to_journal(text: str, journal_file: Path = JOURNAL_FILE) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(journal_file, 'a', encoding='utf-8') as f:
        f.write(f"\\n=== Summary {ts} ===\\n{text.strip()}\\n")

def check_log_size(summarize_fn: Optional[Callable[[str], str]] = None,
                   log_file: Path = LOG_FILE,
                   journal_file: Path = JOURNAL_FILE) -> None:
    """
    If the log file exceeds LOG_SIZE_LIMIT, summarize and rotate.
    summarize_fn: function that takes the log text and returns a summary string.
    """
    try:
        if not log_file.exists() or log_file.stat().st_size < LOG_SIZE_LIMIT:
            return
        text = log_file.read_text(encoding='utf-8', errors='replace')
        summary = summarize_fn(text) if summarize_fn else text[:2000] + "\\n(truncated)"
        append_to_journal(summary, journal_file=journal_file)
        # rotate (clear) the log
        log_file.write_text("", encoding="utf-8")
    except Exception as e:
        # Best-effort — do not crash callers
        err = f"journaler.check_log_size error: {e}"
        log_activity(err, log_file=log_file)
