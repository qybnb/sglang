"""Direct-to-file diagnostics for Ascend disaggregated KV transfers.

These diagnostics deliberately bypass the Python logging configuration.  The
Ascend scheduler is multi-process and transfer work also runs on background
threads, so a normal ``logger.error`` can be hard to find in the launcher log.
Set ``SGLANG_ASCEND_KV_DIAG_DIR`` to write one JSONL file per process.
"""

from __future__ import annotations

import datetime
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any


_write_lock = threading.Lock()
_diag_path: Path | None = None
_diag_dir: str | None = None


def _get_diag_path() -> Path | None:
    global _diag_dir, _diag_path

    diag_dir = os.getenv("SGLANG_ASCEND_KV_DIAG_DIR", "").strip()
    if not diag_dir:
        return None
    if _diag_path is not None and _diag_dir == diag_dir:
        return _diag_path

    path = Path(diag_dir)
    path.mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname().replace("/", "_")
    _diag_dir = diag_dir
    _diag_path = path / f"ascend_kv_{hostname}_pid{os.getpid()}.jsonl"
    return _diag_path


def write_ascend_kv_diag(event: str, **fields: Any) -> None:
    """Append one self-contained diagnostic record without affecting serving."""

    try:
        path = _get_diag_path()
        if path is None:
            return
        record = {
            "timestamp_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="microseconds"),
            "monotonic_ns": time.monotonic_ns(),
            "event": event,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        # Opening for each event makes completed records visible even if the
        # scheduler is killed immediately after a transport failure.
        with _write_lock, path.open("a", encoding="utf-8") as output:
            output.write(line)
            output.flush()
    except Exception:
        # Diagnostics must never turn a recoverable transfer error into a
        # scheduler crash.  Avoid logging here because this helper exists to
        # bypass potentially broken/misdirected logging configuration.
        return


def new_transfer_id() -> str:
    return f"{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}"
