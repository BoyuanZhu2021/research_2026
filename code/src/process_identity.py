"""Fail-closed Linux process ownership checks with correct zombie handling."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


class ProcessIdentityError(RuntimeError):
    """A live PID exists but no longer belongs to the sealed project process."""


def parse_proc_stat(stat_text: str) -> tuple[str, int]:
    """Return Linux process state and start-time ticks from one /proc/PID/stat row."""
    close = stat_text.rfind(")")
    if close < 2:
        raise ProcessIdentityError("malformed /proc stat comm field")
    fields = stat_text[close + 2:].split()
    if len(fields) <= 19:
        raise ProcessIdentityError("truncated /proc stat row")
    return fields[0], int(fields[19])


def classify_owned_process(
    *,
    stat_text: str,
    actual_cmdline: Sequence[str],
    actual_pgid: int,
    expected_start_ticks: int,
    expected_cmdline: Sequence[str],
    expected_pgid: int,
) -> bool:
    """Return live/not-live, raising if a non-zombie PID has changed identity."""
    state, start_ticks = parse_proc_stat(stat_text)
    if state == "Z":
        return False
    if (
        start_ticks != expected_start_ticks
        or list(actual_cmdline) != list(expected_cmdline)
        or actual_pgid != expected_pgid
    ):
        raise ProcessIdentityError("live PID does not match its sealed process identity")
    return True


def owned_process_live(
    pid: int,
    *,
    expected_start_ticks: int,
    expected_cmdline: Sequence[str],
    expected_pgid: int,
) -> bool:
    """Inspect one project PID without treating an unreaped child as still running."""
    proc = Path(f"/proc/{pid}")
    if not proc.is_dir():
        return False
    try:
        stat_text = (proc / "stat").read_text(encoding="utf-8")
        raw_cmdline = (proc / "cmdline").read_bytes()
        cmdline = [
            part.decode("utf-8", "replace")
            for part in raw_cmdline.split(b"\0") if part
        ]
        pgid = os.getpgid(pid)
    except (FileNotFoundError, ProcessLookupError):
        return False
    return classify_owned_process(
        stat_text=stat_text,
        actual_cmdline=cmdline,
        actual_pgid=pgid,
        expected_start_ticks=expected_start_ticks,
        expected_cmdline=expected_cmdline,
        expected_pgid=expected_pgid,
    )


__all__ = [
    "ProcessIdentityError", "classify_owned_process", "owned_process_live",
    "parse_proc_stat",
]
