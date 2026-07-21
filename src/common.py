"""Shared plumbing: tolerant JSONL IO, atomic writes, UTC dates, URL matching, HTTP sessions.

Every module reads/writes the same artifact shapes; this is the single home for those
mechanics so a fix lands everywhere at once.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


def load_jsonl(path: str | Path, tolerate_torn_tail: bool = True) -> list[dict]:
    """Read a JSONL file; missing file → []. A malformed FINAL line (torn by a crash mid-append)
    is dropped with a warning instead of aborting — resume must survive the very crash it exists
    to recover from. A malformed line in the MIDDLE still raises: that's corruption, not a torn tail.
    """
    p = Path(path)
    if not p.exists():
        return []
    # Split on "\n" ONLY (the JSONL record delimiter). Python's str.splitlines() ALSO breaks
    # on \v \f \x1c-\x1e \x85 U+2028 U+2029 - any of which can appear INSIDE a JSON string
    # value (e.g. a result snippet), where json.dumps(ensure_ascii=False) writes them literally.
    # Splitting on them tears an otherwise-valid record mid-string (observed 2026-07: a lone
    # U+2028 in a parallel-turbo excerpt broke both report generation and resume-load).
    # rstrip("\r") keeps CRLF-terminated files parsing.
    lines = [l.rstrip("\r") for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerate_torn_tail and i == len(lines) - 1:
                print(f"WARN: dropping torn final line in {p} (crash mid-write); "
                      f"the affected record will be regenerated", flush=True)
                break
            raise
    return out


def write_jsonl_atomic(path: str | Path, records: list[dict]) -> None:
    """Write records as JSONL atomically (tmp file + os.replace), always with a trailing
    newline — a crash mid-write can never destroy the previous file contents, and
    `cat a.jsonl b.jsonl` can never fuse records across files."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    tmp.write_text(body + ("\n" if body else ""), encoding="utf-8")
    os.replace(tmp, p)


def today_utc() -> str:
    """UTC date string. All date stamps and same-day comparisons use UTC so artifacts produced
    on different machines/timezones agree (run_meta timestamps are already UTC)."""
    return datetime.now(timezone.utc).date().isoformat()


def url_match(candidate: str, gold: str, mode: str) -> bool:
    """Anchor URL matching. 'prefix' is path-boundary aware: gold '…/repo' must not match
    '…/repo-archive', and 'https://who.int' must not match 'https://who.int.evil.com'."""
    c, g = candidate.rstrip("/"), gold.rstrip("/")
    if mode == "exact":
        return c == g
    if mode == "prefix":
        return c == g or c.startswith(g + "/") or c.startswith(g + "?")
    if mode == "domain":
        return (urlparse(c).netloc.removeprefix("www.").lower()
                == urlparse(g).netloc.removeprefix("www.").lower())
    raise ValueError(mode)


def load_queries(paths: str | Path | list) -> list[dict]:
    """Load one or more query JSONLs with a duplicate-qid guard (duplicates silently collapse
    resume keys and rubric slots downstream — fail loudly instead)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    for p in paths:
        out.extend(load_jsonl(p, tolerate_torn_tail=False))
    dupes = [q for q, n in Counter(r["qid"] for r in out).items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate qids across query files: {sorted(dupes)[:5]}")
    return out


def stale_realtime(queries: list[dict], today_iso: str) -> list[str]:
    """Realtime queries synthesized on an earlier (UTC) day — their answers have likely moved on."""
    return [q["qid"] for q in queries
            if q.get("freshness") == "realtime"
            and q.get("meta", {}).get("synth_at")
            and q["meta"]["synth_at"] < today_iso]


_tls = threading.local()


def http_session() -> requests.Session:
    """Per-thread requests.Session: connection pooling / TLS reuse without sharing a Session
    across threads. Cuts per-call handshake cost out of the measured backend latencies."""
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        _tls.session = s
    return s


def require_pairwise_meta(run: Path) -> dict:
    """Load run_meta.json for tools that consume pairwise run dirs, with clear errors for the
    two common operator mistakes (interrupted run, agent-eval dir)."""
    meta_p = Path(run) / "run_meta.json"
    if not meta_p.exists():
        raise SystemExit(f"{meta_p} missing — run interrupted before startup metadata was written?")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if "ours" not in meta:
        raise SystemExit(f"{run} looks like an agent-eval run dir (run_meta.json has no 'ours'); "
                         f"this tool consumes pairwise run dirs produced by src.run_eval")
    return meta
