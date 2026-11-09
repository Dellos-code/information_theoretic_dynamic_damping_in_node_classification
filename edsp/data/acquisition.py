"""Download with an append-only acquisition log (Master Prompt §3.3, rule R4/R8).

Every attempt — successful or not — appends one JSON line to
``data/ACQUISITION_LOG.jsonl`` (or the path in ``EDSP_ACQUISITION_LOG``).
Raw files are cached under ``EDSP_DATA_DIR`` (default ``~/.cache/edsp``) and
are never committed.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    d = Path(os.environ.get("EDSP_DATA_DIR", Path.home() / ".cache" / "edsp"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_path() -> Path:
    return Path(os.environ.get("EDSP_ACQUISITION_LOG", REPO_ROOT / "data" / "ACQUISITION_LOG.jsonl"))


def environment_label() -> str:
    run = os.environ.get("GITHUB_RUN_ID")
    return f"actions-run-{run}" if run else os.environ.get("EDSP_ENVIRONMENT", "local")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def append_log(record: dict) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


class AcquisitionError(RuntimeError):
    pass


def fetch(
    url: str,
    relpath: str,
    *,
    dataset: str,
    cell_id: str,
    candidate_rank: int,
    method: str = "urllib",
    timeout: int = 120,
    expected_md5: str | None = None,
) -> Path:
    """Return a cached local copy of ``url``; log every network attempt.

    The SHA-256 (and MD5) of every downloaded file is recorded in the log. If
    ``expected_md5`` is given (e.g. the checksum a Zenodo record publishes), a
    mismatch deletes the download, is logged as ``failure_reason: checksum`` and
    raises; a cached file is not re-checked."""
    target = data_dir() / relpath
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "cell_id": cell_id,
        "candidate_rank": candidate_rank,
        "dataset": dataset,
        "source_url": url,
        "method": method,
        "environment": environment_label(),
    }
    tmp = target.with_suffix(target.suffix + ".part")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "edsp-dataset-fetch"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            status = getattr(resp, "status", 200)
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                sha.update(chunk)
                md5.update(chunk)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        reason = "404" if exc.code == 404 else ("firewall" if exc.code in (403, 407) else "other")
        append_log({**base, "http_status": exc.code, "bytes": 0, "outcome": "FAILED",
                    "failure_reason": reason, "next_action": "try next route/candidate"})
        raise AcquisitionError(f"{url}: HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        append_log({**base, "http_status": None, "bytes": 0, "outcome": "FAILED",
                    "failure_reason": "firewall" if "403" in str(exc) else "other",
                    "detail": str(exc)[:300], "next_action": "retry in Actions / next route"})
        raise AcquisitionError(f"{url}: {exc}") from exc
    digest = {"sha256_raw": "sha256:" + sha.hexdigest(), "md5_raw": md5.hexdigest()}
    with open(tmp, "rb") as fh:
        head = fh.read(512).lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        # e.g. a Google Drive quota / virus-scan page served with HTTP 200 instead of the file
        size = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        append_log({**base, "http_status": status, "bytes": size, **digest, "outcome": "FAILED",
                    "failure_reason": "format", "detail": "HTML page instead of a data file",
                    "next_action": "try next route/candidate"})
        raise AcquisitionError(f"{url}: received an HTML page ({size} bytes), not a data file")
    if expected_md5 is not None:
        digest["md5_expected"] = expected_md5.lower()
        if md5.hexdigest() != expected_md5.lower():
            size = tmp.stat().st_size
            tmp.unlink(missing_ok=True)
            append_log({**base, "http_status": status, "bytes": size, **digest, "outcome": "FAILED",
                        "failure_reason": "checksum", "next_action": "try next route/candidate"})
            raise AcquisitionError(f"{url}: md5 {md5.hexdigest()} != expected {expected_md5}")
    tmp.replace(target)
    append_log({**base, "http_status": status, "bytes": target.stat().st_size, **digest, "outcome": "OK"})
    return target


def log_processed(dataset: str, cell_id: str, processed_hash: str, report: dict) -> None:
    append_log({"dataset": dataset, "cell_id": cell_id, "environment": environment_label(),
                "outcome": "PROCESSED", "sha256_processed": processed_hash, "report": report})
