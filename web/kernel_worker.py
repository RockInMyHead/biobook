#!/usr/bin/env python3
"""Isolated notebook kernel worker.

Runs untrusted user code outside the API process with:
- scrubbed environment (no API secrets)
- resource limits
- PR_SET_NO_NEW_PRIVS when available
- private /tmp mounted by bubblewrap (the caller must create the sandbox)
"""
from __future__ import annotations

import builtins
import ctypes
import io
import json
import os
import pickle
import resource
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout


# Security: deny reading known secret material even if the file mode slips.
_BLOCKED_PATH_SNIPPETS = (
    "/etc/biobook.env",
    "biobook.env",
    "YANDEX_CLIENT_SECRET",
    "AI_API_KEY",
    "TG_BOT_TOKEN",
    "SESSION_SECRET",
    "ADMIN_KEY",
    "/.ssh/",
    "id_ed25519",
    "id_rsa",
)


def _deny_path(path: str) -> bool:
    low = (path or "").replace("\\", "/").lower()
    return any(s.lower() in low for s in _BLOCKED_PATH_SNIPPETS)


_real_open = builtins.open


def _safe_open(file, *args, **kwargs):
    try:
        path = file if isinstance(file, (str, bytes, os.PathLike)) else getattr(file, "name", "")
        if isinstance(path, bytes):
            path = path.decode("utf-8", "replace")
        if path and _deny_path(str(path)):
            raise PermissionError("Access denied")
    except PermissionError:
        raise
    except Exception:
        pass
    return _real_open(file, *args, **kwargs)


def _apply_no_new_privs() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_NO_NEW_PRIVS = 38
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            pass
    except Exception:
        pass


def _apply_rlimits() -> None:
    soft_hard = {
        resource.RLIMIT_CPU: (15, 20),  # seconds
        resource.RLIMIT_AS: (768 * 1024 * 1024, 768 * 1024 * 1024),
        resource.RLIMIT_DATA: (768 * 1024 * 1024, 768 * 1024 * 1024),
        resource.RLIMIT_FSIZE: (64 * 1024 * 1024, 64 * 1024 * 1024),
        resource.RLIMIT_NOFILE: (128, 128),
        resource.RLIMIT_NPROC: (64, 64),
        resource.RLIMIT_CORE: (0, 0),
    }
    for key, val in soft_hard.items():
        try:
            resource.setrlimit(key, val)
        except Exception:
            pass


def _load_ns(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {"__builtins__": builtins}
    try:
        with _real_open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            data["__builtins__"] = builtins
            return data
    except Exception:
        pass
    return {"__builtins__": builtins}


def _save_ns(path: str, ns: dict) -> None:
    if not path:
        return
    dump = {}
    for k, v in ns.items():
        if k.startswith("_") or k == "__builtins__":
            continue
        try:
            pickle.dumps(v)
            dump[k] = v
        except Exception:
            continue
    try:
        tmp = path + ".tmp"
        with _real_open(tmp, "wb") as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        pass


def main() -> int:
    raw = sys.stdin.buffer.read()
    try:
        req = json.loads(raw.decode("utf-8"))
    except Exception as e:
        sys.stdout.buffer.write(
            json.dumps({"ok": False, "output": f"bad request: {e}", "error": True}).encode()
        )
        return 1

    code = req.get("code") or ""
    cwd = req.get("cwd") or os.getcwd()
    ns_path = req.get("ns_path") or ""
    meta = req.get("meta") or {}

    os.environ["MPLBACKEND"] = "Agg"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TMPDIR"] = "/tmp"
    os.environ["TMP"] = "/tmp"
    os.environ["TEMP"] = "/tmp"

    _apply_no_new_privs()
    _apply_rlimits()
    builtins.open = _safe_open  # security: block secret file paths in user code

    ns = _load_ns(ns_path)
    for k, v in meta.items():
        ns[k] = v

    stdout = io.StringIO()
    stderr = io.StringIO()
    err = False
    try:
        os.makedirs(cwd, exist_ok=True)
        os.chdir(cwd)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, ns)  # noqa: S102 — intentional sandboxed exec
        _save_ns(ns_path, ns)
        out = stdout.getvalue()
        err_text = stderr.getvalue()
        if err_text:
            out = (out + ("\n" if out else "") + err_text)
    except Exception:
        err = True
        out = stdout.getvalue()
        tb = traceback.format_exc()
        err_text = stderr.getvalue()
        parts = [p for p in (out, err_text, tb) if p]
        out = "\n".join(parts)

    sys.stdout.buffer.write(
        json.dumps({"ok": not err, "output": out, "error": err}, ensure_ascii=False).encode("utf-8")
    )
    return 0 if not err else 1


if __name__ == "__main__":
    raise SystemExit(main())
