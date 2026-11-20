"""Shared constants and helpers for the external-data fetch/build scripts.

Guards (checked on every request, not configurable):
  - only the three allowed hosts are contacted;
  - no Hyperliquid request carries an endTime at or after the cutoff;
  - nothing at or after the cutoff is written (enforced by the callers before every write).
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]          # work/extdata
RAW = ROOT / "raw"
BUILD = ROOT / "build"
LOGS = ROOT / "logs"
for _d in (RAW, BUILD, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2026-01-05 00:00:00")        # UTC naive; only records with ts < CUTOFF are kept
CUTOFF_MS = int(CUTOFF.tz_localize("UTC").value // 10**6)
assert CUTOFF_MS == 1767571200000

HL_URL = "https://api.hyperliquid.xyz/info"
ALLOWED_HOSTS = {"api.hyperliquid.xyz", "data.binance.vision", "fapi.binance.com"}

PKG_DIR = "/Users/xc/Documents/arkstream_quest/ArkStream笔试题_v2"
V2_CODE = "/Users/xc/Documents/arkstream_quest/work/v2dev/code"
GROUPS_PARQUET = Path("/Users/xc/Documents/arkstream_quest/work/v2dev/build/groups.parquet")


def ts_ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").value // 10**6)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc_now_iso()}] {msg}", flush=True)


def load_package():
    """Perp fills of the package through the existing v2dev loader."""
    os.environ.setdefault("ARK_DATA_DIR", PKG_DIR)
    if V2_CODE not in sys.path:
        sys.path.insert(0, V2_CODE)
    import lib  # noqa: E402
    return lib.load_fills(perp_only=True)


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def check_host(url: str) -> None:
    host = urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise RuntimeError(f"host not allowed: {host}")


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------- Hyperliquid /info client
class FetchError(Exception):
    pass


class HLClient:
    """Sequential /info client with weight-aware pacing.

    Each request waits until max(min_interval, weight * 60 / budget) seconds have passed since
    the previous one, where weight = 20 + n_items / items_per_weight (the documented cost model).
    HTTP 429 / 5xx -> exponential back-off (30 s doubling, cap 600 s), retried indefinitely.
    Exception: a 500 whose body is literally `null` is what the API returns for an unknown
    coin, so it is treated as a permanent error (a few retries, then FetchError).
    Other errors (4xx, bad JSON, connection errors, timeouts) -> `other_retries` retries.
    """

    def __init__(self, min_interval: float, budget_per_min: float, items_per_weight: float,
                 other_retries: int = 3):
        self.min_interval = min_interval
        self.budget = budget_per_min
        self.items_per_weight = items_per_weight
        self.other_retries = other_retries
        self.session = requests.Session()
        self._next = 0.0
        self.n_requests = 0
        self.n_backoffs = 0
        self.t_first = None

    def _wait(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)

    def _done(self, n_items: int):
        weight = 20 + n_items / self.items_per_weight
        self._next = time.monotonic() + max(self.min_interval, weight * 60.0 / self.budget)

    def post(self, body: dict):
        v = body.get("endTime", body.get("req", {}).get("endTime"))
        if v is None or v >= CUTOFF_MS:
            raise RuntimeError(f"refusing request with endTime {v} (cutoff {CUTOFF_MS})")
        check_host(HL_URL)
        backoff = 30.0
        other = 0
        while True:
            self._wait()
            if self.t_first is None:
                self.t_first = time.monotonic()
            self.n_requests += 1
            err = None
            try:
                r = self.session.post(HL_URL, json=body, timeout=60)
            except (requests.ConnectionError, requests.Timeout) as e:
                r, err = None, f"{type(e).__name__}: {e}"
            if r is not None and (r.status_code == 429 or r.status_code >= 500) \
                    and r.text.strip() != "null":
                self._done(0)
                self.n_backoffs += 1
                log(f"  HTTP {r.status_code}; backing off {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 600.0)
                continue
            if r is not None and r.status_code == 200:
                try:
                    data = r.json()
                except ValueError as e:
                    err = f"bad json: {e}"
                else:
                    self._done(len(data) if isinstance(data, list) else 0)
                    return data
            elif r is not None:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            self._done(0)
            other += 1
            if other > self.other_retries:
                raise FetchError(err)
            log(f"  error ({err}); retry {other}/{self.other_retries} in {10 * other}s")
            time.sleep(10 * other)
