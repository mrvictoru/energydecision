"""Tests for the ennexos fetcher (pure logic + mocked HTTP; no network)."""

import datetime as dt
import importlib.util
import os
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_ennexos",
    Path(__file__).parent.parent / "scripts" / "fetch_ennexos.py",
)
fe = importlib.util.module_from_spec(spec)
sys.modules["fetch_ennexos"] = fe
spec.loader.exec_module(fe)


class _Resp:
    def __init__(self, status=200, content=b'{"ok":true}', text=None):
        self.status_code = status
        self.content = content
        self.text = text if text is not None else content.decode()
        self.headers = {"content-type": "application/json"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise fe.requests.HTTPError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        r = self.responses.pop(0) if self.responses else _Resp()
        return r


def test_build_url_matches_observed_endpoint():
    url = fe.build_url("10574124", "2026-08-20", "Day")
    assert url == ("https://uiapi.sunnyportal.com/api/v1/measurements/10574124/"
                   "energybalance?dateBeginLocal=2026-08-20&interval=Day")


def test_build_headers_auth_variants():
    h = fe.build_headers("a=1", "tok")
    assert h["Authorization"] == "Bearer tok"
    assert h["Cookie"] == "a=1"
    assert "ennexos.sunnyportal.com" in h["Referer"]
    assert "Authorization" not in fe.build_headers()


def test_iter_dates_inclusive():
    ds = list(fe.iter_dates(dt.date(2024, 2, 28), dt.date(2024, 3, 1)))
    assert [d.isoformat() for d in ds] == ["2024-02-28", "2024-02-29", "2024-03-01"]


def test_probe_401_returns_exit_code_2(tmp_path, monkeypatch):
    monkeypatch.setenv("ENNXOS_COOKIE", "x=1")
    s = _Session([_Resp(status=401, content=b"unauthorized")])
    monkeypatch.setattr(fe.requests.Session, "get",
                        lambda self, url, timeout=None: s.get(url))
    rc = fe.main(["--plant-id", "1", "--start", "2026-08-20", "--end", "2026-08-20",
                  "--probe"])
    assert rc == 2


def test_fetch_loop_resumes_and_writes_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ENNXOS_COOKIE", "x=1")
    out = tmp_path / "raw"
    # day 1 pre-existing (resume skip), days 2-3 fetched, day 4 empty -> failed
    (out).mkdir()
    (out / "energybalance_2026-08-20_Day.json").write_bytes(b"old")
    responses = [_Resp(content=b"d2"), _Resp(content=b"d3"), _Resp(content=b"  ")]
    session = _Session(responses)

    def fake_get(self, url, timeout=None):
        session.calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(fe.requests.Session, "get", fake_get)
    rc = fe.main(["--plant-id", "7", "--start", "2026-08-20", "--end", "2026-08-23",
                  "--delay", "0", "--out-dir", str(out)])
    assert rc == 1  # one empty/failed day
    assert (out / "energybalance_2026-08-21_Day.json").read_bytes() == b"d2"
    assert (out / "energybalance_2026-08-22_Day.json").read_bytes() == b"d3"
    assert len(session.calls) == 3  # resumed day not re-requested


def test_no_credentials_fails_fast(monkeypatch, capsys):
    monkeypatch.delenv("ENNXOS_COOKIE", raising=False)
    monkeypatch.delenv("ENNXOS_BEARER", raising=False)
    rc = fe.main(["--plant-id", "1", "--start", "2026-08-20", "--end", "2026-08-20"])
    assert rc == 1
    assert "ENNXOS_COOKIE" in capsys.readouterr().err
