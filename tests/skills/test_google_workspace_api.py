"""Tests for Google Workspace gws bridge and CLI wrapper."""

import importlib.util
import json
import os
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


BRIDGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/gws_bridge.py"
)
API_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)


@pytest.fixture
def bridge_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("gws_bridge_test", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("gws_api_test", API_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # Ensure the gws CLI code path is taken even when the binary isn't
    # installed (CI).  Without this, calendar_list() falls through to the
    # Python SDK path which imports ``googleapiclient`` — not in deps.
    module._gws_binary = lambda: "/usr/bin/gws"
    # Bypass authentication check — no real token file in CI.
    module._ensure_authenticated = lambda: None
    return module


def _write_token(path: Path, *, token="ya29.test", expiry=None, **extra):
    data = {
        "token": token,
        "refresh_token": "1//refresh",
        "client_id": "123.apps.googleusercontent.com",
        "client_secret": "secret",
        "token_uri": "https://oauth2.googleapis.com/token",
        **extra,
    }
    if expiry is not None:
        data["expiry"] = expiry
    path.write_text(json.dumps(data))


def test_bridge_returns_valid_token(bridge_module, tmp_path):
    """Non-expired token is returned without refresh."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.valid", expiry=future)

    result = bridge_module.get_valid_token()
    assert result == "ya29.valid"


def test_bridge_refreshes_expired_token(bridge_module, tmp_path):
    """Expired token triggers a refresh via token_uri."""
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.old", expiry=past)

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "access_token": "ya29.refreshed",
        "expires_in": 3600,
    }).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = bridge_module.get_valid_token()

    assert result == "ya29.refreshed"
    # Verify persisted
    saved = json.loads(token_path.read_text())
    assert saved["token"] == "ya29.refreshed"
    assert saved["type"] == "authorized_user"


def test_bridge_exits_on_missing_token(bridge_module):
    """Missing token file causes exit with code 1."""
    with pytest.raises(SystemExit):
        bridge_module.get_valid_token()


def test_bridge_main_injects_token_env(bridge_module, tmp_path):
    """main() sets GOOGLE_WORKSPACE_CLI_TOKEN in subprocess env."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.injected", expiry=future)

    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return MagicMock(returncode=0)

    with patch.object(sys, "argv", ["gws_bridge.py", "gmail", "+triage"]):
        with patch.object(subprocess, "run", side_effect=capture_run):
            with pytest.raises(SystemExit):
                bridge_module.main()

    assert captured["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "ya29.injected"
    assert captured["cmd"] == ["gws", "gmail", "+triage"]


def test_api_calendar_list_uses_events_list(api_module):
    """calendar_list calls _run_gws with events list + params."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="{}", stderr="")

    args = api_module.argparse.Namespace(
        start="", end="", max=25, calendar="primary", func=api_module.calendar_list,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.calendar_list(args)

    cmd = captured["cmd"]
    # _gws_binary() returns "/usr/bin/gws", so cmd[0] is that binary
    assert cmd[0] == "/usr/bin/gws"
    assert "calendar" in cmd
    assert "events" in cmd
    assert "list" in cmd
    assert "--params" in cmd
    params = json.loads(cmd[cmd.index("--params") + 1])
    assert "timeMin" in params
    assert "timeMax" in params
    assert params["calendarId"] == "primary"


def test_api_calendar_list_respects_date_range(api_module):
    """calendar list with --start/--end passes correct time bounds."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="{}", stderr="")

    args = api_module.argparse.Namespace(
        start="2026-04-01T00:00:00Z",
        end="2026-04-07T23:59:59Z",
        max=25,
        calendar="primary",
        func=api_module.calendar_list,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.calendar_list(args)

    cmd = captured["cmd"]
    params_idx = cmd.index("--params")
    params = json.loads(cmd[params_idx + 1])
    assert params["timeMin"] == "2026-04-01T00:00:00Z"
    assert params["timeMax"] == "2026-04-07T23:59:59Z"


def test_api_get_credentials_refresh_persists_authorized_user_type(api_module, monkeypatch):
    token_path = api_module.TOKEN_PATH
    _write_token(token_path, token="ya29.old")

    class FakeCredentials:
        def __init__(self):
            self.expired = True
            self.refresh_token = "1//refresh"
            self.valid = True

        def refresh(self, request):
            self.expired = False

        def to_json(self):
            return json.dumps({
                "token": "ya29.refreshed",
                "refresh_token": "1//refresh",
                "client_id": "123.apps.googleusercontent.com",
                "client_secret": "secret",
                "token_uri": "https://oauth2.googleapis.com/token",
            })

    class FakeCredentialsModule:
        @staticmethod
        def from_authorized_user_file(filename, scopes):
            assert filename == str(token_path)
            assert scopes == api_module.SCOPES
            return FakeCredentials()

    google_module = types.ModuleType("google")
    oauth2_module = types.ModuleType("google.oauth2")
    credentials_module = types.ModuleType("google.oauth2.credentials")
    credentials_module.Credentials = FakeCredentialsModule
    transport_module = types.ModuleType("google.auth.transport")
    requests_module = types.ModuleType("google.auth.transport.requests")
    requests_module.Request = lambda: object()

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_module)

    creds = api_module.get_credentials()

    saved = json.loads(token_path.read_text())
    assert isinstance(creds, FakeCredentials)
    assert saved["token"] == "ya29.refreshed"
    assert saved["type"] == "authorized_user"


def test_api_get_sendas_primary_prefers_primary_alias(api_module):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(
            {
                "sendAs": [
                    {"sendAsEmail": "alias@example.com", "isDefault": True, "signature": "<p>Default</p>"},
                    {"sendAsEmail": "primary@example.com", "isPrimary": True, "signature": "<p>Primary</p>"},
                ]
            }
        ),
        stderr="",
    )

    with patch.object(subprocess, "run", return_value=completed):
        alias = api_module._get_sendas_primary()

    assert alias["sendAsEmail"] == "primary@example.com"
    assert alias["signature"] == "<p>Primary</p>"


def test_api_get_sendas_primary_gracefully_degrades_on_empty_list(api_module):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"sendAs": []}),
        stderr="",
    )

    with patch.object(subprocess, "run", return_value=completed):
        alias = api_module._get_sendas_primary()

    assert alias == {}


def test_api_gmail_send_appends_signature_and_forces_html(api_module):
    sendas_payload = {
        "sendAs": [
            {"sendAsEmail": "lev@valstratis.com", "isPrimary": True, "signature": "<table>sig</table>"}
        ]
    }
    send_payload = {"id": "msg-123", "threadId": "thr-456"}
    captured = []

    def capture_run(cmd, **kwargs):
        captured.append(cmd)
        if "sendAs" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(sendas_payload), stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(send_payload), stderr="")

    args = api_module.argparse.Namespace(
        to="user@example.com",
        subject="Hi",
        body="<p>Hello</p>",
        cc="",
        from_header="",
        html=False,
        no_signature=False,
        thread_id="",
        func=api_module.gmail_send,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_send(args)

    send_cmd = captured[-1]
    body = json.loads(send_cmd[send_cmd.index("--json") + 1])
    raw = body["raw"]
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
    assert "<p>Hello</p><br><br><table>sig</table>" in decoded
    assert "Content-Type: text/html" in decoded


def test_api_gmail_send_no_signature_leaves_body_unchanged(api_module):
    sendas_payload = {
        "sendAs": [
            {"sendAsEmail": "lev@valstratis.com", "isPrimary": True, "signature": "<table>sig</table>"}
        ]
    }
    send_payload = {"id": "msg-123", "threadId": "thr-456"}
    captured = []

    def capture_run(cmd, **kwargs):
        captured.append(cmd)
        if "sendAs" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(sendas_payload), stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(send_payload), stderr="")

    args = api_module.argparse.Namespace(
        to="user@example.com",
        subject="Hi",
        body="Hello",
        cc="",
        from_header="",
        html=False,
        no_signature=True,
        thread_id="",
        func=api_module.gmail_send,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_send(args)

    send_cmd = captured[-1]
    body = json.loads(send_cmd[send_cmd.index("--json") + 1])
    raw = body["raw"]
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
    assert "\n\nHello" in decoded
    assert "sig" not in decoded
    assert "Content-Type: text/plain" in decoded


def test_api_gmail_reply_appends_signature_and_forces_html(api_module):
    sendas_payload = {
        "sendAs": [
            {"sendAsEmail": "lev@valstratis.com", "isDefault": True, "signature": "<p>sig</p>"}
        ]
    }
    original_payload = {
        "id": "orig-123",
        "threadId": "thr-456",
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "Subject", "value": "Status"},
                {"name": "Message-ID", "value": "<abc@example.com>"},
            ]
        },
    }
    send_payload = {"id": "msg-789", "threadId": "thr-456"}
    captured = []

    def capture_run(cmd, **kwargs):
        captured.append(cmd)
        if "sendAs" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(sendas_payload), stderr="")
        if "messages" in cmd and "get" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(original_payload), stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(send_payload), stderr="")

    args = api_module.argparse.Namespace(
        message_id="abc123",
        body="Thanks",
        from_header="",
        html=False,
        no_signature=False,
        func=api_module.gmail_reply,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_reply(args)

    send_cmd = captured[-1]
    body = json.loads(send_cmd[send_cmd.index("--json") + 1])
    raw = body["raw"]
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
    assert "Thanks<br><br><p>sig</p>" in decoded
    assert "Content-Type: text/html" in decoded
