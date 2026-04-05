"""Regression tests for Google Workspace API credential validation."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)


class FakeAuthorizedCredentials:
    def __init__(self, *, valid=True, expired=False, refresh_token="refresh-token"):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refresh_calls = 0

    def refresh(self, _request):
        self.refresh_calls += 1
        self.valid = True
        self.expired = False

    def to_json(self):
        return json.dumps({
            "token": "refreshed-token",
            "refresh_token": self.refresh_token,
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": [
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.modify",
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/contacts.readonly",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/documents.readonly",
            ],
        })


class FakeCredentialsFactory:
    creds = FakeAuthorizedCredentials()

    @classmethod
    def from_authorized_user_file(cls, _path, _scopes):
        return cls.creds


@pytest.fixture
def google_api_module(monkeypatch, tmp_path):
    google_module = types.ModuleType("google")
    oauth2_module = types.ModuleType("google.oauth2")
    credentials_module = types.ModuleType("google.oauth2.credentials")
    credentials_module.Credentials = FakeCredentialsFactory
    auth_module = types.ModuleType("google.auth")
    transport_module = types.ModuleType("google.auth.transport")
    requests_module = types.ModuleType("google.auth.transport.requests")
    requests_module.Request = object

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_module)
    monkeypatch.setitem(sys.modules, "google.auth", auth_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_module)

    spec = importlib.util.spec_from_file_location("google_workspace_api_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "TOKEN_PATH", tmp_path / "google_token.json")
    return module


def _write_token(path: Path, scopes):
    path.write_text(json.dumps({
        "token": "access-token",
        "refresh_token": "refresh-token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scopes": scopes,
    }))


def test_get_credentials_rejects_missing_scopes(google_api_module, capsys):
    FakeCredentialsFactory.creds = FakeAuthorizedCredentials(valid=True)
    _write_token(google_api_module.TOKEN_PATH, [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/spreadsheets",
    ])

    with pytest.raises(SystemExit):
        google_api_module.get_credentials()

    err = capsys.readouterr().err
    assert "missing google workspace scopes" in err.lower()
    assert "gmail.send" in err


def test_get_credentials_accepts_full_scope_token(google_api_module):
    FakeCredentialsFactory.creds = FakeAuthorizedCredentials(valid=True)
    _write_token(google_api_module.TOKEN_PATH, list(google_api_module.SCOPES))

    creds = google_api_module.get_credentials()

    assert creds is FakeCredentialsFactory.creds


# ---------------------------------------------------------------------------
# _build_from_header
# ---------------------------------------------------------------------------

def test_build_from_header_with_display_name(google_api_module):
    alias = {"sendAsEmail": "user@example.com", "displayName": "Test User"}
    assert google_api_module._build_from_header(alias) == '"Test User" <user@example.com>'


def test_build_from_header_without_display_name(google_api_module):
    alias = {"sendAsEmail": "user@example.com", "displayName": ""}
    assert google_api_module._build_from_header(alias) == "user@example.com"


def test_build_from_header_empty_alias(google_api_module):
    assert google_api_module._build_from_header({}) == ""


# ---------------------------------------------------------------------------
# _get_sendas_primary
# ---------------------------------------------------------------------------

def test_get_sendas_primary_returns_primary_alias(google_api_module, monkeypatch):
    primary = {"sendAsEmail": "user@example.com", "displayName": "Test User", "isPrimary": True, "signature": "<b>Sig</b>"}
    other = {"sendAsEmail": "alias@example.com", "isPrimary": False}

    fake_sendas = {"sendAs": [other, primary]}

    class FakeSettings:
        def sendAs(self):
            return self

        def list(self, userId):
            return self

        def execute(self):
            return fake_sendas

    class FakeService:
        def users(self):
            return self

        def settings(self):
            return FakeSettings()

    monkeypatch.setattr(google_api_module, "build_service", lambda api, ver: FakeService())

    result = google_api_module._get_sendas_primary()
    assert result == primary


def test_get_sendas_primary_degrades_gracefully_on_error(google_api_module, monkeypatch):
    monkeypatch.setattr(google_api_module, "build_service", lambda api, ver: (_ for _ in ()).throw(Exception("network error")))

    result = google_api_module._get_sendas_primary()
    assert result == {}


# ---------------------------------------------------------------------------
# gmail_send — From header and signature injection
# ---------------------------------------------------------------------------

def _make_send_args(body="<p>Hello</p>", html=False, no_signature=False, cc="", thread_id=""):
    import argparse
    args = argparse.Namespace(
        to="recipient@example.com",
        subject="Test subject",
        body=body,
        cc=cc,
        html=html,
        no_signature=no_signature,
        thread_id=thread_id,
    )
    return args


class _CapturingSendService:
    """Fake Gmail service that records the raw message passed to send()."""
    def __init__(self):
        self.sent = None

    def users(self):
        return self

    def messages(self):
        return self

    def send(self, userId, body):
        self.sent = body
        return self

    def execute(self):
        return {"id": "msg-1", "threadId": "thread-1"}


def _decode_mime(raw_b64: str):
    import base64
    from email import message_from_bytes
    raw = base64.urlsafe_b64decode(raw_b64.encode())
    return message_from_bytes(raw)


def test_gmail_send_sets_from_header_and_signature(google_api_module, monkeypatch):
    alias = {"sendAsEmail": "user@example.com", "displayName": "Test User", "isPrimary": True, "signature": "<b>Sig</b>"}
    monkeypatch.setattr(google_api_module, "_get_sendas_primary", lambda: alias)

    svc = _CapturingSendService()
    monkeypatch.setattr(google_api_module, "build_service", lambda api, ver: svc)

    google_api_module.gmail_send(_make_send_args(body="<p>Hello</p>"))

    msg = _decode_mime(svc.sent["raw"])
    assert msg["from"] == '"Test User" <user@example.com>'
    assert "<b>Sig</b>" in msg.get_payload(decode=True).decode()


def test_gmail_send_no_signature_flag(google_api_module, monkeypatch):
    alias = {"sendAsEmail": "user@example.com", "displayName": "Test User", "isPrimary": True, "signature": "<b>Sig</b>"}
    monkeypatch.setattr(google_api_module, "_get_sendas_primary", lambda: alias)

    svc = _CapturingSendService()
    monkeypatch.setattr(google_api_module, "build_service", lambda api, ver: svc)

    google_api_module.gmail_send(_make_send_args(body="Plain text", no_signature=True))

    msg = _decode_mime(svc.sent["raw"])
    payload = msg.get_payload(decode=True).decode()
    assert "<b>Sig</b>" not in payload
    assert msg.get_content_type() == "text/plain"


def test_gmail_send_no_signature_when_alias_empty(google_api_module, monkeypatch):
    monkeypatch.setattr(google_api_module, "_get_sendas_primary", lambda: {})

    svc = _CapturingSendService()
    monkeypatch.setattr(google_api_module, "build_service", lambda api, ver: svc)

    google_api_module.gmail_send(_make_send_args(body="Hello"))

    msg = _decode_mime(svc.sent["raw"])
    assert msg["from"] is None
    assert msg.get_content_type() == "text/plain"
