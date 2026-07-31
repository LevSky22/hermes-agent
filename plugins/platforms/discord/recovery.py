"""Durable state for Discord reconnect message recovery."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sqlite3
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_FILENAME = "discord_message_recovery.db"
_RETENTION_DAYS = 30
_HANDOFF_TEXT_MAX_CHARS = 6000
_HANDOFF_CONTEXT_MAX_CHARS = 2000


class DiscordRecoveryStore:
    """Small profile-scoped SQLite ledger for completed Discord messages."""

    def __init__(self, hermes_home: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._hermes_home = Path(hermes_home or get_hermes_home())

    def path(self) -> Path:
        directory = self._hermes_home / "gateway"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / _DB_FILENAME

    def call(self, fn: Callable[[sqlite3.Connection], Any], default: Any = None) -> Any:
        try:
            with self._lock:
                path = self.path()
                conn = sqlite3.connect(path, timeout=0.1)
                try:
                    if not self._initialized:
                        self._initialize(conn)
                        self._initialized = True
                        with suppress(OSError):
                            os.chmod(path, 0o600)
                    result = fn(conn)
                    conn.commit()
                    return result
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("Discord recovery ledger unavailable: %s", exc)
            return default

    def _initialize(self, conn: sqlite3.Connection) -> None:
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="discord_recovery.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS discord_messages (
                message_id TEXT PRIMARY KEY,
                channel_id TEXT,
                thread_id TEXT,
                parent_channel_id TEXT,
                author_id TEXT,
                created_at TEXT,
                status TEXT NOT NULL,
                replied INTEGER NOT NULL DEFAULT 0,
                emoji_ack INTEGER NOT NULL DEFAULT 0,
                outage_response INTEGER NOT NULL DEFAULT 0,
                response_message_id TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt_at TEXT,
                last_error TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS discord_recovery_scans (
                scan_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                channels TEXT NOT NULL,
                window_seconds REAL NOT NULL,
                limit_count INTEGER NOT NULL,
                scanned INTEGER NOT NULL DEFAULT 0,
                missed INTEGER NOT NULL DEFAULT 0,
                dispatched INTEGER NOT NULL DEFAULT 0,
                error TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS discord_recovery_cursors (
                channel_id TEXT PRIMARY KEY,
                last_message_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS discord_message_handoffs (
                message_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                message_text TEXT NOT NULL,
                context_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)
        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=_RETENTION_DAYS)
        ).isoformat()
        conn.execute("DELETE FROM discord_messages WHERE updated_at < ?", (cutoff,))
        conn.execute(
            "DELETE FROM discord_recovery_scans "
            "WHERE COALESCE(completed_at, started_at) < ?",
            (cutoff,),
        )
        conn.execute(
            "DELETE FROM discord_recovery_cursors WHERE updated_at < ?",
            (cutoff,),
        )
        conn.execute(
            "DELETE FROM discord_message_handoffs WHERE created_at < ?",
            (cutoff,),
        )

    def record_message_handoff(
        self,
        *,
        channel_id: str,
        message_id: str,
        message_text: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Persist a bounded continuation anchor for one outbound message.

        The record contains only the delivered text and caller-supplied routing
        identifiers. It deliberately does not copy the originating agent
        transcript or webhook payload.
        """
        if not channel_id or not message_id or not message_text:
            return False
        bounded_text = str(message_text)[:_HANDOFF_TEXT_MAX_CHARS]
        try:
            context_json = json.dumps(
                context if isinstance(context, dict) else {},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            context_json = "{}"
        if len(context_json) > _HANDOFF_CONTEXT_MAX_CHARS:
            context_json = "{}"
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()

        def _op(conn: sqlite3.Connection) -> bool:
            conn.execute(
                """
                INSERT INTO discord_message_handoffs
                    (message_id, channel_id, message_text, context_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    channel_id=excluded.channel_id,
                    message_text=excluded.message_text,
                    context_json=excluded.context_json,
                    created_at=excluded.created_at
                """,
                (str(message_id), str(channel_id), bounded_text, context_json, created_at),
            )
            return True

        return bool(self.call(_op, False))

    def get_message_handoff(
        self, *, channel_id: str, message_id: str
    ) -> dict[str, Any] | None:
        """Return a continuation anchor only from its original channel."""
        if not channel_id or not message_id:
            return None

        def _op(conn: sqlite3.Connection):
            return conn.execute(
                """
                SELECT message_text, context_json, created_at
                FROM discord_message_handoffs
                WHERE message_id=? AND channel_id=?
                """,
                (str(message_id), str(channel_id)),
            ).fetchone()

        row = self.call(_op)
        if not row:
            return None
        try:
            context = json.loads(row[1]) if row[1] else {}
        except (TypeError, ValueError):
            context = {}
        if not isinstance(context, dict):
            context = {}
        return {
            "message_text": row[0],
            "context": context,
            "created_at": row[2],
        }
