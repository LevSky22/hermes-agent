import asyncio
import base64
import json
import time
from array import array
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.realtime_broker import HermesRunBroker, REALTIME_TOOLS
from plugins.platforms.discord.realtime_voice import (
    DiscordRealtimeSession,
    REALTIME_INPUT_FRAME_BYTES,
    load_realtime_identity_context,
    pcm_24k_mono_to_48k_stereo,
    pcm_48k_stereo_to_24k_mono,
    pcm_rms,
)
from plugins.platforms.discord.voice_mixer import (
    FRAME_SIZE,
    RealtimePCMQueueAudioSource,
    VoiceMixer,
)


def test_realtime_exposes_only_narrow_bridge_tools():
    assert [tool["name"] for tool in REALTIME_TOOLS] == [
        "delegate_to_hermes",
        "remember_user_preference",
        "get_hermes_task_status",
        "send_followup_to_hermes",
        "cancel_hermes_task",
        "approve_hermes_action",
        "wait_for_user",
    ]


@pytest.mark.asyncio
async def test_realtime_becomes_ready_only_after_session_updated():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
    )
    ws = AsyncMock()
    ws.recv.side_effect = [
        json.dumps({"type": "session.created"}),
        json.dumps({"type": "session.updated", "session": {}}),
    ]

    assert not session.connected
    await session._wait_until_session_updated(ws)
    assert session.connected
    assert ws.recv.await_count == 2


def test_pcm_conversion_preserves_duration_and_channels():
    discord_pcm = array("h", range(480 * 2)).tobytes()
    realtime_pcm = pcm_48k_stereo_to_24k_mono(discord_pcm)
    assert len(realtime_pcm) == 240 * 2
    round_trip = pcm_24k_mono_to_48k_stereo(realtime_pcm)
    assert len(round_trip) == 480 * 2 * 2


def test_pcm_rms_distinguishes_silence_and_speech():
    assert pcm_rms(b"\x00\x00" * 240) == 0
    assert pcm_rms(array("h", [1000] * 240).tobytes()) == 1000


@pytest.mark.asyncio
async def test_realtime_gates_idle_silence_but_forwards_vad_tail():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
        vad_prefix_ms=40,
        vad_silence_ms=100,
    )
    session._loop = asyncio.get_running_loop()
    session._ws = AsyncMock()
    silence = b"\x00\x00" * 1920
    speech = array("h", [1000] * 1920).tobytes()

    session.feed_discord_pcm(1, silence)
    await asyncio.sleep(0)
    assert session._input.empty()

    session.feed_discord_pcm(1, speech)
    await asyncio.sleep(0)
    # One 20ms pre-roll frame plus the speech frame entered the send queue.
    assert session._input.qsize() == 2

    session._last_loud_input_at = time.monotonic() - 0.2
    session.feed_discord_pcm(1, silence)
    await asyncio.sleep(0)
    assert session._input.qsize() == 3
    assert not session._input_gate_open
    # Server VAD owns commit and response creation; the client only appends.
    assert session._ws.send.await_count == 0


@pytest.mark.asyncio
async def test_realtime_synthesizes_silence_when_discord_packets_stop():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
        vad_silence_ms=2000,
    )
    session._ws = AsyncMock()
    session._input_gate_open = True
    session._user_speaking = True
    session._last_loud_input_at = time.monotonic()

    sender = asyncio.create_task(session._send_audio())
    await asyncio.sleep(0.07)
    sender.cancel()
    await asyncio.gather(sender, return_exceptions=True)

    payloads = [json.loads(call.args[0]) for call in session._ws.send.await_args_list]
    appended = [
        payload
        for payload in payloads
        if payload["type"] == "input_audio_buffer.append"
    ]
    assert len(appended) >= 2
    assert all(
        len(base64.b64decode(payload["audio"])) == REALTIME_INPUT_FRAME_BYTES
        for payload in appended
    )


@pytest.mark.asyncio
async def test_realtime_speech_stopped_closes_synthetic_silence_gate():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
    )
    session._input_gate_open = True
    session._user_speaking = True
    session._last_loud_input_at = time.monotonic()

    await session._handle_event({"type": "input_audio_buffer.speech_stopped"})

    assert not session._user_speaking
    assert not session._input_gate_open
    assert session._last_loud_input_at == 0.0


def test_realtime_audio_source_ends_after_idle_grace():
    frame = b"\x01\x00" * (FRAME_SIZE // 2)
    source = RealtimePCMQueueAudioSource(frame)
    assert source.read() == frame
    source._last_feed_at = time.monotonic() - 1
    assert source.read() == b""
    assert source.closed


def test_realtime_audio_source_flushes_partial_frame_once():
    source = RealtimePCMQueueAudioSource(b"\x01\x00" * 100)
    source._last_feed_at = time.monotonic() - 1
    chunk = source.read()
    assert len(chunk) == FRAME_SIZE
    assert chunk.startswith(b"\x01\x00" * 100)
    assert source.read() == b""


def test_realtime_playback_guard_covers_active_source_and_tail():
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._realtime_playback_sources = {1: MagicMock(closed=False)}
    adapter._realtime_playback_guard_until = {}
    assert adapter._is_realtime_playback_guarded(1)

    adapter._realtime_playback_sources[1].closed = True
    adapter._realtime_playback_guard_until[1] = time.monotonic() + 1
    assert adapter._is_realtime_playback_guarded(1)

    adapter._realtime_playback_guard_until[1] = time.monotonic() - 1
    assert not adapter._is_realtime_playback_guarded(1)


@pytest.mark.asyncio
async def test_local_speech_gate_interrupts_playback_once():
    callback = AsyncMock()
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
        text_callback=callback,
    )
    session._loop = asyncio.get_running_loop()
    session._active_response = True
    speech = array("h", [1000] * 1920).tobytes()

    session.feed_discord_pcm(1, speech)
    session.feed_discord_pcm(1, speech)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    callback.assert_awaited_once_with("barge_in", "")


def test_voice_mixer_stream_is_bounded_and_clears_on_overrun():
    pytest.importorskip("numpy")
    mixer = VoiceMixer()
    frame = b"\x01\x00" * (FRAME_SIZE // 2)
    for _ in range(mixer._stream_max_frames):
        assert mixer.append_streaming_speech(frame)
    assert not mixer.append_streaming_speech(frame)
    assert not mixer.speech_active
    assert mixer.read() == b"\x00" * FRAME_SIZE


def test_realtime_session_payload_can_disable_caption_transcription():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
        transcription_model=None,
    )
    payload = session._session_update()["session"]
    assert payload["model"] == "gpt-realtime-2.1"
    assert payload["audio"]["output"]["voice"] == "cedar"
    assert "transcription" not in payload["audio"]["input"]
    assert payload["tools"] == REALTIME_TOOLS


def test_realtime_session_payload_supports_semantic_vad():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
        vad_type="semantic_vad",
        vad_eagerness="high",
    )
    turn_detection = session._session_update()["session"]["audio"]["input"][
        "turn_detection"
    ]
    assert turn_detection == {
        "type": "semantic_vad",
        "eagerness": "high",
        "create_response": True,
        "interrupt_response": True,
    }


def test_realtime_identity_uses_canonical_soul_and_memory(monkeypatch):
    store = MagicMock()
    store.format_for_system_prompt.side_effect = lambda target: {
        "memory": "MEMORY (your personal notes)\nUses a project tracker.",
        "user": "USER PROFILE (who the user is)\nPrefers audio summaries.",
    }[target]
    monkeypatch.setattr(
        "agent.prompt_builder.load_soul_md",
        lambda: "Use a calm, professional tone.",
    )
    monkeypatch.setattr(
        "agent.prompt_builder.build_context_files_prompt",
        lambda **_kwargs: "# Project Context\n\nFollow the workspace contract.",
    )
    monkeypatch.setattr(
        "agent.runtime_cwd.resolve_context_cwd",
        lambda: "/opt/data",
    )
    monkeypatch.setattr("tools.memory_tool.load_on_disk_store", lambda: store)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"memory_enabled": True, "user_profile_enabled": True}},
    )

    context = load_realtime_identity_context()
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
        identity_context=context,
    )

    assert "# Canonical identity (SOUL.md)" in session.instructions
    assert "Use a calm, professional tone." in session.instructions
    assert "Follow the workspace contract." in session.instructions
    assert "Uses a project tracker." in session.instructions
    assert "Prefers audio summaries." in session.instructions
    assert "Do not mention models, internal agents" in session.instructions


def test_realtime_fallback_identity_is_public_and_generic():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
    )

    assert "neutral, helpful voice assistant" in session.instructions


@pytest.mark.asyncio
async def test_realtime_function_call_uses_broker_and_returns_output():
    broker = MagicMock()
    broker.handle_tool = AsyncMock(return_value={"ok": True, "task_id": "rtask_1"})
    session = DiscordRealtimeSession(
        api_key="key",
        broker=broker,
        audio_callback=lambda _pcm: None,
    )
    session._ws = AsyncMock()
    await session._handle_event({
        "type": "response.function_call_arguments.done",
        "name": "delegate_to_hermes",
        "call_id": "call_1",
        "arguments": '{"request":"send the email"}',
    })
    broker.handle_tool.assert_awaited_once_with(
        "delegate_to_hermes",
        {"request": "send the email"},
        "call_1",
    )
    sent = [json.loads(call.args[0]) for call in session._ws.send.await_args_list]
    assert sent[0]["item"]["type"] == "function_call_output"
    assert sent[1] == {"type": "response.create"}


@pytest.mark.asyncio
async def test_wait_for_user_ends_turn_without_another_response():
    broker = MagicMock()
    broker.handle_tool = AsyncMock(return_value={"ok": True, "status": "waiting"})
    session = DiscordRealtimeSession(
        api_key="key",
        broker=broker,
        audio_callback=lambda _pcm: None,
    )
    session._ws = AsyncMock()

    await session._handle_event({
        "type": "response.function_call_arguments.done",
        "name": "wait_for_user",
        "call_id": "call_wait",
        "arguments": "{}",
    })

    sent = [json.loads(call.args[0]) for call in session._ws.send.await_args_list]
    assert len(sent) == 1
    assert sent[0]["item"]["type"] == "function_call_output"


@pytest.mark.asyncio
async def test_completed_task_is_announced_once_when_idle():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
    )
    session._ws = AsyncMock()
    task = {"task_id": "rtask_12345678", "status": "completed", "output": "Email sent."}

    await session.notify_task_status(task)
    await session.notify_task_status(task)

    sent = [json.loads(call.args[0]) for call in session._ws.send.await_args_list]
    assert len(sent) == 2
    assert sent[0]["type"] == "conversation.item.create"
    assert "Email sent." in sent[0]["item"]["content"][0]["text"]
    assert sent[1] == {"type": "response.create"}


@pytest.mark.asyncio
async def test_running_task_is_not_announced():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
    )
    session._ws = AsyncMock()

    await session.notify_task_status({"task_id": "rtask_1", "status": "running"})

    session._ws.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_progress_update_is_announced_without_internal_tool_details():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
    )
    session._ws = AsyncMock()

    await session.notify_task_status({
        "task_id": "rtask_12345678",
        "status": "running",
        "progress": "Still working on it.",
        "progress_seq": 1,
    })

    sent = [json.loads(call.args[0]) for call in session._ws.send.await_args_list]
    assert len(sent) == 2
    update = sent[0]["item"]["content"][0]["text"]
    assert "still in progress" in update
    assert "tool" not in update.lower()


@pytest.mark.asyncio
async def test_terminal_update_supersedes_queued_progress():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
    )
    session._ws = AsyncMock()
    session._active_response = True

    await session.notify_task_status({
        "task_id": "rtask_12345678",
        "status": "running",
        "progress": "Still working on it.",
        "progress_seq": 1,
    })
    await session.notify_task_status({
        "task_id": "rtask_12345678",
        "status": "completed",
        "output": "Finished successfully.",
    })
    session._active_response = False
    await session._flush_task_announcements()

    sent = [json.loads(call.args[0]) for call in session._ws.send.await_args_list]
    assert len(sent) == 2
    update = sent[0]["item"]["content"][0]["text"]
    assert "Finished successfully." in update
    assert "still in progress" not in update


@pytest.mark.asyncio
async def test_explicit_status_call_discards_queued_progress_announcement():
    broker = MagicMock()
    broker.handle_tool = AsyncMock(
        return_value={"ok": True, "task_id": "rtask_12345678", "status": "running"}
    )
    session = DiscordRealtimeSession(
        api_key="key",
        broker=broker,
        audio_callback=lambda _pcm: None,
    )
    session._ws = AsyncMock()
    session._active_response = True
    await session.notify_task_status({
        "task_id": "rtask_12345678",
        "status": "running",
        "progress": "Still working on it.",
        "progress_seq": 1,
    })
    assert len(session._pending_task_announcements) == 1

    await session._handle_tool_call({
        "name": "get_hermes_task_status",
        "call_id": "call_status",
        "arguments": '{"task_id":"rtask_12345678"}',
    })

    assert not session._pending_task_announcements


@pytest.mark.asyncio
async def test_task_announcement_waits_for_active_response():
    session = DiscordRealtimeSession(
        api_key="key",
        broker=MagicMock(),
        audio_callback=lambda _pcm: None,
    )
    session._ws = AsyncMock()
    session._active_response = True

    await session.notify_task_status({
        "task_id": "rtask_87654321",
        "status": "waiting_for_approval",
        "approval_id": "approval_1",
        "approval": {"description": "Send the email"},
    })
    session._ws.send.assert_not_awaited()

    await session._handle_event({"type": "response.done", "response": {"output": []}})

    sent = [json.loads(call.args[0]) for call in session._ws.send.await_args_list]
    assert len(sent) == 2
    assert "needs approval" in sent[0]["item"]["content"][0]["text"]
    assert sent[1] == {"type": "response.create"}


@pytest.mark.asyncio
async def test_broker_call_id_is_idempotent(tmp_path):
    broker = HermesRunBroker(
        owner_key="discord:1:2:3",
        api_key="test-key",
        store_path=tmp_path / "realtime.sqlite3",
    )
    broker.delegate = AsyncMock(
        return_value={"ok": True, "task_id": "rtask_one", "status": "queued"}
    )
    try:
        first = await broker.handle_tool(
            "delegate_to_hermes", {"request": "do it"}, "call_1"
        )
        second = await broker.handle_tool(
            "delegate_to_hermes", {"request": "do it"}, "call_1"
        )
    finally:
        await broker.close()
    assert first == second
    broker.delegate.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_emits_sanitized_progress_update(tmp_path):
    callback = AsyncMock()
    broker = HermesRunBroker(
        owner_key="owner",
        api_key="key",
        store_path=tmp_path / "progress.sqlite3",
        status_callback=callback,
    )
    now = time.time()
    broker.store.put_task({
        "task_id": "rtask_progress",
        "owner_key": "owner",
        "prompt": "do work",
        "status": "running",
        "run_id": "run_1",
        "session_id": "session_1",
        "output": None,
        "error": None,
        "approval_id": None,
        "approval_json": None,
        "created_at": now,
        "updated_at": now,
    })
    try:
        await broker._notify_progress("rtask_progress")
    finally:
        await broker.close()

    update = callback.await_args.args[0]
    assert update["status"] == "running"
    assert update["progress"] == "Still working on it."
    assert update["progress_seq"] == 1
    assert "tool" not in update


@pytest.mark.asyncio
async def test_broker_status_check_does_not_delay_scheduled_progress(tmp_path):
    callback = AsyncMock()
    broker = HermesRunBroker(
        owner_key="owner",
        api_key="key",
        store_path=tmp_path / "status-progress.sqlite3",
        status_callback=callback,
        progress_interval_seconds=30,
    )
    now = time.time()
    broker.store.put_task({
        "task_id": "rtask_progress",
        "owner_key": "owner",
        "prompt": "do work",
        "status": "running",
        "run_id": "run_1",
        "session_id": "session_1",
        "output": None,
        "error": None,
        "approval_id": None,
        "approval_json": None,
        "created_at": now,
        "updated_at": now,
    })
    try:
        result = await broker.status("rtask_progress")
        await broker._notify_progress("rtask_progress")
    finally:
        await broker.close()

    assert result["status"] == "running"
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_routes_user_preference_to_durable_user_memory(tmp_path):
    broker = HermesRunBroker(
        owner_key="owner",
        api_key="key",
        store_path=tmp_path / "preference.sqlite3",
    )
    broker.delegate = AsyncMock(
        return_value={"ok": True, "task_id": "rtask_pref", "status": "queued"}
    )
    try:
        result = await broker.handle_tool(
            "remember_user_preference",
            {"preference": "Do not read URLs aloud in voice conversations."},
            "call_pref",
        )
    finally:
        await broker.close()

    assert result["ok"] is True
    request = broker.delegate.await_args.args[0]
    assert "USER memory" in request
    assert "Do not read URLs aloud in voice conversations." in request
    assert "Do not modify SOUL.md or any skill" in request


@pytest.mark.asyncio
async def test_broker_rejects_cross_owner_task(tmp_path):
    first = HermesRunBroker(
        owner_key="owner-a", api_key="key", store_path=tmp_path / "shared.sqlite3"
    )
    now = time.time()
    first.store.put_task({
        "task_id": "rtask_private",
        "owner_key": "owner-a",
        "prompt": "private",
        "status": "completed",
        "run_id": None,
        "session_id": "s1",
        "output": "secret result",
        "error": None,
        "approval_id": None,
        "approval_json": None,
        "created_at": now,
        "updated_at": now,
    })
    second = HermesRunBroker(
        owner_key="owner-b", api_key="key", store_path=tmp_path / "shared.sqlite3"
    )
    try:
        result = await second.status("rtask_private")
    finally:
        await second.close()
        await first.close()
    assert result == {"ok": False, "error": "task_not_found"}


@pytest.mark.asyncio
async def test_always_approval_requires_exact_second_user_turn(tmp_path):
    broker = HermesRunBroker(
        owner_key="owner", api_key="key", store_path=tmp_path / "approval.sqlite3"
    )
    now = time.time()
    broker.store.put_task({
        "task_id": "rtask_approval",
        "owner_key": "owner",
        "prompt": "send mail",
        "status": "waiting_for_approval",
        "run_id": "run_1",
        "session_id": "s1",
        "output": None,
        "error": None,
        "approval_id": "approval_1",
        "approval_json": "{}",
        "created_at": now,
        "updated_at": now,
    })
    broker._request = AsyncMock(return_value={"resolved": 1})
    try:
        first = await broker.approve(
            "rtask_approval", "approval_1", "always", "always allow"
        )
        broker.record_user_utterance("yes, always")
        second = await broker.approve(
            "rtask_approval", "approval_1", "always", "always allow"
        )
        broker.record_user_utterance("always allow")
        third = await broker.approve("rtask_approval", "approval_1", "always", "")
    finally:
        await broker.close()
    assert first["error"] == "always_requires_second_turn_confirmation"
    assert second["error"] == "always_confirmation_not_heard_from_user"
    assert third["ok"] is True
    broker._request.assert_awaited_once()
