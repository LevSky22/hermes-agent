import asyncio
import json
import time
from array import array
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.realtime_broker import HermesRunBroker, REALTIME_TOOLS
from plugins.platforms.discord.realtime_voice import (
    DiscordRealtimeSession,
    pcm_24k_mono_to_48k_stereo,
    pcm_48k_stereo_to_24k_mono,
    pcm_rms,
)
from plugins.platforms.discord.voice_mixer import (
    FRAME_SIZE,
    RealtimePCMQueueAudioSource,
    VoiceMixer,
)


def test_realtime_exposes_exactly_five_narrow_tools():
    assert [tool["name"] for tool in REALTIME_TOOLS] == [
        "delegate_to_hermes",
        "get_hermes_task_status",
        "send_followup_to_hermes",
        "cancel_hermes_task",
        "approve_hermes_action",
    ]


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
