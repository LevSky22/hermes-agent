"""OpenAI Realtime voice session used by the Discord adapter."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import deque
from typing import Any, Awaitable, Callable, Optional

from .realtime_broker import HermesRunBroker, REALTIME_TOOLS

logger = logging.getLogger(__name__)

AudioCallback = Callable[[bytes], None]
TextCallback = Callable[[str, str], Awaitable[None]]

REALTIME_URL = "wss://api.openai.com/v1/realtime"

DEFAULT_INSTRUCTIONS = """You are Léo, Harmony Movers' friendly voice assistant.
Speak naturally and briefly. You may answer only casual small talk directly.
For anything personal, current, operational, factual about company systems, or
consequential, call delegate_to_hermes. Never invent a task result. Keep track
of task IDs and use send_followup_to_hermes only when the user is clearly
continuing that same task. State approval choices clearly. The permanent
choice requires a separate second confirmation with the exact words
"always allow". Do not expose implementation details or low-level tool chatter.
"""


def pcm_48k_stereo_to_24k_mono(pcm: bytes) -> bytes:
    """Downsample Discord s16le 48 kHz stereo PCM to 24 kHz mono."""
    if not pcm:
        return b""
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 4)], dtype="<i2")
    if not samples.size:
        return b""
    stereo = samples.reshape(-1, 2).astype(np.int32)
    mono = ((stereo[:, 0] + stereo[:, 1]) // 2).astype("<i2")
    return mono[::2].tobytes()


def pcm_24k_mono_to_48k_stereo(pcm: bytes) -> bytes:
    """Upsample OpenAI s16le 24 kHz mono PCM to Discord 48 kHz stereo."""
    if not pcm:
        return b""
    import numpy as np

    mono = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2")
    if not mono.size:
        return b""
    doubled = np.repeat(mono, 2)
    stereo = np.column_stack((doubled, doubled)).astype("<i2", copy=False)
    return stereo.tobytes()


def pcm_rms(pcm: bytes) -> float:
    """Return RMS amplitude for little-endian signed 16-bit PCM."""
    if not pcm:
        return 0.0
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2")
    if not samples.size:
        return 0.0
    values = samples.astype(np.float64)
    return float(np.sqrt(np.mean(values * values)))


class DiscordRealtimeSession:
    """One bounded, reconnecting Realtime session for a Discord guild."""

    def __init__(
        self,
        *,
        api_key: str,
        broker: HermesRunBroker,
        audio_callback: AudioCallback,
        text_callback: Optional[TextCallback] = None,
        model: str = "gpt-realtime-2.1",
        voice: str = "cedar",
        reasoning_effort: str = "low",
        transcription_model: Optional[str] = "gpt-4o-mini-transcribe",
        vad_threshold: float = 0.7,
        vad_prefix_ms: int = 300,
        vad_silence_ms: int = 700,
        input_silence_threshold: int = 120,
        manual_turn_timeout_ms: int = 700,
        instructions: str = DEFAULT_INSTRUCTIONS,
        history_turns: int = 50,
        session_rotation_seconds: int = 3000,
    ):
        self.api_key = api_key
        self.broker = broker
        self.audio_callback = audio_callback
        self.text_callback = text_callback
        self.model = model
        self.voice = voice
        self.reasoning_effort = reasoning_effort
        self.transcription_model = (
            str(transcription_model).strip() if transcription_model else None
        )
        self.vad_threshold = max(0.0, min(1.0, float(vad_threshold)))
        self.vad_prefix_ms = max(0, int(vad_prefix_ms))
        self.vad_silence_ms = max(100, int(vad_silence_ms))
        self.input_silence_threshold = max(0, int(input_silence_threshold))
        self.manual_turn_timeout_ms = max(100, int(manual_turn_timeout_ms))
        self.instructions = instructions
        self.history_turns = max(4, int(history_turns))
        self.session_rotation_seconds = max(300, int(session_rotation_seconds))
        # Forty-millisecond chunks, bounded to two seconds. Socket-thread
        # producers drop the oldest chunk rather than growing unbounded.
        self._input: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        self._history: deque[tuple[str, str]] = deque(maxlen=self.history_turns)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws: Any = None
        self._runner: Optional[asyncio.Task] = None
        self._sender: Optional[asyncio.Task] = None
        self._stopping = False
        self._paused = False
        self._connected = asyncio.Event()
        self._active_response = False
        self._handled_call_ids: set[str] = set()
        self._turn_timer: Optional[asyncio.TimerHandle] = None
        self._turn_has_audio = False
        self._last_input_audio_at = 0.0
        self._sent_audio_chunks = 0
        self._received_audio_chunks = 0
        self._session_updated_logged = False
        self.last_error: Optional[str] = None

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def paused(self) -> bool:
        return self._paused

    async def start(self) -> None:
        if self._runner and not self._runner.done():
            return
        self._loop = asyncio.get_running_loop()
        self._stopping = False
        await self.broker.start()
        self._runner = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=15)
        except asyncio.TimeoutError as exc:
            await self.stop()
            raise RuntimeError(
                self.last_error or "Realtime connection timed out"
            ) from exc

    async def stop(self, *, close_broker: bool = False) -> None:
        self._stopping = True
        self._connected.clear()
        self._cancel_turn_timer()
        tasks = [task for task in (self._sender, self._runner) if task is not None]
        for task in tasks:
            task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._runner = None
        self._sender = None
        self._ws = None
        if close_broker:
            await self.broker.close()

    def pause(self) -> None:
        self._paused = True
        self._cancel_turn_timer()
        self._turn_has_audio = False
        self._clear_input()

    def resume(self) -> None:
        self._paused = False

    def feed_discord_pcm(self, user_id: int, pcm: bytes) -> bool:
        """Thread-safe PCM tap. True means classic STT should not consume it."""
        if self._stopping or self._paused or self._loop is None:
            return True
        converted = pcm_48k_stereo_to_24k_mono(pcm)
        if not converted:
            return True
        rms = pcm_rms(converted)
        if rms < self.input_silence_threshold:
            return True
        self._loop.call_soon_threadsafe(self._enqueue_input, converted)
        return True

    async def cancel_response(self) -> None:
        if self._ws is not None and self._active_response:
            await self._send({"type": "response.cancel"})
        self._active_response = False

    def _enqueue_input(self, pcm: bytes) -> None:
        self._sent_audio_chunks += 1
        if self._sent_audio_chunks <= 3 or self._sent_audio_chunks in {10, 25, 50, 100}:
            logger.info(
                "Discord Realtime input chunk #%d: bytes=%d",
                self._sent_audio_chunks,
                len(pcm),
            )
        if self._input.full():
            try:
                self._input.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._input.put_nowait(pcm)
        except asyncio.QueueFull:
            pass
        self._turn_has_audio = True
        self._last_input_audio_at = time.monotonic()
        self._schedule_turn_finalize()

    def _schedule_turn_finalize(self) -> None:
        self._cancel_turn_timer()
        if self._loop is None or self._loop.is_closed():
            return
        self._turn_timer = self._loop.call_later(
            self.manual_turn_timeout_ms / 1000.0,
            self._turn_timeout_fired,
        )

    def _cancel_turn_timer(self) -> None:
        timer = self._turn_timer
        self._turn_timer = None
        if timer is not None:
            timer.cancel()

    def _turn_timeout_fired(self) -> None:
        self._turn_timer = None
        if self._loop is None or self._loop.is_closed():
            return
        self._loop.create_task(self._finalize_input_turn())

    async def _finalize_input_turn(self) -> None:
        if self._stopping or self._paused or not self._turn_has_audio or self._ws is None:
            return
        elapsed_ms = (time.monotonic() - self._last_input_audio_at) * 1000
        if elapsed_ms < self.manual_turn_timeout_ms:
            self._schedule_turn_finalize()
            return
        # A provider-created response means server VAD already finalized this
        # turn. Avoid a duplicate commit/response in that case.
        if self._active_response:
            self._turn_has_audio = False
            return
        self._turn_has_audio = False
        logger.info(
            "Discord Realtime finalizing input turn after %dms silence",
            self.manual_turn_timeout_ms,
        )
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

    def _clear_input(self) -> None:
        while True:
            try:
                self._input.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _run(self) -> None:
        delay = 1.0
        while not self._stopping:
            try:
                await self._connect_once()
                delay = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                self._connected.clear()
                logger.warning("Discord Realtime connection failed: %s", exc)
            if not self._stopping:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20.0)

    async def _connect_once(self) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "websockets is required for Discord Realtime voice"
            ) from exc

        url = f"{REALTIME_URL}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=4 * 1024 * 1024,
        ) as ws:
            self._ws = ws
            await self._send(self._session_update())
            await self._seed_history()
            self._connected.set()
            self.last_error = None
            self._sender = asyncio.create_task(self._send_audio())
            try:
                async with asyncio.timeout(self.session_rotation_seconds):
                    async for raw in ws:
                        event = json.loads(raw)
                        await self._handle_event(event)
            except TimeoutError:
                logger.info(
                    "Rotating Discord Realtime session after %ss",
                    self.session_rotation_seconds,
                )
            finally:
                self._connected.clear()
                if self._sender is not None:
                    self._sender.cancel()
                    await asyncio.gather(self._sender, return_exceptions=True)
                    self._sender = None
                self._ws = None

    def _session_update(self) -> dict[str, Any]:
        audio_input: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {
                "type": "server_vad",
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.vad_prefix_ms,
                "silence_duration_ms": self.vad_silence_ms,
                "create_response": True,
                "interrupt_response": True,
            },
        }
        if self.transcription_model:
            audio_input["transcription"] = {"model": self.transcription_model}
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model,
                "instructions": self.instructions,
                "output_modalities": ["audio"],
                "reasoning": {"effort": self.reasoning_effort},
                "audio": {
                    "input": {
                        **audio_input,
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": self.voice,
                    },
                },
                "tools": REALTIME_TOOLS,
                "tool_choice": "auto",
            },
        }

    async def _seed_history(self) -> None:
        for role, text in self._history:
            await self._send({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text" if role == "user" else "output_text",
                            "text": text,
                        }
                    ],
                },
            })
        recent = self.broker.store.recent(self.broker.owner_key, limit=10)
        if recent:
            recap = [
                {
                    "task_id": task["task_id"],
                    "status": task["status"],
                    "output": (task.get("output") or "")[:600],
                }
                for task in recent
            ]
            await self._send({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "[Hermes task recap] " + json.dumps(recap),
                        }
                    ],
                },
            })

    async def _send_audio(self) -> None:
        while True:
            pcm = await self._input.get()
            if self._paused or not pcm:
                continue
            await self._send({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            })

    async def _handle_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        if kind == "input_audio_buffer.speech_started":
            self._active_response = False
            logger.info("Discord Realtime VAD event: %s", kind)
            if self.text_callback:
                await self.text_callback("barge_in", "")
            return
        if kind == "input_audio_buffer.speech_stopped":
            logger.info("Discord Realtime VAD event: %s", kind)
            return
        if kind == "session.updated":
            if not self._session_updated_logged:
                self._session_updated_logged = True
                logger.info(
                    "Discord Realtime session updated: model=%s voice=%s",
                    self.model,
                    self.voice,
                )
            return
        if kind in {"response.created", "response.output_item.added"}:
            self._active_response = True
            self._turn_has_audio = False
            self._cancel_turn_timer()
            logger.info("Discord Realtime response event: %s", kind)
            return
        if kind == "conversation.item.input_audio_transcription.completed":
            text = str(event.get("transcript") or "").strip()
            if text:
                self.broker.record_user_utterance(text)
                self._history.append(("user", text))
                if self.text_callback:
                    await self.text_callback("user", text)
            return
        if kind in {"response.output_audio.delta", "response.audio.delta"}:
            delta = event.get("delta") or ""
            if delta:
                self._active_response = True
                try:
                    pcm = base64.b64decode(delta)
                except (ValueError, TypeError):
                    pcm = b""
                if pcm:
                    self._received_audio_chunks += 1
                    if self._received_audio_chunks <= 3 or self._received_audio_chunks in {10, 25, 50, 100}:
                        logger.info(
                            "Discord Realtime output chunk #%d: bytes=%d",
                            self._received_audio_chunks,
                            len(pcm),
                        )
                    self.audio_callback(pcm_24k_mono_to_48k_stereo(pcm))
            return
        if kind in {
            "response.output_audio_transcript.done",
            "response.audio_transcript.done",
        }:
            text = str(event.get("transcript") or "").strip()
            if text:
                self._history.append(("assistant", text))
                if self.text_callback:
                    await self.text_callback("assistant", text)
            return
        if kind == "response.function_call_arguments.done":
            await self._handle_tool_call(event)
            return
        if kind == "response.done":
            self._active_response = False
            response = event.get("response") or {}
            for item in response.get("output") or []:
                if item.get("type") == "function_call":
                    await self._handle_tool_call(item)
            return
        if kind == "error":
            error = event.get("error") or event
            self.last_error = str(
                error.get("message") if isinstance(error, dict) else error
            )
            logger.warning("OpenAI Realtime error: %s", self.last_error)

    async def _handle_tool_call(self, event: dict[str, Any]) -> None:
        name = str(event.get("name") or "")
        call_id = str(event.get("call_id") or event.get("item_id") or "")
        if not name or not call_id:
            return
        if call_id in self._handled_call_ids:
            return
        self._handled_call_ids.add(call_id)
        raw_args = event.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (json.JSONDecodeError, TypeError, ValueError):
            args = {}
        result = await self.broker.handle_tool(name, args, call_id)
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        })
        await self._send({"type": "response.create"})

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime WebSocket is not connected")
        await self._ws.send(json.dumps(payload))
