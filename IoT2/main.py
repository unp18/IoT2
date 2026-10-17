"""
IoT Smart Speaker backend

Pipeline:  audio/text in -> Whisper (Groq) -> LLM (Groq, streaming, tools)
           -> sentence-level edge-tts -> ffmpeg -> 16 kHz/16-bit/mono PCM -> ESP32

WebSocket:  /ws/{device_id}?role=device|client&token=...

Client -> server (JSON text frames)
  {"type":"text","text":"...","conversation_id":"<optional>"}
  {"type":"audio_meta","mime":"audio/webm","conversation_id":"<optional>"}   then ONE binary frame
  {"type":"set_output","output":"device"|"client"}
  {"type":"stop"}
  {"type":"ping"}

Device -> server
  {"type":"telemetry","temp":24.1,"hum":55}
  {"type":"audio_meta","mime":"audio/pcm"}  then ONE binary frame of raw PCM (future mic)

Server -> everyone
  {"type":"status","state":"idle|transcribing|thinking|speaking"}

Server -> clients
  {"type":"hello", ...}              {"type":"presence","device_online":bool}
  {"type":"conversation","conversation":{id,title}}
  {"type":"message","role":"user|assistant","id","text","conversation_id","source","created_at"}
  {"type":"message_start","role":"assistant","conversation_id"}
  {"type":"token","conversation_id","text"}
  {"type":"telemetry","temp","hum"}  {"type":"output","output":"device|client"}
  {"type":"info"|"error"|"warning","text":"..."}

Server -> audio target (device or client, depending on set_output)
  {"type":"audio_start","rate":16000,"bits":16,"ch":1}
  <binary PCM chunks, 4096 bytes each>
  {"type":"audio_end"}
  {"type":"stop"}   -> flush buffers and stop playback immediately
"""

import asyncio
import hmac
import json
import logging
import os
import re
import shutil
import time
import uuid
import wave
import io
from datetime import datetime, timezone
from typing import Optional

import edge_tts
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pydantic import BaseModel
from sqlalchemy import DateTime, Float, ForeignKey, String, Text, create_engine, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("speaker")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")
DEVICE_TOKEN = os.getenv("DEVICE_TOKEN", "")          # required for role=device
CLIENT_TOKEN = os.getenv("CLIENT_TOKEN", "")          # optional for role=client / REST
FRONTEND_ORIGINS = [o.strip() for o in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AvaNeural")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))
TURN_TIMEOUT = int(os.getenv("TURN_TIMEOUT", "120"))   # hard cap per turn so a stuck API call can never wedge the room
SESSION_REUSE_SECONDS = 30 * 60   # device voice turns continue the last chat if recent

SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2   # 16-bit mono
CHUNK_BYTES = 4096                # 128 ms of audio
LEAD_SECONDS = 1.0                # never get more than ~1 s ahead of real-time playback
SENSOR_LOG_INTERVAL = 60          # seconds between telemetry DB writes

SYSTEM_PROMPT = (
    "You are a smart speaker assistant. Your replies are spoken aloud, so keep them brief, "
    "natural and conversational. Do not use markdown, lists, emojis or code blocks. "
    "Use the get_dht_sensor_data tool when asked about room temperature, humidity or climate."
)

HALLUCINATIONS = {"", "you", "thank you", "thanks", "thanks for watching", "bye", "okay"}

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY is not set")
if not DEVICE_TOKEN:
    log.warning("DEVICE_TOKEN is not set: devices will be rejected")
if not shutil.which("ffmpeg"):
    log.warning("ffmpeg not found on PATH: TTS audio conversion will fail")

groq = AsyncGroq(api_key=GROQ_API_KEY)

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:               # SQLite returns naive datetimes (stored as UTC)
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    device_id: Mapped[str] = mapped_column(String(50), index=True)
    title: Mapped[str] = mapped_column(String(100), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(12))        # user | assistant
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(10), default="text")   # text | voice
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class SensorLog(Base):
    __tablename__ = "sensor_logs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(50), index=True)
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_db(fn, *args, **kwargs):
    with SessionLocal() as db:
        result = fn(db, *args, **kwargs)
        db.commit()
        return result


async def adb(fn, *args, **kwargs):
    """Run a sync DB function in a worker thread so the event loop never blocks."""
    return await asyncio.to_thread(_run_db, fn, *args, **kwargs)


def db_ensure_conversation(db: Session, device_id: str, conversation_id: Optional[str]) -> dict:
    conv = db.get(Conversation, conversation_id) if conversation_id else None
    if conv is None or conv.device_id != device_id:
        conv = Conversation(id=new_id(), device_id=device_id, title="New chat")
        db.add(conv)
        db.flush()
    return {"id": conv.id, "title": conv.title}


def db_add_message(db: Session, conversation_id: str, role: str, content: str, source: str) -> dict:
    msg = Message(id=new_id(), conversation_id=conversation_id, role=role, content=content, source=source)
    db.add(msg)
    conv = db.get(Conversation, conversation_id)
    conv.updated_at = utcnow()
    if role == "user" and conv.title == "New chat":
        conv.title = (content.strip()[:48] or "New chat")
    db.flush()
    return {"id": msg.id, "created_at": iso(msg.created_at), "title": conv.title}


def db_history(db: Session, conversation_id: str, limit: int) -> list[dict]:
    rows = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    ).all()
    return [{"role": m.role, "content": m.content} for m in reversed(rows)]


def db_log_sensor(db: Session, device_id: str, temp, hum):
    db.add(SensorLog(device_id=device_id, temperature=temp, humidity=hum))


# -----------------------------------------------------------------------------
# Rooms / connections
# -----------------------------------------------------------------------------
class Room:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.devices: set[WebSocket] = set()
        self.clients: set[WebSocket] = set()
        self.output = "device"                       # where TTS audio goes
        self.telemetry: dict = {"temp": None, "hum": None, "ts": 0.0}
        self.status = "idle"
        self.task: Optional[asyncio.Task] = None
        self.active_conversation: Optional[str] = None
        self.last_activity = 0.0
        self.last_sensor_log = 0.0

    def audio_targets(self) -> set[WebSocket]:
        return self.devices if self.output == "device" else self.clients

    async def _send(self, targets: set[WebSocket], *, obj=None, data: Optional[bytes] = None):
        dead = []
        for ws in list(targets):
            try:
                if data is not None:
                    await ws.send_bytes(data)
                else:
                    await ws.send_text(json.dumps(obj))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.devices.discard(ws)
            self.clients.discard(ws)

    async def to_clients(self, obj: dict):
        await self._send(self.clients, obj=obj)

    async def to_devices(self, obj: dict):
        await self._send(self.devices, obj=obj)

    async def to_audio_target(self, obj: Optional[dict] = None, data: Optional[bytes] = None):
        await self._send(self.audio_targets(), obj=obj, data=data)

    async def to_everyone(self, obj: dict):
        await self._send(self.clients, obj=obj)
        await self._send(self.devices, obj=obj)


class Manager:
    def __init__(self):
        self.rooms: dict[str, Room] = {}

    def get(self, device_id: str) -> Room:
        if device_id not in self.rooms:
            self.rooms[device_id] = Room(device_id)
        return self.rooms[device_id]


manager = Manager()


async def set_status(room: Room, state: str):
    room.status = state
    await room.to_everyone({"type": "status", "state": state})


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_dht_sensor_data",
            "description": "Get the current room temperature (Celsius) and humidity from the ESP32 DHT sensor.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def run_tool(room: Room, name: str) -> str:
    if name == "get_dht_sensor_data":
        t = room.telemetry
        if t["temp"] is None:
            return json.dumps({"error": "No sensor reading available yet"})
        return json.dumps(
            {
                "temperature_celsius": t["temp"],
                "humidity_percent": t["hum"],
                "reading_age_seconds": int(time.time() - t["ts"]),
            }
        )
    return json.dumps({"error": f"Unknown tool {name}"})


# -----------------------------------------------------------------------------
# Audio helpers: STT / TTS / PCM
# -----------------------------------------------------------------------------
MIME_EXT = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "m4a",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


def pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


async def transcribe(audio: bytes, mime: str) -> str:
    mime = (mime or "audio/webm").split(";")[0].strip().lower()
    if mime in ("audio/pcm", "audio/l16", "audio/raw"):
        audio, mime = pcm_to_wav(audio), "audio/wav"
    ext = MIME_EXT.get(mime, "webm")
    result = await groq.audio.transcriptions.create(
        file=(f"input.{ext}", audio, mime),
        model=STT_MODEL,
        language="en",
        response_format="text",
    )
    text = result if isinstance(result, str) else getattr(result, "text", "")
    text = text.strip()
    norm = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return "" if norm in HALLUCINATIONS else text


def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`#>~|]", "", text)
    return re.sub(r"\s+", " ", text).strip()


async def mp3_to_pcm(mp3: bytes) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pcm, err = await proc.communicate(mp3)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='ignore')[:200]}")
    return pcm


async def synthesize(text: str) -> bytes:
    text = clean_for_speech(text)
    if not text:
        return b""
    mp3 = bytearray()
    async for chunk in edge_tts.Communicate(text, TTS_VOICE).stream():
        if chunk["type"] == "audio":
            mp3.extend(chunk["data"])
    return await mp3_to_pcm(bytes(mp3)) if mp3 else b""


class Pacer:
    """Keeps us at most LEAD_SECONDS ahead of real-time playback so the ESP32 ring buffer never overflows."""

    def __init__(self):
        self.t0: Optional[float] = None
        self.sent = 0.0

    async def wait(self, nbytes: int):
        now = time.monotonic()
        if self.t0 is None or (now - self.t0) > self.sent:   # first chunk or buffer underrun: reset clock
            self.t0, self.sent = now, 0.0
        self.sent += nbytes / BYTES_PER_SEC
        ahead = self.sent - (time.monotonic() - self.t0)
        if ahead > LEAD_SECONDS:
            await asyncio.sleep(ahead - LEAD_SECONDS)


class SentenceSplitter:
    """Turns a token stream into speakable sentences."""

    END = re.compile(r"(?<=[.!?])\s+|\n+")
    MIN_CHARS = 20

    def __init__(self):
        self.buf = ""

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            m = next((m for m in self.END.finditer(self.buf) if m.start() >= self.MIN_CHARS), None)
            if not m:
                break
            out.append(self.buf[: m.start()].strip())
            self.buf = self.buf[m.end():]
        return [s for s in out if s]

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


# -----------------------------------------------------------------------------
# Conversation pipeline
# -----------------------------------------------------------------------------
async def speaker(room: Room, queue: asyncio.Queue):
    """Consumes sentences, synthesizes them, and streams PCM to the audio target."""
    started = False
    pacer = Pacer() if room.output == "device" else None
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        try:
            pcm = await synthesize(sentence)
        except Exception:
            log.exception("TTS failed for sentence")
            continue
        if not pcm:
            continue
        if not started:
            started = True
            await set_status(room, "speaking")
            await room.to_audio_target(
                {"type": "audio_start", "rate": SAMPLE_RATE, "bits": 16, "ch": 1}
            )
        for i in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[i : i + CHUNK_BYTES]
            await room.to_audio_target(data=chunk)
            if pacer:
                await pacer.wait(len(chunk))
    if started:
        await room.to_audio_target({"type": "audio_end"})


async def run_llm(room: Room, conversation_id: str, history: list[dict],
                  queue: Optional[asyncio.Queue], partial: dict) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    splitter = SentenceSplitter()
    await room.to_clients({"type": "message_start", "role": "assistant", "conversation_id": conversation_id})

    for _round in range(4):                      # allows chained tool calls
        stream = await groq.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS, tool_choice="auto", stream=True
        )
        parts: list[str] = []
        calls: dict[int, dict] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
                partial["text"] += delta.content
                await room.to_clients(
                    {"type": "token", "conversation_id": conversation_id, "text": delta.content}
                )
                if queue is not None:
                    for sentence in splitter.feed(delta.content):
                        await queue.put(sentence)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    entry = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            entry["name"] = tc.function.name
                        if tc.function.arguments:
                            entry["args"] += tc.function.arguments
        if not calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": "".join(parts) or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["args"] or "{}"},
                    }
                    for c in calls.values()
                ],
            }
        )
        for c in calls.values():
            messages.append(
                {"role": "tool", "tool_call_id": c["id"], "content": run_tool(room, c["name"])}
            )

    if queue is not None:
        for sentence in splitter.flush():
            await queue.put(sentence)
    return partial["text"].strip()


async def process_utterance(room: Room, text: str, source: str, conversation_id: Optional[str]):
    conv = await adb(db_ensure_conversation, room.device_id, conversation_id)
    conv_id = conv["id"]
    room.active_conversation = conv_id
    room.last_activity = time.time()

    saved = await adb(db_add_message, conv_id, "user", text, source)
    await room.to_clients({"type": "conversation", "conversation": {"id": conv_id, "title": saved["title"]}})
    await room.to_clients(
        {
            "type": "message", "role": "user", "id": saved["id"], "text": text,
            "conversation_id": conv_id, "source": source, "created_at": saved["created_at"],
        }
    )

    await set_status(room, "thinking")
    history = await adb(db_history, conv_id, HISTORY_LIMIT)

    speak = bool(room.audio_targets())
    if not speak:
        await room.to_clients(
            {"type": "warning", "text": f"No {room.output} connected: showing text only."}
        )

    queue: Optional[asyncio.Queue] = asyncio.Queue() if speak else None
    speaker_task = asyncio.create_task(speaker(room, queue)) if speak else None
    partial = {"text": ""}

    try:
        reply = await run_llm(room, conv_id, history, queue, partial)
        if queue is not None:
            await queue.put(None)
            await speaker_task
        if reply:
            msg = await adb(db_add_message, conv_id, "assistant", reply, source)
            await room.to_clients(
                {
                    "type": "message", "role": "assistant", "id": msg["id"], "text": reply,
                    "conversation_id": conv_id, "source": source, "created_at": msg["created_at"],
                }
            )
    except asyncio.CancelledError:
        if partial["text"].strip():                       # keep what was said before the interruption
            await adb(db_add_message, conv_id, "assistant", partial["text"].strip(), source)
        raise
    finally:
        if speaker_task and not speaker_task.done():
            speaker_task.cancel()
        room.last_activity = time.time()


async def run_turn(room: Room, source: str, conversation_id: Optional[str],
                   text: Optional[str] = None, audio: Optional[tuple[bytes, str]] = None):
    try:
        async with asyncio.timeout(TURN_TIMEOUT):
            if audio is not None:
                await set_status(room, "transcribing")
                text = await transcribe(*audio)
                if not text:
                    await room.to_clients({"type": "info", "text": "Didn't catch that."})
                    return
            if text:
                # NEW: Hardware voice command interceptor
                if "reset wifi" in text.lower() or "reset wi-fi" in text.lower():
                    await room.to_devices({"type": "command", "action": "reset_wifi"})
                    await room.to_clients({"type": "info", "text": "Command sent: Resetting device WiFi."})
                    return # Stop processing, do not send to LLM
                await process_utterance(room, text, source, conversation_id)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        log.error("Turn timed out after %ss", TURN_TIMEOUT)
        await room.to_clients({"type": "error", "text": "That took too long and was cancelled."})
    except Exception as e:
        log.exception("Turn failed")
        await room.to_clients({"type": "error", "text": f"Something went wrong ({type(e).__name__}). Check the server logs."})
    finally:
        await set_status(room, "idle")


async def cancel_turn(room: Room):
    """Cancel the running turn and tell the audio target to flush playback."""
    task = room.task
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        await room.to_devices({"type": "stop"})
        await room.to_clients({"type": "stop"})


async def start_turn(room: Room, **kwargs):
    await cancel_turn(room)
    room.task = asyncio.create_task(run_turn(room, **kwargs))


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="IoT Smart Speaker API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def token_ok(expected: str, given: str, required: bool) -> bool:
    if not expected:
        return not required
    return hmac.compare_digest(expected, given or "")


def require_client(x_token: str = Header(default="")):
    if not token_ok(CLIENT_TOKEN, x_token, required=False):
        raise HTTPException(status_code=401, detail="Invalid token")


@app.websocket("/ws/{device_id}")
async def ws_endpoint(ws: WebSocket, device_id: str,
                      role: str = Query("client"), token: str = Query("")):
    if role not in ("device", "client"):
        await ws.close(code=4400)
        return
    ok = token_ok(DEVICE_TOKEN, token, required=True) if role == "device" \
        else token_ok(CLIENT_TOKEN, token, required=False)
    if not ok:
        await ws.close(code=4401)
        return

    await ws.accept()
    room = manager.get(device_id)
    (room.devices if role == "device" else room.clients).add(ws)
    log.info("%s connected to room %s", role, device_id)

    await ws.send_text(json.dumps({
        "type": "hello", "role": role, "device_online": bool(room.devices),
        "output": room.output, "status": room.status,
        "telemetry": {"temp": room.telemetry["temp"], "hum": room.telemetry["hum"]},
    }))
    await room.to_clients({"type": "presence", "device_online": bool(room.devices)})
    
    if role == "device":
        await start_turn(room, source="system", text="The smart speaker just powered on. Say a short, friendly 1-sentence greeting so the user knows you are online and ready.")
    
    pending_meta: Optional[dict] = None
    pending_audio = bytearray()
    
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            # ---- text frames -------------------------------------------------
            if msg.get("text") is not None:
                raw = msg["text"].strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise ValueError
                except ValueError:
                    data = {"type": "text", "text": raw}      # tolerate plain text from simple clients

                kind = data.get("type")

                if kind == "telemetry" and role == "device":
                    temp, hum = data.get("temp"), data.get("hum")
                    room.telemetry = {"temp": temp, "hum": hum, "ts": time.time()}
                    await room.to_clients({"type": "telemetry", "temp": temp, "hum": hum})
                    if time.time() - room.last_sensor_log > SENSOR_LOG_INTERVAL:
                        room.last_sensor_log = time.time()
                        await adb(db_log_sensor, device_id, temp, hum)

                elif kind == "audio_meta":
                    pending_meta = data
                    pending_audio.clear()

                elif kind == "audio_end":
                    if pending_audio:
                        meta, pending_meta = pending_meta or {}, None
                        default_mime = "audio/pcm" if role == "device" else "audio/webm"
                        mime = meta.get("mime", default_mime)
                        conv_id = meta.get("conversation_id")
                        if role == "device" and not conv_id:
                            fresh = time.time() - room.last_activity < SESSION_REUSE_SECONDS
                            conv_id = room.active_conversation if fresh else None
                        
                        await start_turn(room, source="voice", conversation_id=conv_id,
                                         audio=(bytes(pending_audio), mime))
                        pending_audio.clear()

                elif kind == "text" and role == "client":
                    text = (data.get("text") or "").strip()
                    if text:
                        await start_turn(room, source="text", text=text,
                                         conversation_id=data.get("conversation_id"))

                elif kind == "set_output" and data.get("output") in ("device", "client"):
                    room.output = data["output"]
                    await room.to_clients({"type": "output", "output": room.output})

                elif kind == "stop":
                    await cancel_turn(room)

                elif kind == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

            # ---- binary frames (audio chunks) ---------------------
            elif msg.get("bytes"):
                pending_audio.extend(msg["bytes"])

    except Exception:
        log.exception("WebSocket error in room %s", device_id)
    finally:
        room.devices.discard(ws)
        room.clients.discard(ws)
        log.info("%s disconnected from room %s", role, device_id)
        if role == "device":
            await room.to_clients({"type": "presence", "device_online": bool(room.devices)})
        if not room.devices and not room.clients:
            await cancel_turn(room)


# -----------------------------------------------------------------------------
# REST API (powers the chat sidebar / history)
# -----------------------------------------------------------------------------
class ConversationCreate(BaseModel):
    device_id: str
    title: Optional[str] = None


class ConversationRename(BaseModel):
    title: str


def conv_dict(c: Conversation) -> dict:
    return {"id": c.id, "device_id": c.device_id, "title": c.title,
            "created_at": iso(c.created_at), "updated_at": iso(c.updated_at)}


@app.get("/api/health")
def health():
    return {"status": "online"}


@app.get("/api/devices/{device_id}", dependencies=[Depends(require_client)])
def device_status(device_id: str):
    room = manager.get(device_id)
    return {"device_id": device_id, "online": bool(room.devices), "output": room.output,
            "status": room.status,
            "telemetry": {"temp": room.telemetry["temp"], "hum": room.telemetry["hum"]}}


@app.get("/api/conversations", dependencies=[Depends(require_client)])
def list_conversations(device_id: str, limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Conversation).where(Conversation.device_id == device_id)
        .order_by(Conversation.updated_at.desc()).limit(limit)
    ).all()
    return [conv_dict(c) for c in rows]


@app.post("/api/conversations", dependencies=[Depends(require_client)])
def create_conversation(body: ConversationCreate, db: Session = Depends(get_db)):
    conv = Conversation(id=new_id(), device_id=body.device_id, title=body.title or "New chat")
    db.add(conv)
    db.commit()
    return conv_dict(conv)


@app.get("/api/conversations/{conversation_id}", dependencies=[Depends(require_client)])
def get_conversation(conversation_id: str, limit: int = Query(50, le=200),
                     before: Optional[str] = None, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    q = select(Message).where(Message.conversation_id == conversation_id)
    if before:
        try:
            dt = datetime.fromisoformat(before.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(400, "Invalid 'before' timestamp")
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        q = q.where(Message.created_at < dt)
    rows = list(reversed(db.scalars(q.order_by(Message.created_at.desc()).limit(limit)).all()))
    return {
        **conv_dict(conv),
        "messages": [{"id": m.id, "role": m.role, "text": m.content, "source": m.source,
                      "created_at": iso(m.created_at)} for m in rows],
        "has_more": len(rows) == limit,
    }


@app.patch("/api/conversations/{conversation_id}", dependencies=[Depends(require_client)])
def rename_conversation(conversation_id: str, body: ConversationRename, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    conv.title = body.title.strip()[:100] or conv.title
    db.commit()
    return conv_dict(conv)


@app.delete("/api/conversations/{conversation_id}", dependencies=[Depends(require_client)])
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    db.delete(conv)
    db.commit()
    return {"deleted": conversation_id}


# -----------------------------------------------------------------------------
# Frontend (served last so it never shadows /api or /ws)
# -----------------------------------------------------------------------------
from pathlib import Path
from fastapi.staticfiles import StaticFiles

_static = Path(__file__).parent / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
