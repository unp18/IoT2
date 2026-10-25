"""
IoT Smart Speaker backend (PostgreSQL Version)
Pipeline:  audio/text in -> Whisper (Groq) -> LLM (Groq, streaming, tools)
           -> sentence-level edge-tts -> ffmpeg -> 16 kHz/16-bit/mono PCM -> ESP32
Safety:    every telemetry reading -> trained spike model (fire_model.json) -> alert + voice alarm + dashboard

WebSocket:  /ws/{device_id}?role=device|client&token=...
Client -> server (JSON text frames)
  {"type":"text","text":"...","conversation_id":"<optional>"}
  {"type":"audio_meta","mime":"audio/webm","conversation_id":"<optional>"}   then binary frame(s), then {"type":"audio_end"}
  {"type":"set_output","output":"device"|"client"}
  {"type":"ack_alert"}                 acknowledge a fire alert (silences the voice, clears the banner)
  {"type":"stop"}
  {"type":"ping"}
Device -> server
  {"type":"telemetry","temp":24.1,"hum":55}
  {"type":"audio_meta","mime":"audio/pcm"}  then binary PCM frame(s), then {"type":"audio_end"}
Server -> everyone
  {"type":"status","state":"idle|transcribing|thinking|speaking"}
Server -> clients
  {"type":"hello", ..., "alert": <active alert or null>}     {"type":"presence","device_online":bool}
  {"type":"conversation","conversation":{id,title}}
  {"type":"message","role":"user|assistant","id","text","conversation_id","source","created_at"}
  {"type":"message_start","role":"assistant","conversation_id"}
  {"type":"token","conversation_id","text"}
  {"type":"telemetry","temp","hum","fire":bool,"fire_score":0..1|null}     {"type":"output","output":"device|client"}
  {"type":"alert","kind":"fire","level":"critical","text","temp","rise","score","detector","ts"}
  {"type":"alert_clear","kind":"fire","reason":"..."}
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
import smtplib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import edge_tts
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pydantic import BaseModel
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text, create_engine, select, update
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from fire_model import FireModel, SpikeDetector

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("speaker")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Default updated to PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/speaker_db")
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
# Seconds between telemetry DB writes. Use 1 while recording training data (see train_fire_model.py).
SENSOR_LOG_INTERVAL = int(os.getenv("SENSOR_LOG_INTERVAL", "60"))

# Fire / sudden temperature spike detection (trained model in fire_model.json, see train_fire_model.py)
FIRE_MODEL_PATH = Path(os.getenv("FIRE_MODEL_PATH", str(Path(__file__).parent / "fire_model.json")))
_thr = os.getenv("FIRE_THRESHOLD")
FIRE_THRESHOLD = float(_thr) if _thr else None                  # None = use the threshold saved with the model
FIRE_MIN_TEMP = float(os.getenv("FIRE_MIN_TEMP", "35.0"))       # safety gate: never alarm below this temperature
FIRE_CLEAR_TEMP = float(os.getenv("FIRE_CLEAR_TEMP", str(FIRE_MIN_TEMP - 3)))   # alert auto-clears below this
FIRE_CONFIRMATIONS = int(os.getenv("FIRE_CONFIRMATIONS", "3"))  # consecutive positive windows required
FIRE_MAX_GAP = float(os.getenv("FIRE_MAX_GAP", "5"))            # seconds without data that resets the window
FIRE_REPEAT_SECONDS = int(os.getenv("FIRE_REPEAT_SECONDS", "20"))
FIRE_MAX_ANNOUNCEMENTS = int(os.getenv("FIRE_MAX_ANNOUNCEMENTS", "5"))
ALARM_TEXT = "Warning! Possible fire detected in the room. Please check immediately."

# Email alerts. Render's free tier BLOCKS outbound SMTP (ports 25/465/587), so prefer an HTTPS email API:
#   EMAIL_PROVIDER = resend | brevo | smtp      (empty = email alerts disabled)
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "").strip().lower()
EMAIL_API_KEY = os.getenv("EMAIL_API_KEY", "")                  # resend / brevo API key
ALERT_EMAIL_TO = [e.strip() for e in os.getenv("ALERT_EMAIL_TO", "").split(",") if e.strip()]
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")            # resend: optional; brevo: a verified sender; smtp: optional
EMAIL_COOLDOWN_SECONDS = int(os.getenv("EMAIL_COOLDOWN_SECONDS", "300"))   # at most one alert email per this many seconds
APP_URL = os.getenv("APP_URL", "")                              # e.g. https://your-service.onrender.com (adds a link)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
RESEND_URL = "https://api.resend.com/emails"
BREVO_URL = "https://api.brevo.com/v3/smtp/email"

SYSTEM_PROMPT = (
    "You are a smart speaker assistant. Your replies are spoken aloud, so keep them brief, "
    "natural and conversational. Do not use markdown, lists, emojis or code blocks. "
    "Use the get_dht_sensor_data tool when asked about room temperature, humidity, climate or fire risk."
)

HALLUCINATIONS = {"", "you", "thank you", "thanks", "thanks for watching", "bye", "okay"}

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY is not set")
if not DEVICE_TOKEN:
    log.warning("DEVICE_TOKEN is not set: devices will be rejected")
if not shutil.which("ffmpeg"):
    log.warning("ffmpeg not found on PATH: TTS audio conversion will fail")

try:
    fire_model: Optional[FireModel] = FireModel.load(FIRE_MODEL_PATH)
    log.info("Loaded fire model: window=%d threshold=%.2f trained_at=%s",
             fire_model.window, FIRE_THRESHOLD or fire_model.threshold, fire_model.spec.get("trained_at"))
except Exception as e:
    fire_model = None
    log.warning("Fire model not loaded (%s): falling back to the simple rule-based detector", e)

groq = AsyncGroq(api_key=GROQ_API_KEY)

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def new_id() -> str:
    return uuid.uuid4().hex

def iso(dt: datetime) -> str:
    # PostgreSQL with SQLAlchemy DateTime(timezone=True) natively returns timezone-aware datetimes.
    # This fallback is kept just in case naive objects are ever passed locally.
    if dt.tzinfo is None:
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

class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(50), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rise: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # model probability
    detector: Mapped[Optional[str]] = mapped_column(String(10), nullable=True) # model | rules
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

# Configure PostgreSQL Engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,          # Standard pool size for PG
    max_overflow=20,       # Allow up to 20 extra connections during spikes
    pool_pre_ping=True     # Ensure connections aren't stale before using them
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

def db_add_alert(db: Session, device_id: str, kind: str, message: str, temp: float, rise: float,
                 score: Optional[float], detector: str) -> dict:
    alert = Alert(device_id=device_id, kind=kind, message=message, temperature=temp, rise=rise,
                  score=score, detector=detector)
    db.add(alert)
    db.flush()
    return {"id": alert.id, "created_at": iso(alert.created_at)}

def db_ack_alerts(db: Session, device_id: str):
    db.execute(update(Alert).where(Alert.device_id == device_id, Alert.acknowledged.is_(False))
               .values(acknowledged=True))


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
        # fire detection state
        self.detector = SpikeDetector(fire_model, window=10, threshold=FIRE_THRESHOLD, min_temp=FIRE_MIN_TEMP,
                                      confirmations=FIRE_CONFIRMATIONS, max_gap=FIRE_MAX_GAP)
        self.fire_active = False
        self.alert: Optional[dict] = None
        self.last_score: Optional[float] = None
        self.last_announce = 0.0
        self.announce_count = 0
        self.last_email = 0.0

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
            "description": "Get the current room temperature (Celsius), humidity, and fire/temperature-spike status "
                           "from the ESP32 sensor.",
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
                "fire_alert_active": room.fire_active,
                "spike_risk_percent": None if room.last_score is None else round(room.last_score * 100),
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
                if "reset wifi" in text.lower() or "reset wi-fi" in text.lower():
                    await room.to_devices({"type": "command", "action": "reset_wifi"})
                    await room.to_clients({"type": "info", "text": "Command sent: Resetting device WiFi."})
                    return 
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
# Fire / sudden temperature spike: trained model -> alert, voice alarm, dashboard
# -----------------------------------------------------------------------------
_alarm_pcm: dict[str, bytes] = {}      # synthesized once, replayed instantly afterwards

async def run_announcement(room: Room, text: str):
    """Speak an alert on the device(s) regardless of the reply-output setting."""
    try:
        await set_status(room, "speaking")
        pcm = _alarm_pcm.get(text)
        if pcm is None:
            pcm = await synthesize(text)
            _alarm_pcm[text] = pcm
        if pcm and room.devices:
            await room.to_devices({"type": "audio_start", "rate": SAMPLE_RATE, "bits": 16, "ch": 1})
            pacer = Pacer()
            for i in range(0, len(pcm), CHUNK_BYTES):
                chunk = pcm[i : i + CHUNK_BYTES]
                await room._send(room.devices, data=chunk)
                await pacer.wait(len(chunk))
            await room.to_devices({"type": "audio_end"})
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Announcement failed")
    finally:
        await set_status(room, "idle")

async def clear_alert(room: Room, reason: str):
    if not room.fire_active:
        return
    room.fire_active = False
    room.alert = None
    room.detector.reset()
    await room.to_clients({"type": "alert_clear", "kind": "fire", "reason": reason})

async def ack_alert(room: Room):
    await adb(db_ack_alerts, room.device_id)
    await cancel_turn(room)                      # silence the voice immediately
    await clear_alert(room, "acknowledged")

_bg_tasks: set = set()

def _spawn(coro):
    """Run a coroutine in the background without letting it be garbage-collected mid-flight."""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task

def email_configured() -> bool:
    if not ALERT_EMAIL_TO:
        return False
    if EMAIL_PROVIDER in ("resend", "brevo"):
        return bool(EMAIL_API_KEY)
    if EMAIL_PROVIDER == "smtp":
        return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)
    return False

def _post_json(url: str, headers: dict, payload: dict) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "smart-speaker/1.0", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{EMAIL_PROVIDER} API error {e.code}: {e.read().decode(errors='ignore')[:200]}") from None

def send_email_sync(subject: str, body: str) -> None:
    """Blocking send (call from a thread). Raises RuntimeError / OSError on failure."""
    if not email_configured():
        raise RuntimeError("Email is not configured (set EMAIL_PROVIDER, its credentials, and ALERT_EMAIL_TO)")
    sender = ALERT_EMAIL_FROM
    if EMAIL_PROVIDER == "resend":
        _post_json(RESEND_URL, {"Authorization": f"Bearer {EMAIL_API_KEY}"},
                   {"from": sender or "Smart Speaker <onboarding@resend.dev>", "to": ALERT_EMAIL_TO,
                    "subject": subject, "text": body})
    elif EMAIL_PROVIDER == "brevo":
        if not sender:
            raise RuntimeError("Brevo needs ALERT_EMAIL_FROM set to a verified sender address")
        _post_json(BREVO_URL, {"api-key": EMAIL_API_KEY},
                   {"sender": {"name": "Smart Speaker", "email": sender},
                    "to": [{"email": e} for e in ALERT_EMAIL_TO], "subject": subject, "textContent": body})
    else:                                                        # smtp
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, sender or SMTP_USER, ", ".join(ALERT_EMAIL_TO)
        msg.set_content(body)
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
                smtp.login(SMTP_USER, SMTP_PASSWORD)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(SMTP_USER, SMTP_PASSWORD)
                smtp.send_message(msg)

def build_alert_email(room: Room, alert: dict) -> tuple[str, str]:
    hum = room.telemetry.get("hum")
    subject = f"FIRE ALERT: {room.device_id} at {alert['temp']:.1f} °C (+{alert['rise']:.1f} °C)"
    lines = [
        "Possible fire detected by your smart speaker.",
        "",
        f"Device: {room.device_id}",
        f"Time (UTC): {alert['ts']}",
        f"Temperature: {alert['temp']:.1f} °C (rose {alert['rise']:.1f} °C over the last {room.detector.window} readings)",
        f"Humidity: {hum}%" if hum is not None else None,
        f"Model score: {alert['score']:.2f} ({alert['detector']})",
        f"Dashboard: {APP_URL}" if APP_URL else None,
        "",
        "Check the room immediately. This is an automated alert from a demo project, not a certified fire alarm.",
    ]
    return subject, "\n".join(line for line in lines if line is not None)

async def send_alert_email(room: Room, alert: dict):
    """Email the alert (rate-limited). Never raises: failures are logged and shown on the dashboard."""
    if not email_configured():
        return
    now = time.time()
    if now - room.last_email < EMAIL_COOLDOWN_SECONDS:
        log.info("Alert email skipped (cooldown)")
        return
    room.last_email = now
    subject, body = build_alert_email(room, alert)
    try:
        await asyncio.to_thread(send_email_sync, subject, body)
        log.info("Alert email sent to %s", ", ".join(ALERT_EMAIL_TO))
        await room.to_clients({"type": "info", "text": f"Alert email sent to {', '.join(ALERT_EMAIL_TO)}."})
    except Exception as e:
        log.error("Alert email failed: %s", e)
        await room.to_clients({"type": "warning", "text": f"Alert email failed: {e}"})

async def check_fire(room: Room, temp, allow_voice: bool = True) -> Optional[float]:
    """Run the model on the newest reading. Returns the model score (None while the window fills)."""
    try:
        temp = float(temp)
    except (TypeError, ValueError):
        return room.last_score
    now = time.time()
    score, triggered, rise = room.detector.update(temp, now)
    room.last_score = score

    if not room.fire_active:
        if triggered:
            room.fire_active = True
            room.announce_count = 0
            room.last_announce = 0.0
            text = (f"Possible fire: temperature rose {rise:.1f} °C over the last "
                    f"{room.detector.window} readings and is now {temp:.1f} °C.")
            row = await adb(db_add_alert, room.device_id, "fire", text, temp, rise, score, room.detector.name)
            room.alert = {"kind": "fire", "level": "critical", "text": text, "temp": temp,
                          "rise": round(rise, 1), "score": round(score, 3),
                          "detector": room.detector.name, "ts": row["created_at"]}
            room.detector.reset()                # need fresh evidence before triggering again
            log.warning("FIRE ALERT in %s: %s (score %.2f, %s)", room.device_id, text, score, room.detector.name)
            await room.to_clients({"type": "alert", **room.alert})
            _spawn(send_alert_email(room, room.alert))
    elif temp < FIRE_CLEAR_TEMP:
        await clear_alert(room, "temperature back to normal")

    if (room.fire_active and allow_voice and room.devices
            and room.announce_count < FIRE_MAX_ANNOUNCEMENTS
            and now - room.last_announce >= FIRE_REPEAT_SECONDS):
        room.last_announce = now
        room.announce_count += 1
        await cancel_turn(room)                  # an alarm outranks a conversation
        room.task = asyncio.create_task(run_announcement(room, ALARM_TEXT))
    return room.last_score

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
        "output": room.output, "status": room.status, "alert": room.alert,
        "telemetry": {"temp": room.telemetry["temp"], "hum": room.telemetry["hum"]},
    }))
    await room.to_clients({"type": "presence", "device_online": bool(room.devices)})
    
    if role == "device":
        greeting = "Hi there! I'm powered up, connected, and ready to chat. What's on your mind?"
        await room.to_devices({"type": "message", "role": "system", "text": greeting})
    
        conv = await adb(db_ensure_conversation, device_id, room.active_conversation)
        room.active_conversation = conv["id"]
        msg = await adb(db_add_message, conv["id"], "assistant", greeting, "system")
    
        await room.to_clients({
            "type": "message", "role": "assistant", "id": msg["id"], "text": greeting,
            "conversation_id": conv["id"], "source": "system", "created_at": msg["created_at"]
        })

        if room.audio_targets():
            queue = asyncio.Queue()
            speaker_task = asyncio.create_task(speaker(room, queue))
            await queue.put(greeting)
            await queue.put(None)
            await speaker_task
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
                    score = await check_fire(room, temp, allow_voice=(not pending_audio))
                    await room.to_clients({"type": "telemetry", "temp": temp, "hum": hum,
                                           "fire": room.fire_active,
                                           "fire_score": None if score is None else round(score, 3)})
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

                elif kind == "ack_alert" and role == "client":
                    await ack_alert(room)

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
            "status": room.status, "fire_alert": room.alert, "email_alerts": email_configured(),
            "telemetry": {"temp": room.telemetry["temp"], "hum": room.telemetry["hum"]}}

@app.get("/api/model", dependencies=[Depends(require_client)])
def model_info():
    """Details of the deployed analytics model: dataset, metrics, comparison (for the dashboard / report)."""
    if fire_model is None:
        return {"loaded": False, "detector": "rules"}
    return {"loaded": True, "detector": "model", **fire_model.info(),
            "runtime": {"min_temp": FIRE_MIN_TEMP, "confirmations": FIRE_CONFIRMATIONS,
                        "threshold": FIRE_THRESHOLD or fire_model.threshold}}

@app.post("/api/test-email", dependencies=[Depends(require_client)])
def test_email():
    """Send a test message to check the email settings without needing a fire."""
    if not email_configured():
        raise HTTPException(400, "Email is not configured: set EMAIL_PROVIDER, its credentials and ALERT_EMAIL_TO")
    try:
        send_email_sync("Smart speaker test email",
                        "This is a test message from your smart speaker server. Email alerts are working.")
    except Exception as e:
        raise HTTPException(502, f"Email failed: {e}")
    return {"sent": True, "to": ALERT_EMAIL_TO, "provider": EMAIL_PROVIDER}

@app.get("/api/alerts", dependencies=[Depends(require_client)])
def list_alerts(device_id: str, limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    rows = db.scalars(select(Alert).where(Alert.device_id == device_id)
                      .order_by(Alert.created_at.desc()).limit(limit)).all()
    return [{"id": a.id, "kind": a.kind, "message": a.message, "temperature": a.temperature,
             "rise": a.rise, "score": a.score, "detector": a.detector,
             "acknowledged": a.acknowledged, "created_at": iso(a.created_at)} for a in rows]

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
from fastapi.staticfiles import StaticFiles

_static = Path(__file__).parent / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
