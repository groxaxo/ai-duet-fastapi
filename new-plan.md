Here’s a merged, cleaned-up “best of both” version:
	•	2 agents (LLM↔LLM), user can interrupt, PTT, inject text mid-topic, change system prompts, pick voices
	•	Director settings (max turns, turn length, stop phrase, rules)
	•	Per-agent persistent memory via Memvid .mv2 using memvid-sdk (create/use/put/find/state)  ￼
	•	LLM provider options
	•	OpenAI (default)
	•	Any OpenAI-compatible endpoint (Fireworks, local gateways, etc.)  ￼
	•	Ollama via OpenAI compatibility (/v1/responses or /v1/chat/completions)  ￼
	•	STT/TTS
	•	Default: local faster-whisper + local Kokoro
	•	Optional: DeepInfra endpoints (like the script you pasted)

⸻

1) requirements.txt

fastapi
uvicorn[standard]
openai
webrtcvad
numpy
soundfile
requests
python-multipart

# Local STT/TTS (default path)
faster-whisper
kokoro

# Memory
memvid-sdk


⸻

2) main.py (backend)

import asyncio
import base64
import io
import json
import os
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import soundfile as sf
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI

# ---------------------------
# Optional Memvid SDK
# ---------------------------
MEMVID_AVAILABLE = False
try:
    # Memvid Python SDK usage: create/use, mem.put(title=..., label=..., metadata=..., text=...), mem.find(q,k,mode), mem.state(entity)
    # docs: https://docs.memvid.com/  [oai_citation:3‡Memvid](https://docs.memvid.com/)
    from memvid_sdk import create as memvid_create
    from memvid_sdk import use as memvid_use

    MEMVID_AVAILABLE = True
except Exception:
    MEMVID_AVAILABLE = False


# ---------------------------
# App
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Config
# ---------------------------
# LLM provider:
# - If OPENAI_BASE_URL is unset => talks to OpenAI cloud.
# - If OPENAI_BASE_URL is set (Fireworks / local gateway / Ollama OpenAI compatibility) it uses that.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
LLM_API = os.getenv("LLM_API", "responses").lower().strip()  # "responses" | "chat"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Ollama tips:
# OPENAI_BASE_URL="http://localhost:11434/v1"
# LLM_API="responses" or "chat"
# LLM_MODEL="llama3.2" (or any local model name)
# Ollama supports /v1/responses and /v1/chat/completions in OpenAI compatibility mode.  [oai_citation:4‡Ollama Docs](https://docs.ollama.com/api/openai-compatibility)

# STT/TTS providers:
STT_PROVIDER = os.getenv("STT_PROVIDER", "local").lower().strip()     # "local" | "deepinfra"
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "local").lower().strip()     # "local" | "deepinfra"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
DEEPINFRA_WHISPER_MODEL = os.getenv("DEEPINFRA_WHISPER_MODEL", "openai/whisper-large-v3-turbo")
DEEPINFRA_KOKORO_ENDPOINT = os.getenv("DEEPINFRA_KOKORO_ENDPOINT", "https://api.deepinfra.com/v1/inference/hexgrad/Kokoro-82M")
KOKORO_LANG = os.getenv("KOKORO_LANG", "a")

# Local faster-whisper:
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

# Memory defaults
MEMORY_DEFAULT_ENABLED = os.getenv("MEMORY_DEFAULT_ENABLED", "true").lower() == "true"
MEMORY_DEFAULT_K = int(os.getenv("MEMORY_DEFAULT_K", "6"))
MEMORY_DEFAULT_MODE = os.getenv("MEMORY_DEFAULT_MODE", "lex")  # lex|sem|hyb/auto (SDK accepts 'lex'/'sem'; default often hybrid)

# ---------------------------
# Audio utilities
# ---------------------------
def pcm16_to_wav_bytes(pcm: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


def float32_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    audio16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm16_to_wav_bytes(audio16.tobytes(), sr)


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))


def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return x / 32768.0


# ---------------------------
# VAD segmenter (webrtcvad)
# ---------------------------
class VADSegmenter:
    def __init__(self, sample_rate=16000, frame_ms=20, aggressiveness=3):
        assert frame_ms in (10, 20, 30)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sr = sample_rate
        self.frame_len = int(sample_rate * (frame_ms / 1000.0) * 2)

        self._buf = bytearray()
        self._speech_buf = bytearray()
        self._active = False
        self._silence_frames = 0

        # ~600ms silence cutoff
        self._silence_limit = int(600 / frame_ms)

    def push(self, pcm16: bytes) -> List[bytes]:
        self._buf.extend(pcm16)
        utts = []
        while len(self._buf) >= self.frame_len:
            frame = bytes(self._buf[: self.frame_len])
            del self._buf[: self.frame_len]

            if self.vad.is_speech(frame, self.sr):
                self._active = True
                self._speech_buf.extend(frame)
                self._silence_frames = 0
            elif self._active:
                self._speech_buf.extend(frame)
                self._silence_frames += 1
                if self._silence_frames >= self._silence_limit:
                    utts.append(bytes(self._speech_buf))
                    self._speech_buf.clear()
                    self._active = False
                    self._silence_frames = 0
        return utts


# ---------------------------
# STT providers
# ---------------------------
class STTLocalFasterWhisper:
    def __init__(self, model_size: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_pcm16(self, pcm16: bytes) -> str:
        audio = pcm16_bytes_to_float32(pcm16)
        segments, _info = self.model.transcribe(audio, language=None, beam_size=1)
        return "".join(seg.text for seg in segments).strip()


class STTDeepInfraWhisper:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.deepinfra.com/v1/openai/audio/transcriptions"

    def transcribe_pcm16(self, pcm16: bytes) -> str:
        wav_data = pcm16_to_wav_bytes(pcm16, 16000)
        files = {"file": ("audio.wav", wav_data, "audio/wav")}
        data = {"model": self.model}
        headers = {"Authorization": f"bearer {self.api_key}"}
        try:
            r = requests.post(self.endpoint, files=files, data=data, headers=headers, timeout=60)
            if r.status_code == 200:
                return (r.json().get("text") or "").strip()
        except Exception:
            pass
        return ""


# ---------------------------
# TTS providers
# ---------------------------
class TTSLocalKokoro:
    def __init__(self, lang_code: str = "a"):
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code=lang_code)
        self.sr = 24000

    def synth(self, text: str, voice: str) -> np.ndarray:
        chunks = []
        gen = self.pipeline(text, voice=voice, speed=1.0, split_pattern=r"\n+")
        for _i, (_gs, _ps, audio) in enumerate(gen):
            chunks.append(audio.astype(np.float32))
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks)


class TTSDeepInfraKokoro:
    def __init__(self, api_key: str, endpoint: str, lang_code: str = "a"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.lang = lang_code
        self.sr = 24000

    def synth(self, text: str, voice: str) -> np.ndarray:
        payload = {"text": text, "voice": voice, "lang_code": self.lang}
        headers = {"Authorization": f"bearer {self.api_key}"}
        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
            if r.status_code == 200:
                with io.BytesIO(r.content) as f:
                    data, _ = sf.read(f, dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)  # downmix if needed
                    return data.astype(np.float32)
        except Exception:
            pass
        return np.zeros((0,), dtype=np.float32)


# ---------------------------
# LLM provider (OpenAI SDK)
# ---------------------------
class LLMProvider:
    """
    Works with:
    - OpenAI cloud (default)
    - Fireworks via OpenAI compatibility (set OPENAI_BASE_URL)  [oai_citation:5‡Fireworks AI Docs](https://docs.fireworks.ai/tools-sdks/openai-compatibility)
    - Ollama OpenAI compatibility (OPENAI_BASE_URL=http://localhost:11434/v1)  [oai_citation:6‡Ollama Docs](https://docs.ollama.com/api/openai-compatibility)
    """

    def __init__(self, api_key: str, base_url: str, model: str, api_mode: str):
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)
        self.model = model
        self.api_mode = api_mode  # "responses" or "chat"

    def generate(self, instructions: str, messages: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
        if self.api_mode == "chat":
            # OpenAI-compatible Chat Completions style
            msgs = [{"role": "system", "content": instructions}] + messages
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()

        # Responses API style
        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=messages,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.output_text or "").strip()


# ---------------------------
# Memvid per-agent memory
# ---------------------------
class MemvidAgentMemory:
    def __init__(self, path: str):
        if not MEMVID_AVAILABLE:
            raise RuntimeError("Memvid SDK not available")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        # create() for new, use() for existing  [oai_citation:7‡Memvid](https://docs.memvid.com/)
        self.mem = memvid_use("basic", path) if os.path.exists(path) else memvid_create(path)
        self.lock = asyncio.Lock()

    async def put(self, title: str, label: str, text: str, metadata: Optional[dict] = None):
        metadata = metadata or {}
        async with self.lock:
            def _do():
                # mem.put(title=..., label=..., metadata={}, text=...)  [oai_citation:8‡Memvid](https://docs.memvid.com/)
                self.mem.put(title=title, label=label, metadata=metadata, text=text)
                # Some builds may have commit/seal; call if present (best-effort)
                for fn in ("commit", "seal", "flush"):
                    f = getattr(self.mem, fn, None)
                    if callable(f):
                        try:
                            f()
                            break
                        except Exception:
                            pass

            await asyncio.to_thread(_do)

    async def find(self, query: str, k: int = 6, mode: str = "lex") -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        async with self.lock:
            def _do():
                # mem.find(query, k=..., mode='lex'/'sem')  [oai_citation:9‡Memvid](https://docs.memvid.com/)
                return self.mem.find(q, k=k, mode=mode)

            res = await asyncio.to_thread(_do)

        # SDK return shape can vary; normalize to list of hits dicts
        hits = []
        if isinstance(res, dict) and "hits" in res:
            for h in res.get("hits") or []:
                hits.append(h if isinstance(h, dict) else {"preview": str(h)})
        elif isinstance(res, list):
            for h in res:
                hits.append(h if isinstance(h, dict) else {"preview": str(h)})
        else:
            # unknown; best effort
            hits.append({"preview": str(res)})
        return hits

    async def wipe(self):
        async with self.lock:
            def _do():
                try:
                    close = getattr(self.mem, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
                if os.path.exists(self.path):
                    os.remove(self.path)
                self.mem = memvid_create(self.path)

            await asyncio.to_thread(_do)


# ---------------------------
# Session / Agents
# ---------------------------
@dataclass
class Agent:
    id: str
    name: str
    instructions: str
    voice: str


@dataclass
class Director:
    instructions: str = ""
    max_turns: int = 200
    turn_length: str = "medium"  # short|medium|long
    stop_phrase: str = ""


@dataclass
class Session:
    session_id: str
    agents: Dict[str, Agent]
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    running: bool = False
    next_speaker: str = "A"
    cancel_speaking: asyncio.Event = field(default_factory=asyncio.Event)

    director: Director = field(default_factory=Director)

    # Memory
    memory_available: bool = MEMVID_AVAILABLE
    memory_enabled: bool = MEMORY_DEFAULT_ENABLED and MEMVID_AVAILABLE
    memory_k: int = MEMORY_DEFAULT_K
    memory_mode: str = MEMORY_DEFAULT_MODE
    memories: Dict[str, MemvidAgentMemory] = field(default_factory=dict)


sessions: Dict[str, Session] = {}


def max_tokens_from_turn_length(turn_length: str) -> int:
    tl = (turn_length or "medium").lower()
    if tl == "short":
        return 160
    if tl == "long":
        return 420
    return 260


def format_messages_for_agent(session: Session, agent_id: str) -> List[Dict[str, Any]]:
    """
    Convert shared transcript into per-agent messages.
    Other agent's messages are presented as 'user' to the current agent.
    """
    out: List[Dict[str, Any]] = []
    for m in session.transcript[-30:]:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["text"]})
        elif m["role"] == "agent":
            out.append(
                {"role": "assistant", "content": m["text"]}
                if m.get("who") == agent_id
                else {"role": "user", "content": m["text"]}
            )
    return out


def build_memory_query(session: Session) -> str:
    tail = session.transcript[-10:]
    return " ".join([x.get("text", "") for x in tail]).strip()[-1000:]


def render_memory_block(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        title = (h.get("title") or "").strip()
        label = (h.get("label") or "").strip()
        preview = (h.get("preview") or h.get("text") or "").strip()
        head = f"{i}. {title}" if title else f"{i}."
        if label:
            head += f" [{label}]"
        lines.append(f"{head}\n{preview}")
    return "\n\n".join(lines).strip()


def get_or_create_session(session_id: str) -> Session:
    if session_id in sessions:
        return sessions[session_id]

    s = Session(
        session_id=session_id,
        agents={
            "A": Agent("A", "Alice", "You are Alice. You are logical and precise.", "af_heart"),
            "B": Agent("B", "Bob", "You are Bob. You are emotional and creative.", "am_michael"),
        },
        director=Director(
            instructions="Keep it conversational, avoid repetition, and obey user interruptions immediately.",
            max_turns=200,
            turn_length="medium",
            stop_phrase="",
        ),
    )

    if s.memory_enabled:
        base = os.path.join("memories", session_id)
        os.makedirs(base, exist_ok=True)
        s.memories["A"] = MemvidAgentMemory(os.path.join(base, "agentA.mv2"))
        s.memories["B"] = MemvidAgentMemory(os.path.join(base, "agentB.mv2"))

    sessions[session_id] = s
    return s


# ---------------------------
# Instantiate providers
# ---------------------------
llm = LLMProvider(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model=LLM_MODEL,
    api_mode=LLM_API,
)

if STT_PROVIDER == "deepinfra":
    stt = STTDeepInfraWhisper(DEEPINFRA_API_KEY, DEEPINFRA_WHISPER_MODEL)
else:
    stt = STTLocalFasterWhisper(WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)

if TTS_PROVIDER == "deepinfra":
    tts = TTSDeepInfraKokoro(DEEPINFRA_API_KEY, DEEPINFRA_KOKORO_ENDPOINT, KOKORO_LANG)
else:
    tts = TTSLocalKokoro(KOKORO_LANG)


# ---------------------------
# Agent turn + loop
# ---------------------------
async def agent_turn(ws: WebSocket, session: Session, agent_id: str):
    agent = session.agents[agent_id]

    msgs = format_messages_for_agent(session, agent_id)

    # Memory retrieval (per agent)
    if session.memory_enabled and agent_id in session.memories and session.transcript:
        q = build_memory_query(session)
        hits = await session.memories[agent_id].find(q, k=session.memory_k, mode=session.memory_mode)
        if hits:
            mem_block = render_memory_block(hits)
            msgs = [{"role": "developer", "content": "Relevant long-term memory:\n\n" + mem_block}] + msgs
            await ws.send_text(json.dumps({"type": "memory_retrieved", "agent": agent_id, "text": mem_block[:600]}))

    instructions = agent.instructions.strip()
    if session.director.instructions.strip():
        instructions += "\n\n[Director rules]\n" + session.director.instructions.strip()

    max_tokens = max_tokens_from_turn_length(session.director.turn_length)

    # LLM generate
    text = await asyncio.to_thread(llm.generate, instructions, msgs, max_tokens, LLM_TEMPERATURE)

    if not text or session.cancel_speaking.is_set():
        return

    session.transcript.append({"role": "agent", "who": agent_id, "text": text})
    await ws.send_text(json.dumps({"type": "agent_text", "agent": agent_id, "name": agent.name, "text": text}))

    # Store agent output to its memory
    if session.memory_enabled and agent_id in session.memories:
        await session.memories[agent_id].put(
            title=f"{agent.name} said",
            label="dialogue",
            text=text,
            metadata={"who": agent_id, "ts": time.time(), "session": session.session_id},
        )

    # Stop phrase
    sp = (session.director.stop_phrase or "").strip()
    if sp and sp in text:
        session.running = False

    # TTS
    if session.cancel_speaking.is_set():
        return

    audio = await asyncio.to_thread(tts.synth, text, agent.voice)
    wav_b64 = b64e(float32_to_wav_bytes(audio, tts.sr))
    if not session.cancel_speaking.is_set():
        await ws.send_text(json.dumps({"type": "agent_audio", "agent": agent_id, "wav_b64": wav_b64}))


async def duet_loop(ws: WebSocket, session: Session):
    turns = 0
    session.running = True
    max_turns = max(1, int(session.director.max_turns or 200))

    while session.running and turns < max_turns:
        session.cancel_speaking.clear()
        await agent_turn(ws, session, session.next_speaker)
        session.next_speaker = "B" if session.next_speaker == "A" else "A"
        turns += 1
        await asyncio.sleep(0.35)

    session.running = False
    await ws.send_text(json.dumps({"type": "duet_ended", "turns": turns}))


# ---------------------------
# WebSocket
# ---------------------------
@app.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    session = get_or_create_session(session_id)
    vad = VADSegmenter()
    task: Optional[asyncio.Task] = None

    # initial state
    await ws.send_text(
        json.dumps(
            {
                "type": "session_state",
                "agents": {
                    k: {"name": a.name, "instructions": a.instructions, "voice": a.voice}
                    for k, a in session.agents.items()
                },
                "director": {
                    "instructions": session.director.instructions,
                    "max_turns": session.director.max_turns,
                    "turn_length": session.director.turn_length,
                    "stop_phrase": session.director.stop_phrase,
                },
                "memory": {
                    "available": session.memory_available,
                    "enabled": session.memory_enabled,
                    "k": session.memory_k,
                    "mode": session.memory_mode,
                },
            }
        )
    )

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            mtype = data.get("type")

            if mtype == "start_duet":
                session.running = True
                if task:
                    task.cancel()
                task = asyncio.create_task(duet_loop(ws, session))
                await ws.send_text(json.dumps({"type": "ok", "what": "duet_started"}))

            elif mtype == "stop":
                session.running = False
                session.cancel_speaking.set()
                if task:
                    task.cancel()
                await ws.send_text(json.dumps({"type": "ok", "what": "stopped"}))

            elif mtype == "interrupt":
                session.cancel_speaking.set()
                await ws.send_text(json.dumps({"type": "stop_audio"}))

            elif mtype == "user_text":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                session.cancel_speaking.set()
                await ws.send_text(json.dumps({"type": "stop_audio"}))

                session.transcript.append({"role": "user", "text": text})
                await ws.send_text(json.dumps({"type": "user_text_ack", "text": text}))

                # store user message into BOTH memories
                if session.memory_enabled and "A" in session.memories and "B" in session.memories:
                    await asyncio.gather(
                        session.memories["A"].put("User message", "dialogue", text, {"who": "user", "ts": time.time()}),
                        session.memories["B"].put("User message", "dialogue", text, {"who": "user", "ts": time.time()}),
                    )

            elif mtype == "user_audio":
                pcm = b64d(data["pcm16_b64"])
                for utt in vad.push(pcm):
                    session.cancel_speaking.set()
                    await ws.send_text(json.dumps({"type": "stop_audio"}))

                    text = await asyncio.to_thread(stt.transcribe_pcm16, utt)
                    text = (text or "").strip()
                    if not text:
                        continue

                    session.transcript.append({"role": "user", "text": text})
                    await ws.send_text(json.dumps({"type": "user_text_ack", "text": text}))

                    if session.memory_enabled and "A" in session.memories and "B" in session.memories:
                        await asyncio.gather(
                            session.memories["A"].put("User message (STT)", "dialogue", text, {"who": "user", "ts": time.time()}),
                            session.memories["B"].put("User message (STT)", "dialogue", text, {"who": "user", "ts": time.time()}),
                        )

            elif mtype == "set_agent":
                agent_id = data.get("agent")
                if agent_id in session.agents:
                    if "instructions" in data:
                        session.agents[agent_id].instructions = str(data["instructions"])
                    if "voice" in data:
                        session.agents[agent_id].voice = str(data["voice"])
                    await ws.send_text(json.dumps({"type": "ok", "what": "agent_updated", "agent": agent_id}))

            elif mtype == "set_director":
                d = data.get("director") or {}
                if "instructions" in d:
                    session.director.instructions = str(d["instructions"])
                if "max_turns" in d:
                    session.director.max_turns = int(d["max_turns"])
                if "turn_length" in d:
                    session.director.turn_length = str(d["turn_length"])
                if "stop_phrase" in d:
                    session.director.stop_phrase = str(d["stop_phrase"])
                await ws.send_text(json.dumps({"type": "ok", "what": "director_updated"}))

            elif mtype == "set_memory":
                enabled = bool(data.get("enabled", True))
                session.memory_enabled = enabled and session.memory_available

                if "k" in data:
                    session.memory_k = int(data["k"])
                if "mode" in data:
                    session.memory_mode = str(data["mode"])

                # lazily create memories if toggled on after session creation
                if session.memory_enabled and not session.memories and session.memory_available:
                    base = os.path.join("memories", session.session_id)
                    os.makedirs(base, exist_ok=True)
                    session.memories["A"] = MemvidAgentMemory(os.path.join(base, "agentA.mv2"))
                    session.memories["B"] = MemvidAgentMemory(os.path.join(base, "agentB.mv2"))

                await ws.send_text(json.dumps({"type": "ok", "what": "memory_updated"}))

            elif mtype == "wipe_memory":
                agent_id = data.get("agent")
                if agent_id in session.memories:
                    await session.memories[agent_id].wipe()
                    await ws.send_text(json.dumps({"type": "ok", "what": "memory_wiped", "agent": agent_id}))

            elif mtype == "save_session_to_memory":
                # Optional: write a single "snapshot" entry into both memories
                if session.memory_enabled and "A" in session.memories and "B" in session.memories:
                    blob = "\n".join(
                        [
                            (f'{m.get("who","User")}: {m["text"]}' if m["role"] == "agent" else f'User: {m["text"]}')
                            for m in session.transcript[-200:]
                        ]
                    )
                    await asyncio.gather(
                        session.memories["A"].put("Session snapshot", "snapshot", blob, {"ts": time.time()}),
                        session.memories["B"].put("Session snapshot", "snapshot", blob, {"ts": time.time()}),
                    )
                    await ws.send_text(json.dumps({"type": "ok", "what": "session_saved"}))

            elif mtype == "set_topic":
                # Optional debate helper (like your pasted solution)
                topic = (data.get("topic") or "").strip()
                if topic:
                    session.agents["A"].instructions = f"You are Alice. You SUPPORT: {topic}. Be clear and concise."
                    session.agents["B"].instructions = f"You are Bob. You OPPOSE: {topic}. Be clear and concise."
                    session.transcript.clear()
                    session.next_speaker = "A"
                    await ws.send_text(json.dumps({"type": "ok", "what": "topic_set", "topic": topic}))

            else:
                await ws.send_text(json.dumps({"type": "error", "message": f"Unknown type: {mtype}"}))

    except WebSocketDisconnect:
        session.running = False
        session.cancel_speaking.set()
        if task:
            task.cancel()


# ---------------------------
# Serve the UI
# ---------------------------
INDEX_HTML_PATH = os.path.join(os.path.dirname(__file__), "index.html")

@app.get("/")
async def root():
    if os.path.exists(INDEX_HTML_PATH):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h3>index.html not found next to main.py</h3>")


⸻

3) index.html (Tailwind UI + PTT + voice/prompt/director/memory)

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LLM↔LLM Voice Duet + Memvid</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    .custom-scrollbar::-webkit-scrollbar { width: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: #111827; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
    @keyframes pulse-ring { 0% { transform: scale(0.9); opacity: 0.6; } 100% { transform: scale(1.12); opacity: 0; } }
    .recording-pulse { animation: pulse-ring 1.5s cubic-bezier(0.215, 0.61, 0.355, 1) infinite; }
  </style>
</head>

<body class="bg-gray-900 text-gray-100 min-h-screen flex items-center justify-center p-4 font-sans">

  <div class="w-full max-w-6xl bg-gray-800 rounded-3xl shadow-2xl overflow-hidden border border-gray-700 flex flex-col lg:flex-row h-[88vh]">

    <!-- LEFT: Controls -->
    <div class="w-full lg:w-[380px] bg-gray-900 p-5 flex flex-col gap-4 border-r border-gray-700 overflow-y-auto custom-scrollbar">

      <div>
        <h1 class="text-2xl font-black text-blue-400">AI DUET</h1>
        <p class="text-xs text-gray-500">LLM↔LLM • STT/TTS • Memvid</p>
      </div>

      <!-- Status -->
      <div class="flex items-center justify-between bg-gray-800 p-3 rounded-xl border border-gray-700">
        <span class="text-xs font-bold text-gray-400">STATUS</span>
        <span id="status-badge" class="text-xs font-bold px-2 py-1 bg-red-900 text-red-200 rounded">OFFLINE</span>
      </div>

      <!-- Start/Stop -->
      <div class="grid grid-cols-2 gap-3">
        <button id="start-btn" class="bg-green-600 hover:bg-green-500 text-white font-bold py-3 rounded-xl transition" disabled>START</button>
        <button id="stop-btn" class="bg-red-600 hover:bg-red-500 text-white font-bold py-3 rounded-xl transition" disabled>STOP</button>
      </div>

      <button id="interrupt-btn" class="bg-yellow-700 hover:bg-yellow-600 text-yellow-100 font-bold py-2 rounded-xl transition" disabled>
        INTERRUPT (stop audio)
      </button>

      <!-- User text injection -->
      <div class="bg-gray-800 p-3 rounded-xl border border-gray-700">
        <div class="flex items-center gap-2 mb-2">
          <i class="fa-solid fa-comment-dots text-blue-300"></i>
          <span class="text-sm font-bold">Inject text</span>
        </div>
        <input id="user-text" class="w-full bg-gray-900 text-sm p-3 rounded-lg border border-gray-700 focus:border-blue-500 focus:outline-none"
               placeholder="Steer mid-topic: e.g. switch to robotics..." />
        <button id="send-text-btn" class="mt-2 w-full bg-blue-700 hover:bg-blue-600 text-xs font-bold py-2 rounded transition" disabled>SEND</button>
      </div>

      <!-- Debate topic helper -->
      <div class="bg-gray-800 p-3 rounded-xl border border-gray-700">
        <div class="flex items-center gap-2 mb-2">
          <i class="fa-solid fa-bolt text-amber-300"></i>
          <span class="text-sm font-bold">Quick debate topic</span>
        </div>
        <input id="topic-input" class="w-full bg-gray-900 text-sm p-3 rounded-lg border border-gray-700 focus:border-blue-500 focus:outline-none"
               placeholder="e.g. Future of Mars" />
        <button id="set-topic-btn" class="mt-2 w-full bg-gray-700 hover:bg-gray-600 text-xs font-bold py-2 rounded transition" disabled>SET TOPIC</button>
      </div>

      <!-- Memory -->
      <div class="bg-gray-800 p-3 rounded-xl border border-gray-700">
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center gap-2">
            <i class="fa-solid fa-memory text-purple-300"></i>
            <span class="text-sm font-bold">Memory (Memvid)</span>
          </div>
          <label class="flex items-center gap-2 text-xs text-gray-300">
            <input type="checkbox" id="mem-enabled" class="accent-purple-400" />
            enabled
          </label>
        </div>

        <div class="grid grid-cols-3 gap-2">
          <div>
            <label class="text-[10px] text-gray-400 font-bold">Top-K</label>
            <input id="mem-k" type="number" min="1" step="1" value="6"
                   class="w-full bg-gray-900 text-sm p-2 rounded-lg border border-gray-700 focus:border-purple-500 focus:outline-none" />
          </div>
          <div>
            <label class="text-[10px] text-gray-400 font-bold">Mode</label>
            <select id="mem-mode" class="w-full bg-gray-900 text-sm p-2 rounded-lg border border-gray-700 focus:border-purple-500 focus:outline-none">
              <option value="lex">lex</option>
              <option value="sem">sem</option>
              <option value="auto">auto</option>
              <option value="hyb">hyb</option>
            </select>
          </div>
          <div class="flex items-end">
            <button id="mem-update-btn" class="w-full bg-purple-900/60 hover:bg-purple-800 text-xs font-bold py-2 rounded transition" disabled>UPDATE</button>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-2 mt-2">
          <button id="wipe-a" class="bg-gray-700 hover:bg-gray-600 text-xs font-bold py-2 rounded transition" disabled>Wipe A</button>
          <button id="wipe-b" class="bg-gray-700 hover:bg-gray-600 text-xs font-bold py-2 rounded transition" disabled>Wipe B</button>
        </div>

        <button id="save-session-btn" class="mt-2 w-full bg-purple-900/50 hover:bg-purple-800 text-xs font-bold py-2 rounded transition" disabled>
          Save snapshot
        </button>

        <p id="mem-available" class="text-[10px] text-gray-500 mt-2"></p>
      </div>

      <!-- Director -->
      <div class="bg-gray-800 p-3 rounded-xl border border-gray-700">
        <div class="flex items-center gap-2 mb-2">
          <i class="fa-solid fa-clapperboard text-emerald-300"></i>
          <span class="text-sm font-bold">Director</span>
        </div>

        <label class="text-[10px] text-gray-400 font-bold">Rules</label>
        <textarea id="director-rules" rows="5"
                  class="w-full bg-gray-900 text-sm p-3 rounded-lg border border-gray-700 focus:border-emerald-500 focus:outline-none"></textarea>

        <div class="grid grid-cols-2 gap-2 mt-2">
          <div>
            <label class="text-[10px] text-gray-400 font-bold">Max turns</label>
            <input id="director-max-turns" type="number" min="1" step="1" value="200"
                   class="w-full bg-gray-900 text-sm p-2 rounded-lg border border-gray-700 focus:border-emerald-500 focus:outline-none" />
          </div>
          <div>
            <label class="text-[10px] text-gray-400 font-bold">Turn length</label>
            <select id="director-turn-length" class="w-full bg-gray-900 text-sm p-2 rounded-lg border border-gray-700 focus:border-emerald-500 focus:outline-none">
              <option value="short">short</option>
              <option value="medium" selected>medium</option>
              <option value="long">long</option>
            </select>
          </div>
        </div>

        <label class="text-[10px] text-gray-400 font-bold mt-2 block">Stop phrase</label>
        <input id="director-stop-phrase"
               class="w-full bg-gray-900 text-sm p-2 rounded-lg border border-gray-700 focus:border-emerald-500 focus:outline-none"
               placeholder="e.g. [[END]]" />

        <button id="director-update-btn" class="mt-2 w-full bg-emerald-900/60 hover:bg-emerald-800 text-xs font-bold py-2 rounded transition" disabled>
          UPDATE DIRECTOR
        </button>
      </div>

      <!-- Agent settings -->
      <div class="bg-gray-800 p-3 rounded-xl border border-gray-700">
        <div class="flex items-center gap-2 mb-2">
          <i class="fa-solid fa-user-gear text-indigo-300"></i>
          <span class="text-sm font-bold">Agents</span>
        </div>

        <!-- Agent A -->
        <div class="border border-gray-700 rounded-xl p-3 mb-3">
          <div class="text-xs font-black text-indigo-200 mb-2">Agent A (Alice)</div>
          <label class="text-[10px] text-gray-400 font-bold">Voice</label>
          <select id="a-voice" class="w-full bg-gray-900 text-sm p-2 rounded-lg border border-gray-700 focus:border-indigo-500 focus:outline-none">
            <option value="af_heart">af_heart</option>
            <option value="am_michael">am_michael</option>
            <option value="am_adam">am_adam</option>
            <option value="af_sky">af_sky</option>
          </select>

          <label class="text-[10px] text-gray-400 font-bold mt-2 block">System prompt</label>
          <textarea id="a-prompt" rows="4"
                    class="w-full bg-gray-900 text-sm p-2 rounded-lg border border-gray-700 focus:border-indigo-500 focus:outline-none"></textarea>

          <button id="a-update-btn" class="mt-2 w-full bg-indigo-900/60 hover:bg-indigo-800 text-xs font-bold py-2 rounded transition" disabled>
            UPDATE A
          </button>
        </div>

        <!-- Agent B -->
        <div class="border border-gray-700 rounded-xl p-3">
          <div class="text-xs font-black text-emerald-200 mb-2">Agent B (Bob)</div>
          <label class="text-[10px] text-gray-400 font-bold">Voice</label>
          <select id="b-voice" class="w-full bg-gray-900 text-sm p-2 rounded-lg border border-gray-700 focus:border-emerald-500 focus:outline-none">
            <option value="am_michael">am_michael</option>
            <option value="af_heart">af_heart</option>
            <option value="am_adam">am_adam</option>
            <option value="af_sky">af_sky</option>
          </select>

          <label class="text-[10px] text-gray-400 font-bold mt-2 block">System prompt</label>
          <textarea id="b-prompt" rows="4"
                    class="w-full bg-gray-900 text-sm p-2 rounded-lg border border-gray-700 focus:border-emerald-500 focus:outline-none"></textarea>

          <button id="b-update-btn" class="mt-2 w-full bg-emerald-900/60 hover:bg-emerald-800 text-xs font-bold py-2 rounded transition" disabled>
            UPDATE B
          </button>
        </div>
      </div>
    </div>

    <!-- RIGHT: Transcript + mic -->
    <div class="flex-1 flex flex-col relative bg-gray-800">
      <div id="transcript" class="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar pb-28">
        <div class="text-center text-gray-500 text-sm mt-10">Waiting for connection...</div>
      </div>

      <div class="absolute bottom-6 left-6 right-6 flex gap-3">
        <button id="mic-btn"
                class="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-black py-5 rounded-2xl shadow-lg transition-all active:scale-[0.98] flex items-center justify-center gap-3 group"
                disabled>
          <i class="fas fa-microphone text-xl group-hover:scale-110 transition"></i>
          <span>HOLD TO SPEAK (PTT)</span>
        </button>
      </div>
    </div>

  </div>

<script>
(() => {
  let ws;
  let audioCtx, micStream, scriptProcessor;
  let currentAudio = null;

  const statusBadge = document.getElementById('status-badge');
  const transcript = document.getElementById('transcript');

  const startBtn = document.getElementById('start-btn');
  const stopBtn = document.getElementById('stop-btn');
  const interruptBtn = document.getElementById('interrupt-btn');

  const micBtn = document.getElementById('mic-btn');

  const userText = document.getElementById('user-text');
  const sendTextBtn = document.getElementById('send-text-btn');

  const topicInput = document.getElementById('topic-input');
  const setTopicBtn = document.getElementById('set-topic-btn');

  const memEnabled = document.getElementById('mem-enabled');
  const memK = document.getElementById('mem-k');
  const memMode = document.getElementById('mem-mode');
  const memUpdateBtn = document.getElementById('mem-update-btn');
  const wipeA = document.getElementById('wipe-a');
  const wipeB = document.getElementById('wipe-b');
  const saveSessionBtn = document.getElementById('save-session-btn');
  const memAvailable = document.getElementById('mem-available');

  const directorRules = document.getElementById('director-rules');
  const directorMaxTurns = document.getElementById('director-max-turns');
  const directorTurnLength = document.getElementById('director-turn-length');
  const directorStopPhrase = document.getElementById('director-stop-phrase');
  const directorUpdateBtn = document.getElementById('director-update-btn');

  const aVoice = document.getElementById('a-voice');
  const bVoice = document.getElementById('b-voice');
  const aPrompt = document.getElementById('a-prompt');
  const bPrompt = document.getElementById('b-prompt');
  const aUpdateBtn = document.getElementById('a-update-btn');
  const bUpdateBtn = document.getElementById('b-update-btn');

  function setOnline(online) {
    if (online) {
      statusBadge.innerText = "ONLINE";
      statusBadge.className = "text-xs font-bold px-2 py-1 bg-green-900 text-green-200 rounded";
    } else {
      statusBadge.innerText = "OFFLINE";
      statusBadge.className = "text-xs font-bold px-2 py-1 bg-red-900 text-red-200 rounded";
    }

    const dis = !online;
    startBtn.disabled = dis;
    stopBtn.disabled = dis;
    interruptBtn.disabled = dis;
    micBtn.disabled = dis;
    sendTextBtn.disabled = dis;
    setTopicBtn.disabled = dis;

    memUpdateBtn.disabled = dis;
    wipeA.disabled = dis;
    wipeB.disabled = dis;
    saveSessionBtn.disabled = dis;

    directorUpdateBtn.disabled = dis;
    aUpdateBtn.disabled = dis;
    bUpdateBtn.disabled = dis;
  }

  function addSystemMessage(text, color='yellow') {
    const div = document.createElement('div');
    div.className = "flex justify-center my-2";
    div.innerHTML = `<span class="text-[11px] font-mono bg-gray-900 border border-${color}-900 text-${color}-400 px-2 py-1 rounded">${escapeHtml(text)}</span>`;
    transcript.appendChild(div);
    transcript.scrollTop = transcript.scrollHeight;
  }

  function addMessage(name, text, who) {
    const div = document.createElement('div');
    const isUser = who === 'user';
    div.className = `flex flex-col ${isUser ? 'items-end' : 'items-start'}`;

    let bubble = isUser ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-200';
    if (name === 'Alice') bubble = 'bg-indigo-900/80 text-indigo-100 border border-indigo-700';
    if (name === 'Bob') bubble = 'bg-emerald-900/80 text-emerald-100 border border-emerald-700';

    div.innerHTML = `
      <span class="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-1 px-1">${escapeHtml(name)}</span>
      <div class="max-w-[85%] px-4 py-3 rounded-2xl text-sm ${bubble} shadow-sm">
        ${escapeHtml(text)}
      </div>
    `;
    transcript.appendChild(div);
    transcript.scrollTop = transcript.scrollHeight;
  }

  function escapeHtml(s) {
    return (s || "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
  }

  function playAudio(b64) {
    if (currentAudio) {
      try { currentAudio.pause(); } catch (_) {}
      currentAudio = null;
    }
    const audio = new Audio("data:audio/wav;base64," + b64);
    currentAudio = audio;
    audio.play();
  }

  function stopAudio() {
    if (currentAudio) {
      try { currentAudio.pause(); } catch (_) {}
      currentAudio = null;
    }
  }

  // WS connect
  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws/demo`);

    ws.onopen = () => {
      setOnline(true);
      transcript.innerHTML = "";
      addSystemMessage("System ready", "green");
    };

    ws.onclose = () => {
      setOnline(false);
      addSystemMessage("Disconnected", "red");
      stopAudio();
    };

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);

      if (data.type === "session_state") {
        // seed UI
        if (data.agents?.A) {
          aPrompt.value = data.agents.A.instructions || "";
          aVoice.value = data.agents.A.voice || "af_heart";
        }
        if (data.agents?.B) {
          bPrompt.value = data.agents.B.instructions || "";
          bVoice.value = data.agents.B.voice || "am_michael";
        }

        if (data.director) {
          directorRules.value = data.director.instructions || "";
          directorMaxTurns.value = data.director.max_turns || 200;
          directorTurnLength.value = data.director.turn_length || "medium";
          directorStopPhrase.value = data.director.stop_phrase || "";
        }

        if (data.memory) {
          const avail = !!data.memory.available;
          memEnabled.checked = !!data.memory.enabled;
          memK.value = data.memory.k || 6;
          memMode.value = data.memory.mode || "lex";
          memAvailable.innerText = avail ? "Memvid available" : "Memvid not installed (memory disabled)";
          if (!avail) {
            memEnabled.disabled = true;
            memUpdateBtn.disabled = true;
            wipeA.disabled = true;
            wipeB.disabled = true;
            saveSessionBtn.disabled = true;
          }
        }
      }

      if (data.type === "agent_text") {
        addMessage(data.name || data.agent, data.text, "agent");
      }

      if (data.type === "user_text_ack") {
        addMessage("You", data.text, "user");
      }

      if (data.type === "agent_audio") {
        playAudio(data.wav_b64);
      }

      if (data.type === "stop_audio") {
        stopAudio();
      }

      if (data.type === "memory_retrieved") {
        addSystemMessage(`🧠 ${data.agent} recalled memory`, "purple");
      }

      if (data.type === "ok") {
        addSystemMessage(`✅ ${data.what}`, "green");
      }

      if (data.type === "error") {
        addSystemMessage(`❗ ${data.message}`, "red");
      }
    };
  }

  // Buttons
  startBtn.onclick = () => ws.send(JSON.stringify({type: "start_duet"}));
  stopBtn.onclick = () => ws.send(JSON.stringify({type: "stop"}));
  interruptBtn.onclick = () => {
    ws.send(JSON.stringify({type: "interrupt"}));
    stopAudio();
  };

  sendTextBtn.onclick = () => {
    const t = userText.value.trim();
    if (!t) return;
    ws.send(JSON.stringify({type: "user_text", text: t}));
    userText.value = "";
  };

  setTopicBtn.onclick = () => {
    const t = topicInput.value.trim();
    if (!t) return;
    ws.send(JSON.stringify({type: "set_topic", topic: t}));
  };

  memUpdateBtn.onclick = () => {
    ws.send(JSON.stringify({
      type: "set_memory",
      enabled: memEnabled.checked,
      k: parseInt(memK.value || "6", 10),
      mode: memMode.value
    }));
  };

  wipeA.onclick = () => ws.send(JSON.stringify({type: "wipe_memory", agent: "A"}));
  wipeB.onclick = () => ws.send(JSON.stringify({type: "wipe_memory", agent: "B"}));
  saveSessionBtn.onclick = () => ws.send(JSON.stringify({type: "save_session_to_memory"}));

  directorUpdateBtn.onclick = () => {
    ws.send(JSON.stringify({
      type: "set_director",
      director: {
        instructions: directorRules.value,
        max_turns: parseInt(directorMaxTurns.value || "200", 10),
        turn_length: directorTurnLength.value,
        stop_phrase: directorStopPhrase.value
      }
    }));
  };

  aUpdateBtn.onclick = () => {
    ws.send(JSON.stringify({
      type: "set_agent",
      agent: "A",
      instructions: aPrompt.value,
      voice: aVoice.value
    }));
  };

  bUpdateBtn.onclick = () => {
    ws.send(JSON.stringify({
      type: "set_agent",
      agent: "B",
      instructions: bPrompt.value,
      voice: bVoice.value
    }));
  };

  // PTT mic
  micBtn.onmousedown = async () => {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const source = audioCtx.createMediaStreamSource(micStream);
      scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);

      source.connect(scriptProcessor);
      scriptProcessor.connect(audioCtx.destination);

      scriptProcessor.onaudioprocess = (e) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);

        const pcm16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          const s = Math.max(-1, Math.min(1, input[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        const u8 = new Uint8Array(pcm16.buffer);
        let bin = "";
        const chunk = 0x8000;
        for (let i = 0; i < u8.length; i += chunk) {
          bin += String.fromCharCode(...u8.subarray(i, i + chunk));
        }

        ws.send(JSON.stringify({ type: "user_audio", pcm16_b64: btoa(bin) }));
      };

      micBtn.classList.remove('bg-blue-600');
      micBtn.classList.add('bg-red-600', 'recording-pulse');
      micBtn.innerHTML = `<i class="fas fa-wave-square"></i><span>LISTENING...</span>`;

    } catch (err) {
      console.error(err);
    }
  };

  micBtn.onmouseup = () => {
    try {
      if (micStream) micStream.getTracks().forEach(t => t.stop());
      if (scriptProcessor) scriptProcessor.disconnect();
      if (audioCtx) audioCtx.close();
    } catch (_) {}

    micStream = null; scriptProcessor = null; audioCtx = null;

    micBtn.classList.add('bg-blue-600');
    micBtn.classList.remove('bg-red-600', 'recording-pulse');
    micBtn.innerHTML = `<i class="fas fa-microphone text-xl"></i><span>HOLD TO SPEAK (PTT)</span>`;
  };

  // Boot
  setOnline(false);
  connect();

  // Default director rules
  directorRules.value = directorRules.value || [
    "Keep it conversational and avoid repetition.",
    "Each turn: 1–3 short paragraphs.",
    "Ask a short question every few turns.",
    "If the user interrupts, follow the new direction immediately."
  ].join("\\n");
})();
</script>

</body>
</html>


⸻

Run

pip install -r requirements.txt
export OPENAI_API_KEY="..."
uvicorn main:app --reload
# open http://localhost:8000


⸻

Provider configs

OpenAI (default)

export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"
export LLM_API="responses"

Fireworks (OpenAI-compatible)

Fireworks supports the OpenAI SDK by pointing base_url to their endpoint.  ￼

export OPENAI_API_KEY="FW_..."
export OPENAI_BASE_URL="https://api.fireworks.ai/inference/v1"
export LLM_API="chat"           # safest for compatibility
export LLM_MODEL="accounts/fireworks/models/deepseek-v3"

Ollama (local)

Ollama supports OpenAI compatibility including /v1/responses.  ￼

export OPENAI_API_KEY="ollama"              # any string works
export OPENAI_BASE_URL="http://localhost:11434/v1"
export LLM_API="responses"                  # or "chat"
export LLM_MODEL="llama3.2"                 # or any pulled model

DeepInfra STT/TTS (optional)

export DEEPINFRA_API_KEY="..."
export STT_PROVIDER="deepinfra"
export TTS_PROVIDER="deepinfra"


⸻

If you want the voice picker to auto-populate from your Kokoro install (instead of hardcoded options), I can add a /voices endpoint that returns available voice IDs and have the UI load them.