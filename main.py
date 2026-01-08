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
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

# ---------------------------
# Optional Memvid SDK
# ---------------------------
MEMVID_AVAILABLE = False
try:
    # Memvid Python SDK usage: create/use, mem.put(title=..., label=..., metadata=..., text=...), mem.find(q,k,mode), mem.state(entity)
    # docs: https://docs.memvid.com/
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
LLM_API = os.getenv("LLM_API", "chat").lower().strip()  # "responses" | "chat"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Ollama tips:
# OPENAI_BASE_URL="http://localhost:11434/v1"
# LLM_API="responses" or "chat"
# LLM_MODEL="llama3.2" (or any local model name)
# Ollama supports /v1/responses and /v1/chat/completions in OpenAI compatibility mode.

# STT/TTS providers:
STT_PROVIDER = os.getenv("STT_PROVIDER", "local").lower().strip()     # "local" | "deepinfra"
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "local").lower().strip()     # "local" | "deepinfra"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
DEEPINFRA_WHISPER_MODEL = os.getenv("DEEPINFRA_WHISPER_MODEL", "openai/whisper-large-v3-turbo")
DEEPINFRA_TTS_MODEL = os.getenv("DEEPINFRA_TTS_MODEL", "hexgrad/Kokoro-82M")  # Model to use for TTS
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
    
    def get_voices(self) -> List[str]:
        """Return list of available voices"""
        try:
            return list(self.pipeline.voice_map.keys()) if hasattr(self.pipeline, 'voice_map') else []
        except Exception:
            return []


class TTSDeepInfra:
    """
    Unified DeepInfra TTS provider supporting multiple models:
    - hexgrad/Kokoro-82M (voices: af_heart, am_michael, am_adam, af_sky, etc.)
    - resemblyai/chatterbox-turbo (voices: chatterbox_default, various emotion styles)
    - canopylabs/orpheus-3b-0.1-ft (advanced expressive TTS)
    """
    
    # Model configurations
    MODELS = {
        "hexgrad/Kokoro-82M": {
            "voices": ["af_heart", "am_michael", "am_adam", "af_sky", "af_bella", "af_nicole", "af_sarah", "am_echo"],
            "supports_lang_code": True,
            "sample_rate": 24000
        },
        "resemblyai/chatterbox-turbo": {
            "voices": ["chatterbox_default", "nova", "alloy", "echo", "fable", "onyx", "shimmer"],
            "supports_lang_code": False,
            "sample_rate": 24000
        },
        "canopylabs/orpheus-3b-0.1-ft": {
            "voices": ["orpheus_default", "expressive", "narrator"],
            "supports_lang_code": False,
            "sample_rate": 24000
        }
    }
    
    def __init__(self, api_key: str, model: str = "hexgrad/Kokoro-82M", lang_code: str = "a"):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://api.deepinfra.com/v1/inference/{model}"
        self.lang = lang_code
        
        # Get model config
        config = self.MODELS.get(model, self.MODELS["hexgrad/Kokoro-82M"])
        self.sr = config["sample_rate"]
        self.supports_lang_code = config["supports_lang_code"]
        self.model_voices = config["voices"]

    def synth(self, text: str, voice: str) -> np.ndarray:
        # Build payload based on model type
        if self.supports_lang_code:
            payload = {"text": text, "voice": voice, "lang_code": self.lang}
        else:
            payload = {"text": text, "voice": voice}
        
        headers = {"Authorization": f"bearer {self.api_key}"}
        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
            if r.status_code == 200:
                with io.BytesIO(r.content) as f:
                    data, _ = sf.read(f, dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)  # downmix if needed
                    return data.astype(np.float32)
        except Exception as e:
            print(f"TTS Error: {e}")
        return np.zeros((0,), dtype=np.float32)
    
    def get_voices(self) -> List[str]:
        """Return list of available voices for current model"""
        return self.model_voices
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Return list of all available TTS models"""
        return list(cls.MODELS.keys())


# ---------------------------
# LLM provider (OpenAI SDK)
# ---------------------------
class LLMProvider:
    """
    Works with:
    - OpenAI cloud (default)
    - Fireworks via OpenAI compatibility (set OPENAI_BASE_URL)
    - Ollama OpenAI compatibility (OPENAI_BASE_URL=http://localhost:11434/v1)
    """

    def __init__(self, api_key: str, base_url: str, model: str, api_mode: str):
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        else:
            # Use a dummy key to avoid initialization error, will fail at generation time
            kwargs["api_key"] = "dummy_key_not_set"
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
        # create() for new, use() for existing
        self.mem = memvid_use("basic", path) if os.path.exists(path) else memvid_create(path)
        self.lock = asyncio.Lock()

    async def put(self, title: str, label: str, text: str, metadata: Optional[dict] = None):
        metadata = metadata or {}
        async with self.lock:
            def _do():
                # mem.put(title=..., label=..., metadata={}, text=...)
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
                # mem.find(query, k=..., mode='lex'/'sem')
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
        try:
            s.memories["A"] = MemvidAgentMemory(os.path.join(base, "agentA.mv2"))
            s.memories["B"] = MemvidAgentMemory(os.path.join(base, "agentB.mv2"))
        except RuntimeError:
            # Memvid not available
            s.memory_enabled = False

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
    tts = TTSDeepInfra(DEEPINFRA_API_KEY, DEEPINFRA_TTS_MODEL, KOKORO_LANG)
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
                    try:
                        session.memories["A"] = MemvidAgentMemory(os.path.join(base, "agentA.mv2"))
                        session.memories["B"] = MemvidAgentMemory(os.path.join(base, "agentB.mv2"))
                    except RuntimeError:
                        session.memory_enabled = False

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
# Voices endpoint
# ---------------------------
@app.get("/voices")
async def get_voices():
    """Return list of available TTS voices"""
    voices = tts.get_voices()
    return JSONResponse({"voices": voices})


@app.get("/tts_models")
async def get_tts_models():
    """Return list of available TTS models (for DeepInfra provider)"""
    if TTS_PROVIDER == "deepinfra":
        models = TTSDeepInfra.get_available_models()
        return JSONResponse({"models": models, "current": DEEPINFRA_TTS_MODEL})
    return JSONResponse({"models": [], "current": "local"})


@app.get("/tts_info")
async def get_tts_info():
    """Return current TTS provider and configuration info"""
    info = {
        "provider": TTS_PROVIDER,
        "voices": tts.get_voices()
    }
    if TTS_PROVIDER == "deepinfra":
        info["model"] = DEEPINFRA_TTS_MODEL
        info["available_models"] = TTSDeepInfra.get_available_models()
    return JSONResponse(info)


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
