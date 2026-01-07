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
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from faster_whisper import WhisperModel
import soundfile as sf
import requests  # Added for DeepInfra TTS

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "AI Duet FastAPI Server", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}


# ---------- Utilities ----------


def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return x / 32768.0


def float32_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    audio16 = np.clip(audio, -1.0, 1.0)
    audio16 = (audio16 * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio16.tobytes())
    return buf.getvalue()


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))


# ---------- LLM (Fireworks AI with Llama v3.1) ----------


class LLM:
    def __init__(
        self, model: str = "accounts/fireworks/models/llama-v3p1-405b-instruct"
    ):
        self.client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.getenv("FIREWORKS_API_KEY"),
        )
        self.model = model

    def generate(
        self,
        instructions: str,
        messages: List[Dict[str, Any]],
        max_output_tokens: int = 250,
    ) -> str:
        from typing import cast
        from openai.types.chat import ChatCompletionMessageParam

        # Convert messages to proper type
        msgs = [{"role": "system", "content": instructions}] + messages
        typed_msgs = cast(List[ChatCompletionMessageParam], msgs)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=typed_msgs,  # type: ignore
            max_tokens=max_output_tokens,
        )

        content = resp.choices[0].message.content
        if content is None:
            return ""
        return content.strip()


# ---------- STT (faster-whisper) ----------


class STT:
    def __init__(
        self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_pcm16(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        audio = pcm16_bytes_to_float32(pcm16)
        segments, _info = self.model.transcribe(audio, language=None, beam_size=1)
        text = "".join(seg.text for seg in segments).strip()
        return text


# ---------- TTS (DeepInfra Kokoro-82M for English and Spanish) ----------


class TTS:
    def __init__(
        self, api_key: Optional[str] = None, lang: str = "en"
    ):  # Default English
        self.api_key = api_key
        if api_key is None:
            import warnings

            warnings.warn(
                "DeepInfra API key not provided. TTS will return silent audio."
            )
        self.endpoint = "https://api.deepinfra.com/v1/inference/hexgrad/Kokoro-82M"
        self.lang = lang
        self.sr = 24000
        self.voices = {
            "en": "am_adam",  # English voice example
            "es": "es_male1",  # Spanish voice example, adjust based on actual
        }

    def synth(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0
    ) -> np.ndarray:
        if self.api_key is None:
            # Return silent audio if no API key
            return np.zeros((24000,), dtype=np.float32)  # 1 second of silence

        if not voice:
            voice = self.voices.get(self.lang, "am_adam")
        payload = {"text": text, "voice": voice, "speed": speed, "lang_code": self.lang}
        headers = {"Authorization": f"bearer {self.api_key}"}
        response = requests.post(self.endpoint, json=payload, headers=headers)
        if response.status_code == 200:
            # Assume response.content is WAV bytes
            with io.BytesIO(response.content) as f:
                data, sr = sf.read(f, dtype="float32")
            return data
        else:
            print(f"TTS error: {response.text}")
            return np.zeros((0,), dtype=np.float32)


# ---------- VAD ----------


class VADSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        start_trigger_ms: int = 200,
        end_trigger_ms: int = 600,
    ):
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.bytes_per_frame = int(sample_rate * (frame_ms / 1000.0) * 2)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.start_trigger_frames = max(1, start_trigger_ms // frame_ms)
        self.end_trigger_frames = max(1, end_trigger_ms // frame_ms)
        self._buf = bytearray()
        self._in_speech = False
        self._speech_buf = bytearray()
        self._speech_run = 0
        self._silence_run = 0

    def push(self, pcm16: bytes) -> List[bytes]:
        self._buf.extend(pcm16)
        utterances = []

        while len(self._buf) >= self.bytes_per_frame:
            frame = bytes(self._buf[: self.bytes_per_frame])
            del self._buf[: self.bytes_per_frame]

            is_speech = self.vad.is_speech(frame, self.sr)

            if not self._in_speech:
                if is_speech:
                    self._speech_run += 1
                    self._speech_buf.extend(frame)
                    if self._speech_run >= self.start_trigger_frames:
                        self._in_speech = True
                        self._silence_run = 0
                else:
                    self._speech_run = 0
                    self._speech_buf.clear()
            else:
                self._speech_buf.extend(frame)
                if is_speech:
                    self._silence_run = 0
                else:
                    self._silence_run += 1
                    if self._silence_run >= self.end_trigger_frames:
                        utterances.append(bytes(self._speech_buf))
                        self._speech_buf.clear()
                        self._in_speech = False
                        self._speech_run = 0
                        self._silence_run = 0

        return utterances

    def flush(self) -> Optional[bytes]:
        if self._speech_buf and self._in_speech:
            out = bytes(self._speech_buf)
            self._speech_buf.clear()
            self._in_speech = False
            self._speech_run = 0
            self._silence_run = 0
            return out
        return None


# ---------- Conversation ----------


@dataclass
class Agent:
    id: str
    name: str
    instructions: str
    voice: str = "am_adam"
    lang: str = "en"  # Added language


@dataclass
class Session:
    session_id: str
    agents: Dict[str, Agent]
    transcript: List[Dict[str, str]] = field(default_factory=list)
    running: bool = False
    next_speaker: str = "A"
    cancel_speaking: asyncio.Event = field(default_factory=asyncio.Event)


def format_messages_for_agent(session: Session, agent_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in session.transcript:
        role = item["role"]
        who = item.get("who", "")
        text = item["text"]
        if role == "user":
            out.append({"role": "user", "content": text})
        elif role == "agent":
            if who == agent_id:
                out.append({"role": "assistant", "content": text})
            else:
                out.append({"role": "user", "content": text})
    return out[-40:]


# ---------- Globals ----------

llm = LLM()
stt = STT()

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")  # Set env var
tts_en = TTS(api_key=DEEPINFRA_API_KEY, lang="en")
tts_es = TTS(api_key=DEEPINFRA_API_KEY, lang="es")  # Spanish

sessions: Dict[str, Session] = {}


def get_or_create_session(session_id: str) -> Session:
    if session_id not in sessions:
        sessions[session_id] = Session(
            session_id=session_id,
            agents={
                "A": Agent(
                    id="A",
                    name="Agent A",
                    instructions="You are Agent A. Be helpful and curious.",
                    voice="am_adam",
                    lang="en",
                ),
                "B": Agent(
                    id="B",
                    name="Agent B",
                    instructions="You are Agent B. Be concise and witty.",
                    voice="es_male1",
                    lang="es",
                ),
            },
            transcript=[],
        )
    return sessions[session_id]


async def speak_agent_turn(ws: WebSocket, session: Session, agent_id: str):
    agent = session.agents[agent_id]
    msgs = format_messages_for_agent(session, agent_id)

    text = await asyncio.to_thread(
        llm.generate,
        agent.instructions,
        msgs,
        250,
    )
    if not text:
        text = "(...)"

    session.transcript.append({"role": "agent", "who": agent_id, "text": text})

    await ws.send_text(
        json.dumps(
            {
                "type": "agent_text",
                "agent": agent_id,
                "name": agent.name,
                "text": text,
                "ts": time.time(),
            }
        )
    )

    if session.cancel_speaking.is_set():
        return

    tts = tts_en if agent.lang == "en" else tts_es
    audio = await asyncio.to_thread(tts.synth, text, agent.voice, 1.0)
    wav = float32_to_wav_bytes(audio, tts.sr)

    await ws.send_text(
        json.dumps(
            {
                "type": "agent_audio",
                "agent": agent_id,
                "name": agent.name,
                "sr": tts.sr,
                "wav_b64": b64e(wav),
                "ts": time.time(),
            }
        )
    )


async def duet_loop(ws: WebSocket, session: Session, max_turns: int = 200):
    session.running = True
    turns = 0
    while session.running and turns < max_turns:
        session.cancel_speaking.clear()
        agent_id = session.next_speaker
        await speak_agent_turn(ws, session, agent_id)
        session.next_speaker = "B" if agent_id == "A" else "A"
        turns += 1
        await asyncio.sleep(0.05)
    session.running = False


# ---------- WebSocket ----------


@app.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    session = get_or_create_session(session_id)

    vad = VADSegmenter()
    duet_task: Optional[asyncio.Task] = None

    try:
        await ws.send_text(
            json.dumps(
                {
                    "type": "session_state",
                    "session_id": session_id,
                    "agents": {
                        k: {
                            "name": a.name,
                            "instructions": a.instructions,
                            "voice": a.voice,
                            "lang": a.lang,
                        }
                        for k, a in session.agents.items()
                    },
                }
            )
        )

        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            mtype = data.get("type")

            if mtype == "start_duet":
                if duet_task and not duet_task.done():
                    continue
                session.running = True
                duet_task = asyncio.create_task(duet_loop(ws, session))
                await ws.send_text(json.dumps({"type": "ok", "what": "duet_started"}))

            elif mtype == "stop":
                session.running = False
                session.cancel_speaking.set()
                if duet_task:
                    duet_task.cancel()
                await ws.send_text(json.dumps({"type": "ok", "what": "stopped"}))

            elif mtype == "interrupt":
                session.cancel_speaking.set()
                await ws.send_text(json.dumps({"type": "stop_audio"}))

            elif mtype == "set_agent":
                agent_id = data["agent"]
                if agent_id in session.agents:
                    if "instructions" in data:
                        session.agents[agent_id].instructions = data["instructions"]
                    if "voice" in data:
                        session.agents[agent_id].voice = data["voice"]
                    if "lang" in data:
                        session.agents[agent_id].lang = data["lang"]
                    await ws.send_text(
                        json.dumps(
                            {"type": "ok", "what": "agent_updated", "agent": agent_id}
                        )
                    )

            elif mtype == "user_text":
                text = (data.get("text") or "").strip()
                if text:
                    session.cancel_speaking.set()
                    await ws.send_text(json.dumps({"type": "stop_audio"}))
                    session.transcript.append({"role": "user", "text": text})
                    await ws.send_text(
                        json.dumps({"type": "user_text_ack", "text": text})
                    )

            elif mtype == "user_audio":
                chunk = b64d(data["pcm16_b64"])
                utterances = vad.push(chunk)
                for utt in utterances:
                    session.cancel_speaking.set()
                    await ws.send_text(json.dumps({"type": "stop_audio"}))
                    text = await asyncio.to_thread(stt.transcribe_pcm16, utt, 16000)
                    text = text.strip()
                    if text:
                        session.transcript.append({"role": "user", "text": text})
                        await ws.send_text(
                            json.dumps({"type": "user_text_ack", "text": text})
                        )

            elif mtype == "reset":
                session.running = False
                session.cancel_speaking.set()
                session.transcript.clear()
                session.next_speaker = "A"
                await ws.send_text(json.dumps({"type": "ok", "what": "reset"}))

            else:
                await ws.send_text(
                    json.dumps({"type": "error", "message": f"Unknown type: {mtype}"})
                )

    except WebSocketDisconnect:
        session.running = False
        session.cancel_speaking.set()
        if duet_task:
            duet_task.cancel()
