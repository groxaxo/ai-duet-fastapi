# AI Duet FastAPI

A sophisticated real-time conversational AI system featuring two AI agents (LLM↔LLM) that converse with each other, complete with user interruption, push-to-talk (PTT), persistent memory, and customizable personalities.

## Overview

AI Duet enables two AI agents to have autonomous conversations while allowing users to interrupt, inject text, change system prompts, select voices, and manage conversation flow through a director. Each agent has persistent memory powered by Memvid SDK, creating more coherent and context-aware conversations.

## Key Features

### Core Capabilities
- **2 AI Agents (LLM↔LLM)**: Two agents converse autonomously with distinct personalities
- **User Interaction**: Interrupt anytime, inject text mid-topic, or use push-to-talk (PTT)
- **Real-time Voice**: Speech-to-text (STT) and text-to-speech (TTS) with voice customization
- **Persistent Memory**: Per-agent memory using Memvid .mv2 format for long-term context
- **Director Controls**: Set max turns, turn length, stop phrases, and conversation rules
- **Multiple LLM Providers**: OpenAI, Fireworks, Ollama, or any OpenAI-compatible endpoint
- **Flexible STT/TTS**: Local (faster-whisper + Kokoro) or cloud (DeepInfra)

### Agent Customization
- Modify system prompts on-the-fly
- Select from multiple TTS voices (dynamically loaded)
- Set debate topics with automatic agent positioning
- Individual memory management per agent

### Memory System
- **Memvid Integration**: Persistent, searchable memory for each agent
- **Memory Modes**: Lexical (lex), semantic (sem), or hybrid search
- **Memory Actions**: Put, find, wipe, and save session snapshots
- **Context Retrieval**: Automatically pulls relevant memories during conversations

## Architecture

```
┌─────────────────┐     WebSocket     ┌─────────────────┐
│   Web Client    │◄────────────────►│   FastAPI Server │
│  (index.html)   │                   │    (main.py)     │
└─────────────────┘                   └─────────────────┘
         │                                     │
         │ Audio/Text                         │ Providers
         ▼                                     ▼
┌─────────────────┐                   ┌─────────────────┐
│   User Input    │                   │  LLM Provider   │
│  - PTT Voice    │                   │  STT Provider   │
│  - Text Input   │                   │  TTS Provider   │
│  - Controls     │                   │  Memvid Memory  │
└─────────────────┘                   └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- API keys (depending on provider choice):
  - OpenAI API key (for OpenAI LLM)
  - OR Fireworks API key (for Fireworks LLM)
  - DeepInfra API key (optional, for cloud STT/TTS)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-duet-fastapi
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and preferences
   ```

4. **Run the server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Open the UI**:
   Navigate to `http://localhost:8000` in your browser

## Configuration

### LLM Provider Options

#### OpenAI (Default)
```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"
export LLM_API="chat"
```

#### Fireworks AI
```bash
export OPENAI_API_KEY="fw_..."
export OPENAI_BASE_URL="https://api.fireworks.ai/inference/v1"
export LLM_API="chat"
export LLM_MODEL="accounts/fireworks/models/deepseek-v3"
```

#### Ollama (Local)
```bash
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export LLM_API="chat"  # or "responses"
export LLM_MODEL="llama3.2"
```

### STT/TTS Provider Options

#### Local (Default)
```bash
export STT_PROVIDER="local"
export TTS_PROVIDER="local"
export WHISPER_MODEL="small"
export WHISPER_DEVICE="cpu"
export KOKORO_LANG="a"
```

#### DeepInfra (Cloud)
```bash
export STT_PROVIDER="deepinfra"
export TTS_PROVIDER="deepinfra"
export DEEPINFRA_API_KEY="your_key"
```

### Memory Configuration

```bash
export MEMORY_DEFAULT_ENABLED="true"
export MEMORY_DEFAULT_K="6"
export MEMORY_DEFAULT_MODE="lex"  # lex, sem, or hyb
```

## Usage

### Web Interface

The web UI provides a comprehensive control panel with:

1. **Status & Controls**
   - Start/Stop duet
   - Interrupt current speaker
   - Connection status

2. **User Input**
   - Text injection: Type and send messages mid-conversation
   - Push-to-Talk (PTT): Hold mic button to speak
   - Quick debate topics: Set opposing viewpoints instantly

3. **Memory Management**
   - Enable/disable memory
   - Configure retrieval parameters (k, mode)
   - Wipe individual agent memories
   - Save session snapshots

4. **Director Settings**
   - Set conversation rules
   - Configure max turns and turn length
   - Define stop phrases

5. **Agent Configuration**
   - Customize system prompts
   - Select voices (dynamically loaded)
   - Update settings on-the-fly

### WebSocket API

Connect to `ws://localhost:8000/ws/{session_id}`

#### Available Commands

**Start/Stop Conversation**
```json
{"type": "start_duet"}
{"type": "stop"}
{"type": "interrupt"}
```

**User Input**
```json
{"type": "user_text", "text": "Your message here"}
{"type": "user_audio", "pcm16_b64": "base64_encoded_audio"}
```

**Agent Configuration**
```json
{
  "type": "set_agent",
  "agent": "A",
  "instructions": "You are a philosopher...",
  "voice": "af_heart"
}
```

**Director Configuration**
```json
{
  "type": "set_director",
  "director": {
    "instructions": "Keep it brief",
    "max_turns": 100,
    "turn_length": "short",
    "stop_phrase": "[[END]]"
  }
}
```

**Memory Operations**
```json
{"type": "set_memory", "enabled": true, "k": 6, "mode": "lex"}
{"type": "wipe_memory", "agent": "A"}
{"type": "save_session_to_memory"}
```

**Quick Debate**
```json
{"type": "set_topic", "topic": "AI Safety"}
```

#### Server Responses

- `session_state`: Initial session configuration
- `agent_text`: Agent's text response
- `agent_audio`: Agent's audio (base64 WAV)
- `user_text_ack`: User message acknowledged
- `memory_retrieved`: Memory context loaded
- `stop_audio`: Stop current playback
- `ok`: Operation successful
- `error`: Error message

## API Endpoints

### `GET /`
Serves the web UI (index.html)

### `GET /voices`
Returns list of available TTS voices
```json
{"voices": ["af_heart", "am_michael", "am_adam", "af_sky", ...]}
```

### `WS /ws/{session_id}`
WebSocket endpoint for real-time communication

## Advanced Features

### Turn Length Configuration
- **Short**: ~160 tokens
- **Medium**: ~260 tokens (default)
- **Long**: ~420 tokens

### Memory Retrieval
- Automatically retrieves top-k relevant memories before each agent turn
- Supports lexical (keyword) and semantic (meaning) search
- Presents memories as context to the LLM

### Voice Activity Detection (VAD)
- Real-time speech detection with configurable sensitivity
- ~600ms silence cutoff for natural conversation flow
- Optimized for 16kHz mono PCM audio

## Development

### Project Structure
```
ai-duet-fastapi/
├── main.py              # FastAPI backend with all providers
├── index.html           # Tailwind UI frontend
├── requirements.txt     # Python dependencies
├── .env.example         # Environment configuration template
├── README.md           # This file
└── memories/           # Per-session memory storage (created at runtime)
```

### Key Classes

- **LLMProvider**: Unified interface for OpenAI/Fireworks/Ollama
- **STTLocalFasterWhisper**: Local speech-to-text
- **TTSLocalKokoro**: Local text-to-speech
- **MemvidAgentMemory**: Per-agent persistent memory
- **VADSegmenter**: Voice activity detection
- **Session**: Manages conversation state
- **Agent**: Represents individual AI agent
- **Director**: Controls conversation flow

## Performance Considerations

- **Local STT**: faster-whisper on CPU is fast for "small" model
- **Local TTS**: Kokoro provides good quality with reasonable latency
- **Memory**: Transcript limited to last 30 messages per agent for context
- **Concurrency**: Async/await for non-blocking operations
- **Memory Retrieval**: Configurable k (default 6) balances context vs speed

## Troubleshooting

### Common Issues

1. **Memvid not available**: Memory features will be disabled. Install with `pip install memvid-sdk`
2. **No audio output**: Check TTS provider configuration and API keys
3. **Slow responses**: Consider using smaller whisper model or GPU acceleration
4. **WebSocket disconnections**: Client should reconnect with same session_id to preserve state

### Debug Logging

Enable debug mode in development:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

MIT License

## Acknowledgments

- [OpenAI](https://openai.com) for GPT models and API standards
- [Fireworks AI](https://fireworks.ai) for fast LLM inference
- [Ollama](https://ollama.com) for local LLM hosting
- [DeepInfra](https://deepinfra.com) for cloud STT/TTS services
- [Memvid](https://memvid.com) for persistent memory SDK
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Kokoro](https://github.com/hexgrad/kokoro) for text-to-speech
- [FastAPI](https://fastapi.tiangolo.com) for the web framework
- [Tailwind CSS](https://tailwindcss.com) for the UI styling

## Support

For issues and feature requests, please open an issue on the GitHub repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
