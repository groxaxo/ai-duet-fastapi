# AI Duet FastAPI

A real-time conversational AI duet system where two AI agents converse with each other in multiple languages, with support for user interruption, voice interaction, and customizable personalities.

## Overview

This system enables two AI agents (Agent A and Agent B) to have real-time conversations with each other. Each agent has its own personality, language preference (English or Spanish), and voice characteristics. Users can listen to the conversation, interrupt with text or voice input, and customize agent behavior on the fly.

## Features

- **Real-time AI Duet**: Two AI agents converse back-and-forth automatically
- **Multi-language Support**: Agents can speak English or Spanish with appropriate TTS voices
- **Voice Interaction**: Users can speak to interrupt/participate via real-time STT
- **WebSocket API**: Real-time bidirectional communication for audio/text streaming
- **Customizable Agents**: Modify agent instructions, voices, and languages during runtime
- **Voice Activity Detection**: Real-time VAD for detecting speech segments
- **High-quality TTS**: DeepInfra Kokoro-82M for natural-sounding speech synthesis
- **Fast STT**: faster-whisper for accurate speech-to-text transcription
- **Powerful LLM**: Fireworks AI Llama v3.1 405B for intelligent conversation

## Architecture

```
┌─────────────────┐     WebSocket     ┌─────────────────┐
│   Client App    │◄────────────────►│   FastAPI Server │
└─────────────────┘                   └─────────────────┘
         │                                    │
         │ Audio/Text                        │ AI Services
         ▼                                    ▼
┌─────────────────┐                   ┌─────────────────┐
│   User Input    │                   │   LLM (Fireworks)│
│  - Voice (STT)  │                   │   TTS (DeepInfra)│
│  - Text Input   │                   │   STT (Whisper) │
└─────────────────┘                   └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- API keys for:
  - Fireworks AI (for LLM)
  - DeepInfra (for TTS)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-duet-fastapi
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run the server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Configuration

### Environment Variables

Create a `.env` file with the following:

```bash
# Required API Keys
FIREWORKS_API_KEY=your_fireworks_api_key_here
DEEPINFRA_API_KEY=your_deepinfra_api_key_here

# Optional Settings
WHISPER_MODEL_SIZE=small  # tiny, base, small, medium, large
WHISPER_DEVICE=cpu        # cpu or cuda
WHISPER_COMPUTE_TYPE=int8 # float16, int8, int8_float16
```

### Agent Configuration

Default agents are configured in `main.py`:

- **Agent A**: English-speaking, helpful and curious personality
- **Agent B**: Spanish-speaking, concise and witty personality

You can modify these defaults or update agents dynamically via the WebSocket API.

## Usage

### Starting the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`

### WebSocket Client

Connect to the WebSocket endpoint:
```
ws://localhost:8000/ws/{session_id}
```

Where `{session_id}` is a unique identifier for your session.

### Basic WebSocket Commands

#### 1. Start the Duet
```json
{"type": "start_duet"}
```

#### 2. Stop the Conversation
```json
{"type": "stop"}
```

#### 3. Interrupt Current Speaker
```json
{"type": "interrupt"}
```

#### 4. Send Text Input
```json
{"type": "user_text", "text": "Hello agents!"}
```

#### 5. Send Audio Input
```json
{
  "type": "user_audio",
  "pcm16_b64": "base64_encoded_pcm16_audio_here"
}
```

#### 6. Update Agent Settings
```json
{
  "type": "set_agent",
  "agent": "A",
  "instructions": "You are now a French philosopher.",
  "voice": "fr_male1",
  "lang": "fr"
}
```

#### 7. Reset Session
```json
{"type": "reset"}
```

### Server Responses

The server sends various message types:

- `session_state`: Initial session information
- `agent_text`: Agent's text response
- `agent_audio`: Agent's audio response (base64 encoded WAV)
- `user_text_ack`: Acknowledgement of user text
- `stop_audio`: Command to stop current audio playback
- `ok`: General success response
- `error`: Error messages

## API Reference

### WebSocket Endpoint

`GET /ws/{session_id}`

Establishes a WebSocket connection for real-time communication.

### Health Check

`GET /health`

Returns server status.

## Example Client

A basic HTML/JavaScript client is included in `client_example.html`. This provides:

- WebSocket connection management
- Real-time audio playback
- Text chat interface
- Voice recording (using browser MediaRecorder API)
- Controls for starting/stopping duet

## Advanced Configuration

### Changing LLM Model

Modify the `LLM` class initialization in `main.py`:

```python
class LLM:
    def __init__(self, model: str = "accounts/fireworks/models/llama-v3p1-405b-instruct"):
        # Change to other Fireworks models as needed
```

### Customizing TTS Voices

Update the `TTS` class voices dictionary:

```python
self.voices = {
    "en": "am_adam",      # English male
    "es": "es_male1",     # Spanish male
    "fr": "fr_female1",   # Add French voice
    # Add more languages as needed
}
```

### Adjusting VAD Parameters

Modify `VADSegmenter` initialization:

```python
vad = VADSegmenter(
    sample_rate=16000,
    frame_ms=20,
    aggressiveness=2,      # 0-3 (3 most aggressive)
    start_trigger_ms=200,  # ms of speech to start
    end_trigger_ms=600     # ms of silence to end
)
```

## Performance Considerations

- **STT**: Uses faster-whisper with `small` model by default for balance of speed/accuracy
- **TTS**: DeepInfra Kokoro-82M provides good quality with reasonable latency
- **LLM**: Fireworks AI offers fast inference with large context windows
- **Memory**: Transcripts limited to last 40 messages per agent
- **Concurrency**: Uses async/await for non-blocking I/O operations

## Troubleshooting

### Common Issues

1. **No audio output**: Check DeepInfra API key and network connectivity
2. **Slow responses**: Consider using smaller whisper model or GPU acceleration
3. **WebSocket disconnections**: Ensure client reconnects with same session_id
4. **High memory usage**: Reduce transcript history or use smaller models

### Logging

Enable debug logging by adding to `main.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

MIT License

## Acknowledgments

- [Fireworks AI](https://fireworks.ai) for LLM inference
- [DeepInfra](https://deepinfra.com) for TTS services
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [FastAPI](https://fastapi.tiangolo.com) for web framework

## Support

For issues and feature requests, please open an issue on the GitHub repository.