# AI Duet FastAPI

A real-time multi-agent conversational AI system where multiple AI agents can converse with each other in various languages, with support for user interruption, voice interaction, customizable personalities, and flexible LLM provider options.

## Overview

This system enables multiple AI agents (configurable number) to have real-time conversations with each other. Each agent has its own personality, language preference, and voice characteristics. Users can listen to the conversation, interrupt with text or voice input, add or remove agents dynamically, and customize agent behavior and conversation flow on the fly.

## Features

- **🤖 Multi-Agent Duet**: Support for 2 or more AI agents conversing in rotation
- **➕ Dynamic Agent Management**: Add or remove agents during runtime
- **🌍 Multi-language Support**: Agents can speak English or Spanish with appropriate TTS voices
- **🎙️ Voice Interaction**: Users can speak to interrupt/participate via real-time STT (Push-to-Talk)
- **⚡ WebSocket API**: Real-time bidirectional communication for audio/text streaming
- **🎛️ Director Controls**: Configure max turns, turn length, stop phrases, and inter-turn delays
- **⏱️ Configurable Delays**: Adjust the delay between agent responses (0-10 seconds)
- **🔧 Customizable Agents**: Modify agent instructions, voices, and languages during runtime
- **🔊 Voice Activity Detection**: Real-time VAD for detecting speech segments
- **🎵 High-quality TTS**: DeepInfra Kokoro-82M for natural-sounding speech synthesis
- **📝 Fast STT**: faster-whisper for accurate speech-to-text transcription
- **🧠 Powerful LLM**: Support for multiple providers:
  - **Fireworks AI** (Llama v3.1 405B)
  - **OpenAI** (GPT-4, GPT-3.5, etc.)
  - **Ollama** (local models via OpenAI compatibility)
  - Any OpenAI-compatible endpoint

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
# Choose your LLM provider
LLM_PROVIDER=fireworks  # Options: "openai", "fireworks", "ollama"

# Fireworks AI (if using Fireworks)
FIREWORKS_API_KEY=your_fireworks_api_key_here
LLM_MODEL=accounts/fireworks/models/llama-v3p1-405b-instruct

# OpenAI (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4

# Ollama (if using Ollama - local models)
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2

# TTS Provider
DEEPINFRA_API_KEY=your_deepinfra_api_key_here

# Optional Settings
WHISPER_MODEL_SIZE=small  # tiny, base, small, medium, large
WHISPER_DEVICE=cpu        # cpu or cuda
WHISPER_COMPUTE_TYPE=int8 # float16, int8, int8_float16
LLM_TEMPERATURE=0.7       # 0.0-2.0
```

### LLM Provider Configuration

The system supports multiple LLM providers:

#### 1. Fireworks AI (Default)
```bash
LLM_PROVIDER=fireworks
FIREWORKS_API_KEY=your_key_here
LLM_MODEL=accounts/fireworks/models/llama-v3p1-405b-instruct
```

#### 2. OpenAI
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4
```

#### 3. Ollama (Local Models)
```bash
LLM_PROVIDER=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

#### 4. Any OpenAI-Compatible Endpoint
```bash
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://your-custom-endpoint.com/v1
OPENAI_API_KEY=your_key_here
LLM_MODEL=your_model_name
```

### Agent Configuration

Default agents are configured in `main.py`:

- **Agent A**: English-speaking, helpful and curious personality
- **Agent B**: Spanish-speaking, concise and witty personality

You can:
- Add more agents dynamically via the web UI or WebSocket API
- Modify agent properties (name, instructions, voice, language) at runtime
- Remove agents (minimum 1 agent required)

### Director Configuration

The Director controls the overall conversation flow:
- **Instructions**: Global rules for all agents
- **Max Turns**: Maximum number of turns before stopping
- **Turn Length**: short (160 tokens), medium (260 tokens), or long (420 tokens)
- **Stop Phrase**: Phrase that ends the conversation when detected
- **Delay Between Turns**: Configurable delay (0-10 seconds) between agent responses

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
  "name": "Alice",
  "instructions": "You are now a French philosopher.",
  "voice": "am_adam",
  "lang": "en"
}
```

#### 7. Add New Agent
```json
{
  "type": "add_agent",
  "agent_id": "C",
  "name": "Agent C",
  "instructions": "You are a helpful assistant.",
  "voice": "am_adam",
  "lang": "en"
}
```

#### 8. Remove Agent
```json
{
  "type": "remove_agent",
  "agent": "C"
}
```

#### 9. Update Director Settings
```json
{
  "type": "set_director",
  "director": {
    "instructions": "Keep responses brief and engaging.",
    "max_turns": 100,
    "turn_length": "short",
    "stop_phrase": "[[END]]",
    "delay_between_turns": 1.0
  }
}
```

#### 10. Reset Session
```json
{"type": "reset"}
```

### Server Responses

The server sends various message types:

- `session_state`: Initial session information (includes agents and director config)
- `agent_text`: Agent's text response
- `agent_audio`: Agent's audio response (base64 encoded WAV)
- `user_text_ack`: Acknowledgement of user text
- `stop_audio`: Command to stop current audio playback
- `duet_ended`: Notification when conversation ends
- `ok`: General success response (includes "what" field describing the action)
- `error`: Error messages

## Web UI

Access the web interface at `http://localhost:8000` after starting the server.

### UI Features

- **Real-time Transcript**: View the conversation as it happens
- **Agent Management**: Add, remove, and configure agents on the fly
- **Director Controls**: Adjust conversation parameters
  - Max turns limit
  - Response length (short/medium/long)
  - Stop phrase detection
  - Inter-turn delay configuration
- **PTT (Push-to-Talk)**: Hold the microphone button to speak
- **Text Input**: Type messages to inject into the conversation
- **Status Indicators**: Connection status and system state

## API Reference

### WebSocket Endpoint

`GET /ws/{session_id}`

Establishes a WebSocket connection for real-time communication.

### Health Check

`GET /health`

Returns server status.

## Example Client

A modern HTML/JavaScript client is included in `index.html` with a beautiful Tailwind CSS interface. This provides:

- WebSocket connection management
- Real-time audio playback
- Text chat interface with color-coded agents
- Voice recording using browser Push-to-Talk (PTT)
- Controls for starting/stopping duet
- Dynamic agent management (add/remove agents)
- Director configuration panel
- Turn delay adjustment
- Responsive design for desktop and mobile

## Advanced Configuration

### Changing LLM Model

Set the `LLM_MODEL` environment variable:

```bash
# For Fireworks
LLM_MODEL=accounts/fireworks/models/llama-v3p1-405b-instruct

# For OpenAI
LLM_MODEL=gpt-4

# For Ollama
LLM_MODEL=llama3.2
```

### Adding More Agents

Agents can be added dynamically through the web UI or via WebSocket:

```javascript
ws.send(JSON.stringify({
  type: "add_agent",
  agent_id: "C",
  name: "Charlie",
  instructions: "You are a creative storyteller.",
  voice: "am_adam",
  lang: "en"
}));
```

### Adjusting Turn Delays

Control the pacing of conversations by adjusting the delay between turns:

```javascript
ws.send(JSON.stringify({
  type: "set_director",
  director: {
    delay_between_turns: 2.0  // 2 seconds between each agent's turn
  }
}));
```

Or use the UI slider in the Director Settings panel.

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