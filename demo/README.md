# AI Duet Messenger - Classic Edition 💬

A retro-inspired MSN Messenger-style demo showcasing the AI Duet project end-to-end. Built with React and featuring a polished "classic messenger" aesthetic reminiscent of MSN Messenger circa 2000.

## Features

### 🎨 Retro Design
- Classic MSN Messenger aesthetics
- Soft gradients and subtle bevels
- Authentic window chrome and controls
- Pixel-perfect chat bubble styling
- Smooth, delightful animations

### 💬 Conversation Presets
12 pre-generated conversation ideas covering different scenarios:
- 💬 Customer Support
- 💡 Brainstorming
- 🎭 Roleplay: Interview
- ❓ Q&A Expert
- 🚀 Product Onboarding
- 🐛 Bug Triage
- ✍️ Writing Assistant
- 📚 Learning Tutor
- 📅 Event Planning
- 💪 Fitness Coach
- 👨‍🍳 Cooking Helper
- ✈️ Travel Guide

### ⚙️ Settings Panel
- **LLM Model Selector**: Choose from available AI models
- **TTS Model Selector**: Select voice synthesis model
- Real-time model switching
- Clean settings interface

### 🔊 Voice Features
- Text-to-Speech for AI responses
- Audio playback controls
- Visual feedback for playing state

### 📱 Responsive Design
- **Desktop**: Multi-column messenger layout
- **Mobile**: Collapsible panels, chat-first design
- Adaptive UI components
- Touch-friendly controls

### ♿ Accessibility
- Keyboard-friendly navigation
- ARIA labels throughout
- Readable contrast ratios
- Focus indicators
- Reduced motion support

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+ (for backend)
- API keys (OpenAI, Fireworks, or DeepInfra)

### Installation

1. **Install Backend Dependencies**
   ```bash
   cd /path/to/ai-duet-fastapi
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install Demo Dependencies**
   ```bash
   cd demo
   npm install
   ```

### Running the Demo

**Option 1: Development Mode (Separate Servers)**

1. Start the backend server:
   ```bash
   # From project root
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. In a new terminal, start the React dev server:
   ```bash
   # From demo directory
   cd demo
   npm run dev
   ```

3. Open http://localhost:3000 in your browser

**Option 2: Production Mode (Integrated)**

1. Build the React app:
   ```bash
   cd demo
   npm run build
   ```

2. The built files will be in `demo/dist/`. Configure your backend to serve these static files, or access the demo through the FastAPI backend at http://localhost:8000/demo

## Project Structure

```
demo/
├── src/
│   ├── App.jsx          # Main messenger component
│   ├── App.css          # Layout and component styles
│   ├── index.css        # Global styles and retro theme
│   └── main.jsx         # React entry point
├── public/              # Static assets
├── index.html           # HTML template
├── package.json         # Dependencies
└── vite.config.js       # Vite configuration
```

## API Endpoints Used

- `GET /api/models/tts` - Get available TTS models
- `GET /api/models/llm` - Get available LLM models
- `POST /api/chat` - Send message and get AI response
- `POST /api/tts` - Generate TTS audio for text

## Environment Variables

Backend API keys (in project root `.env`):

```bash
# LLM Provider (choose one)
OPENAI_API_KEY="sk-..."           # For OpenAI
OPENAI_BASE_URL="..."             # For Fireworks/Ollama
LLM_MODEL="gpt-4o-mini"

# TTS/STT Providers
TTS_PROVIDER="deepinfra"          # or "local"
STT_PROVIDER="deepinfra"          # or "local"
DEEPINFRA_API_KEY="..."
DEEPINFRA_TTS_MODEL="hexgrad/Kokoro-82M"
```

## Customization

### Adding More Presets
Edit `CONVERSATION_PRESETS` in `src/App.jsx`:

```javascript
{
  id: 'custom',
  title: '🎯 Custom Title',
  description: 'Brief description',
  prompt: "Initial message to send..."
}
```

### Changing Suggested Prompts
Modify `SUGGESTED_PROMPTS` array in `src/App.jsx`.

### Styling
- Global theme: `src/index.css`
- Component layout: `src/App.css`
- CSS variables in `:root` for easy color customization

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari 14+, Chrome Android 90+)

## Performance

- Lazy loading for optimal startup
- Efficient re-renders with React hooks
- Optimized animations
- Responsive images and assets

## Troubleshooting

**Issue**: API calls failing
- **Solution**: Ensure backend is running on port 8000 and check CORS settings

**Issue**: TTS not working
- **Solution**: Check TTS provider configuration and API keys in backend `.env`

**Issue**: Models not loading
- **Solution**: Verify API endpoints are accessible and returning data

**Issue**: Mobile layout broken
- **Solution**: Clear cache, ensure responsive CSS is loaded

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Inspired by MSN Messenger (circa 2000)
- Built with React and Vite
- Powered by AI Duet FastAPI backend
- Font: Tahoma, Segoe UI (system fonts)
