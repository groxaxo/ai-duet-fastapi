import { useState, useEffect, useRef } from 'react'
import './App.css'

// Configuration
const TTS_MAX_LENGTH = 500; // Maximum characters to send to TTS

// Conversation presets with different tones
const CONVERSATION_PRESETS = [
  {
    id: 'support',
    title: '💬 Customer Support',
    description: 'Help with a technical issue',
    prompt: "Hi! I'm having trouble setting up my new device. Can you help me get started?"
  },
  {
    id: 'brainstorm',
    title: '💡 Brainstorming',
    description: 'Creative ideas session',
    prompt: "I need help brainstorming ideas for a school project about renewable energy. Any suggestions?"
  },
  {
    id: 'roleplay',
    title: '🎭 Roleplay: Interview',
    description: 'Practice job interview',
    prompt: "Can you help me prepare for a job interview? Let's do a mock interview for a software engineer position."
  },
  {
    id: 'qa',
    title: '❓ Q&A Expert',
    description: 'Ask anything',
    prompt: "I've been wondering about how artificial intelligence actually works. Can you explain it simply?"
  },
  {
    id: 'onboarding',
    title: '🚀 Product Onboarding',
    description: 'Learn new software',
    prompt: "I'm new to using AI assistants. Can you give me a tour of what you can do?"
  },
  {
    id: 'bug-triage',
    title: '🐛 Bug Triage',
    description: 'Debug code issues',
    prompt: "My Python script keeps crashing with a TypeError. Can you help me debug it?"
  },
  {
    id: 'writing',
    title: '✍️ Writing Assistant',
    description: 'Help with writing',
    prompt: "I need help writing a professional email to my professor about extending a deadline."
  },
  {
    id: 'learning',
    title: '📚 Learning Tutor',
    description: 'Study and learn',
    prompt: "Can you help me understand calculus derivatives? I'm having trouble with the chain rule."
  },
  {
    id: 'planning',
    title: '📅 Event Planning',
    description: 'Plan an event',
    prompt: "I'm planning a birthday party for my friend. Can you help me organize the details?"
  },
  {
    id: 'fitness',
    title: '💪 Fitness Coach',
    description: 'Workout and health',
    prompt: "I want to start a workout routine but don't know where to begin. Can you help?"
  },
  {
    id: 'cooking',
    title: '👨‍🍳 Cooking Helper',
    description: 'Recipe and cooking tips',
    prompt: "I have chicken, rice, and vegetables. What's a quick and healthy meal I can make?"
  },
  {
    id: 'travel',
    title: '✈️ Travel Guide',
    description: 'Trip planning',
    prompt: "I'm planning a trip to Japan. What are some must-see places and cultural tips?"
  }
]

// Suggested follow-up prompts
const SUGGESTED_PROMPTS = [
  "Tell me more",
  "Can you give an example?",
  "What about alternatives?",
  "How does that work?",
  "Thanks! What else should I know?"
]

function App() {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [ttsModels, setTtsModels] = useState([])
  const [llmModels, setLlmModels] = useState([])
  const [selectedTtsModel, setSelectedTtsModel] = useState('')
  const [selectedLlmModel, setSelectedLlmModel] = useState('')
  const [isPlaying, setIsPlaying] = useState(false)
  const [showPresets, setShowPresets] = useState(true)
  const [showSettings, setShowSettings] = useState(false)
  const [currentAudio, setCurrentAudio] = useState(null)
  
  const chatContainerRef = useRef(null)
  const audioRef = useRef(null)

  // Load available models on mount
  useEffect(() => {
    fetchModels()
  }, [])

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [messages])

  const fetchModels = async () => {
    try {
      const [ttsRes, llmRes] = await Promise.all([
        fetch('/api/models/tts'),
        fetch('/api/models/llm')
      ])
      
      const ttsData = await ttsRes.json()
      const llmData = await llmRes.json()
      
      setTtsModels(ttsData.models || [])
      setSelectedTtsModel(ttsData.current || ttsData.models[0])
      
      setLlmModels(llmData.models || [])
      setSelectedLlmModel(llmData.current || llmData.models[0])
    } catch (error) {
      console.error('Failed to fetch models:', error)
    }
  }

  const sendMessage = async (text) => {
    if (!text.trim() || isLoading) return

    const userMessage = { role: 'user', content: text, timestamp: new Date() }
    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          llm_model: selectedLlmModel,
          conversation_history: messages
        })
      })

      const data = await response.json()

      if (data.success) {
        const assistantMessage = {
          role: 'assistant',
          content: data.response,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, assistantMessage])

        // Generate TTS for the response
        generateTTS(data.response)
      } else {
        throw new Error(data.error || 'Failed to get response')
      }
    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage = {
        role: 'system',
        content: '❌ Failed to send message. Please try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const generateTTS = async (text) => {
    try {
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.substring(0, TTS_MAX_LENGTH),
          tts_model: selectedTtsModel
        })
      })

      const data = await response.json()

      if (data.success) {
        playAudio(data.audio_b64)
      }
    } catch (error) {
      console.error('TTS error:', error)
    }
  }

  const playAudio = (base64Audio) => {
    try {
      // Stop current audio if playing
      if (currentAudio) {
        currentAudio.pause()
        currentAudio.currentTime = 0
      }

      const audio = new Audio(`data:audio/wav;base64,${base64Audio}`)
      setCurrentAudio(audio)
      
      audio.onplay = () => setIsPlaying(true)
      audio.onended = () => setIsPlaying(false)
      audio.onerror = () => setIsPlaying(false)
      
      audio.play()
    } catch (error) {
      console.error('Audio playback error:', error)
      setIsPlaying(false)
    }
  }

  const stopAudio = () => {
    if (currentAudio) {
      currentAudio.pause()
      currentAudio.currentTime = 0
      setIsPlaying(false)
    }
  }

  const handlePresetClick = (preset) => {
    setMessages([])
    sendMessage(preset.prompt)
    setShowPresets(false)
  }

  const handleSuggestedPrompt = (prompt) => {
    sendMessage(prompt)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(inputValue)
    }
  }

  return (
    <div className="app">
      <div className="messenger-container">
        <div className="window">
          {/* Window Title Bar */}
          <div className="window-title">
            <div className="window-title-text">
              <span className="window-icon">💬</span>
              <span>AI Duet Messenger - Conversation</span>
            </div>
            <div className="window-controls">
              <button className="window-btn" aria-label="Minimize">_</button>
              <button className="window-btn" aria-label="Maximize">□</button>
              <button className="window-btn" aria-label="Close">×</button>
            </div>
          </div>

          <div className="messenger-body">
            {/* Left Sidebar - Conversation Presets */}
            <div className={`sidebar-left ${!showPresets ? 'collapsed' : ''}`}>
              <div className="sidebar-header">
                <h3>💭 Conversation Ideas</h3>
                <button 
                  className="btn-icon"
                  onClick={() => setShowPresets(!showPresets)}
                  aria-label={showPresets ? "Hide presets" : "Show presets"}
                >
                  {showPresets ? '«' : '»'}
                </button>
              </div>
              {showPresets && (
                <div className="presets-list">
                  {CONVERSATION_PRESETS.map(preset => (
                    <button
                      key={preset.id}
                      className="preset-item"
                      onClick={() => handlePresetClick(preset)}
                      title={preset.description}
                    >
                      <div className="preset-title">{preset.title}</div>
                      <div className="preset-desc">{preset.description}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Center Panel - Chat */}
            <div className="chat-panel">
              {/* Status Bar */}
              <div className="status-bar">
                <div className="status-info">
                  <span className={`status-indicator ${llmModels.length > 0 ? 'status-online' : 'status-offline'}`}></span>
                  <span className="status-text">
                    {llmModels.length > 0 ? 'Connected' : 'Offline'}
                  </span>
                </div>
                <button 
                  className="btn-settings"
                  onClick={() => setShowSettings(!showSettings)}
                  aria-label="Settings"
                >
                  ⚙️ Settings
                </button>
              </div>

              {/* Chat Messages */}
              <div className="chat-container" ref={chatContainerRef}>
                {messages.length === 0 && (
                  <div className="welcome-message">
                    <h2>🎉 Welcome to AI Duet Messenger!</h2>
                    <p>Select a conversation preset from the left, or type your own message below.</p>
                  </div>
                )}
                
                {messages.map((msg, idx) => (
                  <div key={idx} className={`message-row ${msg.role}`}>
                    <div className={`chat-bubble chat-bubble-${msg.role}`}>
                      {msg.role === 'assistant' && <strong>🤖 Assistant:</strong>}
                      {msg.role === 'user' && <strong>You:</strong>}
                      <div className="message-content">{msg.content}</div>
                      <div className="message-time">
                        {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="message-row assistant">
                    <div className="chat-bubble chat-bubble-assistant loading">
                      <strong>🤖 Assistant:</strong>
                      <div className="typing-indicator">
                        <span></span><span></span><span></span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Suggested Prompts */}
              {messages.length > 0 && !isLoading && (
                <div className="suggested-prompts">
                  <span className="prompts-label">Quick replies:</span>
                  {SUGGESTED_PROMPTS.map((prompt, idx) => (
                    <button
                      key={idx}
                      className="btn-prompt"
                      onClick={() => handleSuggestedPrompt(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              )}

              {/* Input Area */}
              <div className="input-area">
                <div className="voice-controls">
                  <button 
                    className={`btn-voice ${isPlaying ? 'playing' : ''}`}
                    onClick={isPlaying ? stopAudio : null}
                    disabled={!isPlaying}
                    aria-label={isPlaying ? "Stop audio" : "No audio playing"}
                  >
                    {isPlaying ? '⏸️ Stop' : '🔊 Voice'}
                  </button>
                </div>
                <input
                  type="text"
                  className="input chat-input"
                  placeholder="Type a message..."
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={isLoading}
                  aria-label="Message input"
                />
                <button
                  className="btn btn-primary send-btn"
                  onClick={() => sendMessage(inputValue)}
                  disabled={isLoading || !inputValue.trim()}
                  aria-label="Send message"
                >
                  Send
                </button>
              </div>
            </div>

            {/* Right Panel - Settings */}
            {showSettings && (
              <div className="sidebar-right">
                <div className="sidebar-header">
                  <h3>⚙️ Settings</h3>
                  <button 
                    className="btn-icon"
                    onClick={() => setShowSettings(false)}
                    aria-label="Close settings"
                  >
                    ×
                  </button>
                </div>
                
                <div className="settings-content">
                  <div className="setting-group">
                    <label htmlFor="llm-select" className="setting-label">
                      🧠 LLM Model
                    </label>
                    <select
                      id="llm-select"
                      className="select"
                      value={selectedLlmModel}
                      onChange={(e) => setSelectedLlmModel(e.target.value)}
                    >
                      {llmModels.map(model => (
                        <option key={model} value={model}>{model}</option>
                      ))}
                    </select>
                    <p className="setting-hint">Choose the AI model for responses</p>
                  </div>

                  <div className="setting-group">
                    <label htmlFor="tts-select" className="setting-label">
                      🔊 TTS Model
                    </label>
                    <select
                      id="tts-select"
                      className="select"
                      value={selectedTtsModel}
                      onChange={(e) => setSelectedTtsModel(e.target.value)}
                    >
                      {ttsModels.map(model => (
                        <option key={model} value={model}>{model}</option>
                      ))}
                    </select>
                    <p className="setting-hint">Choose the voice synthesis model</p>
                  </div>

                  <div className="setting-group">
                    <button 
                      className="btn btn-primary"
                      style={{ width: '100%' }}
                      onClick={() => {
                        setMessages([])
                        setShowSettings(false)
                      }}
                    >
                      🔄 New Conversation
                    </button>
                  </div>

                  <div className="about-section">
                    <h4>About</h4>
                    <p>AI Duet Messenger Classic Edition</p>
                    <p>Built with ❤️ using React</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Status Bar at Bottom */}
        <div className="bottom-status-bar">
          <div className="status-item">
            💬 {messages.length} messages
          </div>
          <div className="status-item">
            🤖 {selectedLlmModel || 'No model'}
          </div>
          <div className="status-item">
            🔊 {selectedTtsModel || 'No TTS'}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
