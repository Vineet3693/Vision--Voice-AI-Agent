# 🧠 Vision + Voice AI Agent with Gemini AI

## Advanced Multi-Agent System for Visually Impaired Assistance

A state-of-the-art dual-LLM multi-agent AI system that combines **Groq + Llama3** for speed-critical tasks and **Google Gemini AI** for vision understanding and deep reasoning.

### 🌟 Key Features

- **🎤 Wake Word Detection** - "Hey Vision" activation
- **👁️ Dual-Layer Vision** - YOLOv8 (fast detection) + Gemini Vision (deep understanding)
- **🧠 Multi-Agent Architecture** - Vision, Memory, Chat, and Web Search agents
- **⚡ Dual LLM Strategy** - Groq for speed, Gemini for intelligence
- **🌍 Multilingual Support** - English, Hindi, Spanish, French
- **😄 Emotion Detection** - Detects user emotional state
- **⚠️ Danger Detection** - Fire, smoke, obstacles, proximity alerts
- **🔄 Scene Change Detection** - Proactive environment monitoring
- **💾 Persistent Memory** - Remembers conversations and objects

### 🏗 System Architecture

```
User Voice → Wake Word → Whisper STT → Master Orchestrator (Groq)
                                              ↓
                                    Agent Router
                                              ↓
        ┌─────────────┬─────────────┬─────────────┬─────────────┐
        ↓             ↓             ↓             ↓             ↓
   Vision Agent   Memory Agent   Chat Agent   Web Agent   Danger Detection
   (Gemini+YOLO)  (LangChain)    (Groq)      (DuckDuckGo)  (Gemini Vision)
        ↓             ↓             ↓             ↓             ↓
        └─────────────┴─────────────┴─────────────┴─────────────┘
                                              ↓
                                    Response Generator (Gemini Pro)
                                              ↓
                                          Coqui TTS
                                              ↓
                                      Streamlit Dashboard
```

### 🤖 The Four Agents

1. **Vision Agent**: Object detection (YOLOv8) + Scene understanding (Gemini Vision)
2. **Memory Agent**: Conversation history and object memory (LangChain)
3. **Chat Agent**: General Q&A (Groq + Llama3)
4. **Web Search Agent**: Internet queries (DuckDuckGo API)

### 🚀 Quick Start

#### Prerequisites

- Python 3.9+
- Google Gemini API Key
- Groq API Key
- Webcam/Microphone

#### Installation

```bash
pip install -r requirements.txt
```

#### Configuration

1. Copy `.env.example` to `.env`
2. Add your API keys:
   ```
   GEMINI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```

#### Run the Application

```bash
streamlit run src/dashboard.py
```

### 📊 Technical Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Speed Layer | Groq + Llama3 | Fast intent detection |
| Vision Layer | Gemini Vision | Deep image understanding |
| Reasoning Layer | Gemini Pro | Smart response generation |
| Memory Layer | LangChain | Conversation & object memory |
| Detection Layer | YOLOv8 | Real-time object detection |
| Speech Layer | Whisper | Voice-to-text |
| Voice Layer | Coqui TTS | Text-to-speech |
| Interface Layer | Streamlit | Visual dashboard |

### 🎯 Social Impact

This project is designed as an **AI Vision Assistant for Visually Impaired People**, providing:

- Real-time environment description
- Danger warnings (fire, obstacles, people)
- Object location and identification
- Conversational memory for context-aware assistance
- Multilingual support for global accessibility

### 📈 Project Strength

This advanced architecture represents a **top 1% student AI project** with:
- ✅ Dual-LLM multi-agent system (extremely rare in portfolios)
- ✅ Real-world social impact application
- ✅ Production-grade architecture
- ✅ Multiple AI models working in concert
- ✅ Proactive rather than reactive AI behavior

---

*Built with ❤️ for accessibility and innovation*
