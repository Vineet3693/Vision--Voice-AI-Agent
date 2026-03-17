# 🚀 Getting Started Guide

## Vision + Voice AI Agent with Gemini AI

This guide will help you set up and run the advanced multi-agent AI system.

---

## 📋 Prerequisites

- Python 3.9 or higher
- Webcam (for vision features)
- Microphone (for voice features)
- API Keys:
  - [Google Gemini API Key](https://makersuite.google.com/app/apikey)
  - [Groq API Key](https://console.groq.com/keys)

---

## 🔧 Installation Steps

### Step 1: Clone/Navigate to Project
```bash
cd /workspace
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** Some packages like `pyaudio` may require system dependencies:
- **Ubuntu/Debian:** `sudo apt-get install portaudio19-dev python3-pyaudio`
- **macOS:** `brew install portaudio`
- **Windows:** Download pre-built wheels from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

### Step 4: Configure API Keys
```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```
GEMINI_API_KEY=your_actual_gemini_key_here
GROQ_API_KEY=your_actual_groq_key_here
```

---

## 🎯 Running the Application

### Option 1: Streamlit Dashboard (Recommended)
```bash
streamlit run src/dashboard.py
```

This opens a web interface at `http://localhost:8501` with:
- Live camera feed
- Real-time object detection
- Conversation interface
- Danger alerts
- Spatial awareness display

### Option 2: Demo Mode
```bash
python src/main.py --demo
```

Shows system architecture and capabilities without requiring API keys.

### Option 3: Test Individual Components
```bash
# Test vision agent
python tests/test_vision.py

# Test chat agent (requires GROQ_API_KEY)
python tests/test_chat_agent.py
```

---

## 🏗 System Architecture Overview

```
User Voice/Input
      ↓
Wake Word Detection ("Hey Vision")
      ↓
Speech Recognition (Whisper)
      ↓
Master Orchestrator (Groq + Llama3)
      ↓
Agent Router
      ↓
┌─────────┬──────────┬────────┬──────────┐
│ Vision  │  Memory  │  Chat  │   Web    │
│ Agent   │  Agent   │ Agent  │  Agent   │
│(Gemini+ │(LangChain│(Groq)  │(DuckDuck-│
│ YOLOv8) │          │        │    Go)   │
└────┬────┴────┬─────┴───┬────┴────┬─────┘
     │         │         │         │
     └─────────┴────┬────┴─────────┘
                    ↓
         Response Generator (Gemini Pro)
                    ↓
              Text-to-Speech (Coqui TTS)
                    ↓
              Speaker Output
```

---

## 🤖 The Four Agents

### 1. Vision Agent
**Purpose:** Everything related to camera and seeing

**Two Layers:**
- **Layer 1 (Fast):** YOLOv8 detects objects instantly
- **Layer 2 (Smart):** Gemini Vision provides deep scene understanding

**Example:**
```
YOLOv8: laptop, bottle, person
Gemini: "A person is sitting at a desk working on a laptop. 
         A water bottle is on their right side."
```

### 2. Memory Agent
**Purpose:** Remember conversations and detected objects

**Features:**
- Conversation history
- Object persistence tracking
- Context-aware responses

**Example:**
```
User: What is in front of me?
AI: I see a laptop and a bottle.

User: Where is the bottle?
AI: The bottle is to the right of your laptop.
```

### 3. Chat Agent
**Purpose:** General conversation and Q&A

**Powered by:** Groq + Llama3 (super fast)

**Example:**
```
User: What is the capital of France?
AI: The capital of France is Paris.
```

### 4. Web Search Agent
**Purpose:** Search internet for current information

**Tool:** DuckDuckGo Search API (free)

**Example:**
```
User: What is the latest news about AI?
AI: [Searches and provides recent AI news]
```

---

## 🌟 Advanced Features

### Wake Word Detection
Say "Hey Vision" to activate the system hands-free.

### Danger Detection
Automatic warnings for:
- 🔥 Fire or smoke
- ⚠️ Obstacles in path
- 👤 People too close
- 🚨 Other hazards

### Multilingual Support
Speak and receive responses in:
- English
- Hindi
- Spanish
- French

### Emotion Detection
Gemini Vision can detect if you look stressed or tired and respond appropriately.

### Scene Change Detection
Proactive notifications when:
- New object appears
- Person enters frame
- Environment changes

---

## 📊 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Speed Layer** | Groq + Llama3 | Fast intent detection |
| **Vision Layer** | Gemini Vision | Deep image understanding |
| **Reasoning** | Gemini Pro | Smart responses |
| **Memory** | LangChain | Conversation & object memory |
| **Detection** | YOLOv8 | Real-time object detection |
| **Speech→Text** | Whisper | Voice recognition |
| **Text→Speech** | Coqui TTS | Voice output |
| **Interface** | Streamlit | Visual dashboard |
| **Web Search** | DuckDuckGo | Internet queries |

---

## 🎯 Use Cases

### Primary: AI Vision Assistant for Visually Impaired
- Describe surroundings
- Warn about dangers
- Locate objects
- Answer questions about environment
- Remember previous interactions

### Secondary Applications
- Security monitoring
- Elderly assistance
- Educational tool
- Robotics vision system
- Smart home integration

---

## 🐛 Troubleshooting

### Camera Not Working
```bash
# Check camera permissions
ls -l /dev/video0

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### API Key Errors
- Ensure `.env` file exists in project root
- Check keys are copied correctly (no extra spaces)
- Verify API keys are active on provider websites

### Audio Issues
```bash
# List audio devices
arecord -l  # Linux
# or
python -c "import pyaudio; p = pyaudio.PyAudio(); print(p.get_device_count())"
```

### Slow Performance
- Use smaller YOLO model: `YOLO_MODEL=yolov8n.pt` in `.env`
- Reduce camera resolution in dashboard settings
- Close other applications using GPU

---

## 📈 Project Strength

This architecture represents a **top 1% student AI project**:

✅ **Dual-LLM Multi-Agent System** - Extremely rare in portfolios  
✅ **Real-World Social Impact** - Helps visually impaired people  
✅ **Production-Grade Architecture** - Modular, scalable design  
✅ **Multiple AI Models** - Groq, Gemini, YOLOv8, Whisper working together  
✅ **Proactive AI Behavior** - Not just reactive, but anticipatory  

---

## 🎓 Learning Outcomes

By studying this project, you'll understand:

1. **Multi-Agent Systems** - How to coordinate multiple AI agents
2. **LLM Orchestration** - Routing tasks to appropriate models
3. **Computer Vision** - Object detection + scene understanding
4. **Speech Processing** - STT and TTS integration
5. **Memory Systems** - Conversation and context management
6. **Real-Time Systems** - Handling streaming video/audio
7. **API Integration** - Multiple external services
8. **Accessibility Tech** - Building for social impact

---

## 📝 Next Steps

1. ✅ Run demo mode to understand architecture
2. ✅ Set up API keys
3. ✅ Launch Streamlit dashboard
4. ✅ Test each agent individually
5. ✅ Customize for your use case
6. ✅ Deploy to cloud (optional)

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new agents
- Improve existing features
- Add more languages
- Enhance the UI
- Optimize performance

---

## 📄 License

Educational use - Built for learning and portfolio demonstration

---

**Built with ❤️ for accessibility and innovation**

*Questions? Check the README.md or explore individual agent files in `src/agents/`*
