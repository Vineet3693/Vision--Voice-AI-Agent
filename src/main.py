"""
Main application entry point
Demonstrates the complete Vision + Voice AI Agent system
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config.settings import Config
from src.agents import initialize_agents
from src.utils.audio_utils import SpeechRecognizer, TextToSpeech

def demo_mode():
    """Run a demonstration of the system capabilities"""
    
    print("=" * 60)
    print("👁️  Vision + Voice AI Agent - Demo Mode")
    print("=" * 60)
    print()
    
    # Check configuration
    print("📋 Checking configuration...")
    errors = Config.validate()
    
    if errors:
        print("⚠️  Configuration warnings:")
        for error in errors:
            print(f"   - {error}")
        print()
        print("💡 To enable full functionality:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your API keys")
        print()
    else:
        print("✅ Configuration valid!")
        print()
    
    # Show architecture overview
    print("🏗 System Architecture:")
    print("   ┌─────────────────────────┐")
    print("   │  Master Orchestrator    │")
    print("   │  (Groq + Llama3)        │")
    print("   └───────────┬─────────────┘")
    print("               │")
    print("   ┌───────────┴─────────────┐")
    print("   │      Agent Router       │")
    print("   └───┬────┬────┬────┬──────┘")
    print("       │    │    │    │")
    print("   ┌───▼─┐ ┌▼──┐ ┌▼──┐ ┌────▼────┐")
    print("   │Vision│ │Mem│ │Chat│ │  Web    │")
    print("   │Agent │ │ory│ │Agent│ │ Search  │")
    print("   └──┬───┘ └▲──┘ └┬──┘ └─────────┘")
    print("      │       │     │")
    print("   ┌──▼───────▼─────▼──┐")
    print("   │  Gemini AI        │")
    print("   │  (Vision + Pro)   │")
    print("   └───────────────────┘")
    print()
    
    # Show features
    print("🌟 Key Features:")
    print("   ✅ Wake Word Detection ('Hey Vision')")
    print("   ✅ Dual-Layer Vision (YOLOv8 + Gemini)")
    print("   ✅ Multi-Agent Architecture")
    print("   ✅ Dual LLM Strategy (Groq + Gemini)")
    print("   ✅ Multilingual Support")
    print("   ✅ Emotion Detection")
    print("   ✅ Danger Detection")
    print("   ✅ Scene Change Detection")
    print("   ✅ Persistent Memory")
    print()
    
    # Show agents
    print("🤖 Active Agents:")
    print("   1. Vision Agent - Object detection & scene understanding")
    print("   2. Memory Agent - Conversation & object memory")
    print("   3. Chat Agent - General conversation (Groq)")
    print("   4. Web Agent - Internet search (DuckDuckGo)")
    print()
    
    print("=" * 60)
    print("🚀 Ready to start!")
    print("=" * 60)
    print()
    print("To run the full application:")
    print("   streamlit run src/dashboard.py")
    print()
    print("Or test individual components:")
    print("   python tests/test_vision_agent.py")
    print("   python tests/test_chat_agent.py")
    print()


def main():
    """Main entry point"""
    
    # Check if running demo or full app
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
        return
    
    # Try to initialize agents
    try:
        print("Initializing Vision + Voice AI Agent...")
        orchestrator = initialize_agents()
        print("✅ All agents initialized successfully!")
        
        # Example interaction (would be connected to voice/camera in real app)
        print("\n🎤 System ready! Listening...")
        
        # In a real implementation, this would:
        # 1. Listen for wake word
        # 2. Record audio
        # 3. Transcribe with Whisper
        # 4. Process through orchestrator
        # 5. Speak response with TTS
        
        print("(Run 'streamlit run src/dashboard.py' for interactive interface)")
        
    except ValueError as e:
        print(f"❌ Initialization failed: {e}")
        print("\nRunning in demo mode instead...")
        print()
        demo_mode()


if __name__ == "__main__":
    main()
