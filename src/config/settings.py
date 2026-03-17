"""
Configuration settings for the Vision + Voice AI Agent
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
    
    # Model Settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
    
    # Hardware Settings
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
    AUDIO_INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE", "default")
    AUDIO_OUTPUT_DEVICE = os.getenv("AUDIO_OUTPUT_DEVICE", "default")
    
    # Wake Word
    WAKE_WORD = "hey vision"
    
    # Danger Detection Keywords
    DANGER_KEYWORDS = [
        "fire", "smoke", "danger", "obstacle", 
        "person too close", "warning", "alert"
    ]
    
    # Supported Languages
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French"
    }
    
    # Memory Settings
    MAX_MEMORY_LENGTH = 10  # Number of conversations to remember
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == "your_gemini_api_key_here":
            errors.append("GEMINI_API_KEY not configured in .env file")
        
        if not cls.GROQ_API_KEY or cls.GROQ_API_KEY == "your_groq_api_key_here":
            errors.append("GROQ_API_KEY not configured in .env file")
        
        return errors
