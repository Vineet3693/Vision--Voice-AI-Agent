"""
Utility functions for speech processing, TTS, and audio handling
"""
import numpy as np

# Lazy imports to avoid errors when packages not installed
def get_whisper_model(model_name):
    """Lazy load Whisper model"""
    try:
        import whisper
        return whisper.load_model(model_name)
    except ImportError:
        print("Warning: whisper not installed. Speech recognition unavailable.")
        return None

class SpeechRecognizer:
    """Handles speech-to-text using Whisper"""
    
    def __init__(self):
        """Initialize Whisper model"""
        from src.config.settings import Config
        self.model = get_whisper_model(Config.WHISPER_MODEL)
        
    def transcribe(self, audio_data):
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            str: Transcribed text
        """
        if self.model is None:
            return "Speech recognition not available (whisper not installed)"
        
        try:
            result = self.model.transcribe(audio_data)
            return result['text'].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def transcribe_file(self, audio_file_path):
        """
        Transcribe audio from file
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            str: Transcribed text
        """
        return self.transcribe(audio_file_path)


class TextToSpeech:
    """Handles text-to-speech using Coqui TTS"""
    
    def __init__(self):
        """Initialize TTS engine"""
        try:
            from TTS.api import TTS
            self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress=False)
            self.available = True
        except Exception as e:
            print(f"TTS initialization warning: {e}")
            print("TTS will be unavailable, using fallback")
            self.available = False
            self.tts = None
    
    def speak(self, text, output_file="output.wav"):
        """
        Convert text to speech and save to file
        
        Args:
            text: Text to convert
            output_file: Output audio file path
            
        Returns:
            str: Path to generated audio file or None
        """
        if not self.available:
            print(f"[TTS Fallback] Would speak: {text}")
            return None
        
        try:
            self.tts.tts_to_file(text=text, file_path=output_file)
            return output_file
        except Exception as e:
            print(f"TTS error: {e}")
            return None
    
    def speak_multilingual(self, text, language='en', output_file="output.wav"):
        """
        Generate speech in different languages
        
        Args:
            text: Text to convert
            language: Language code (en, hi, es, fr)
            output_file: Output file path
            
        Returns:
            str: Path to audio file or None
        """
        # Note: For multilingual support, you'd need to load appropriate models
        # This is a simplified version
        if language != 'en':
            print(f"Multilingual TTS for {language} requires additional model setup")
        
        return self.speak(text, output_file)


def record_audio(duration=5, sample_rate=16000):
    """
    Record audio from microphone
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
        
    Returns:
        numpy.ndarray: Audio data
    """
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        frames = []
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        print(f"Recording for {duration} seconds...")
        
        for _ in range(int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_array
        
    except Exception as e:
        print(f"Audio recording error: {e}")
        return None


def play_audio(file_path):
    """
    Play audio file
    
    Args:
        file_path: Path to audio file
    """
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        stream = p.open(
            format=p.get_format_from_width(2),
            channels=1,
            rate=16000,
            output=True
        )
        
        stream.write(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    except Exception as e:
        print(f"Audio playback error: {e}")
