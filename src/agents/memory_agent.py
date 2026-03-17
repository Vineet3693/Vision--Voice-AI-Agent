"""
Memory Agent - Conversation and object memory using LangChain
"""
from collections import deque
import json
from datetime import datetime
from src.config.settings import Config


class ConversationBufferMemory:
    """Conversation memory buffer for storing interaction history"""
    def __init__(self, return_messages=True, max_token_limit=1000):
        self.return_messages = return_messages
        self.max_token_limit = max_token_limit
        self.messages = []
        self.buffer = ""
    
    def save_context(self, inputs, outputs):
        """Save context from this conversation"""
        self.messages.append({"input": inputs, "output": outputs})
        self.buffer += f"\nInput: {inputs}\nOutput: {outputs}"
    
    def load_memory_variables(self, inputs):
        """Load memory variables"""
        return {"history": self.buffer}

class MemoryAgent:
    """Handles all memory-related tasks"""
    
    def __init__(self):
        """Initialize Memory Agent with LangChain"""
        # Conversation memory
        self.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            max_token_limit=1000
        )
        
        # Object memory (recent detections)
        self.object_memory = deque(maxlen=Config.MAX_MEMORY_LENGTH)
        
        # Session data
        self.session_start = datetime.now()
        self.interaction_count = 0
        
    def add_conversation(self, user_input, ai_response):
        """
        Add conversation to memory
        
        Args:
            user_input: User's question/statement
            ai_response: AI's response
        """
        self.conversation_memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )
        self.interaction_count += 1
        
    def add_object_detection(self, detections):
        """
        Store recent object detections
        
        Args:
            detections: List of detected objects
        """
        timestamp = datetime.now().isoformat()
        self.object_memory.append({
            'timestamp': timestamp,
            'objects': [d['label'] for d in detections],
            'count': len(detections)
        })
    
    def get_recent_objects(self, limit=5):
        """
        Get recent object detections
        
        Args:
            limit: Number of recent detections to return
            
        Returns:
            list: Recent object detections
        """
        return list(self.object_memory)[-limit:]
    
    def check_object_persistence(self, object_name):
        """
        Check if an object was seen before
        
        Args:
            object_name: Name of object to check
            
        Returns:
            bool: True if object was detected previously
        """
        for detection in self.object_memory:
            if object_name.lower() in [obj.lower() for obj in detection['objects']]:
                return True
        return False
    
    
    def get_conversation_history(self):
        """
        Get full conversation history
        
        Returns:
            str: Formatted conversation history
        """
        messages = self.conversation_memory.messages
        history = []
        
        for msg in messages:
            history.append(f"User: {msg.get('input', {}).get('input', 'N/A')}")
            history.append(f"AI: {msg.get('output', {}).get('output', 'N/A')}")
        
        return "\n".join(history)

    
    def get_context_for_query(self, query):
        """
        Build context from memory for current query
        
        Args:
            query: Current user query
            
        Returns:
            str: Context string for LLM
        """
        context_parts = []
        
        # Add recent objects
        if self.object_memory:
            recent = self.get_recent_objects(3)
            objects_str = ", ".join([
                f"{d['count']} objects at {d['timestamp']}"
                for d in recent
            ])
            context_parts.append(f"Recent detections: {objects_str}")
        
        # Add conversation summary
        history = self.get_conversation_history()
        if history:
            context_parts.append(f"Conversation history:\n{history}")
        
        return "\n\n".join(context_parts) if context_parts else "No previous context."
    
    def find_object_in_memory(self, object_name):
        """
        Find when an object was last seen
        
        Args:
            object_name: Name of object to find
            
        Returns:
            dict: Detection info or None
        """
        for detection in reversed(self.object_memory):
            if object_name.lower() in [obj.lower() for obj in detection['objects']]:
                return detection
        return None
    
    def clear_memory(self):
        """Clear all memory"""
        self.conversation_memory.clear()
        self.object_memory.clear()
        self.interaction_count = 0
        self.session_start = datetime.now()
    
    def get_session_summary(self):
        """
        Get summary of current session
        
        Returns:
            dict: Session statistics
        """
        return {
            'session_start': self.session_start.isoformat(),
            'interaction_count': self.interaction_count,
            'unique_objects_seen': len(set(
                obj for det in self.object_memory 
                for obj in det['objects']
            )),
            'recent_detections': len(self.object_memory)
        }
