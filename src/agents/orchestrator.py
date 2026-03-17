"""
Master Orchestrator - Routes intents to appropriate agents
Uses Groq + Llama3 for fast intent detection and routing
"""
from groq import Groq
import json
from src.config.settings import Config

class MasterOrchestrator:
    """Master controller that routes requests to appropriate agents"""
    
    def __init__(self, vision_agent, memory_agent, chat_agent, web_agent):
        """
        Initialize orchestrator with all agents
        
        Args:
            vision_agent: VisionAgent instance
            memory_agent: MemoryAgent instance
            chat_agent: ChatAgent instance
            web_agent: WebAgent instance
        """
        self.vision_agent = vision_agent
        self.memory_agent = memory_agent
        self.chat_agent = chat_agent
        self.web_agent = web_agent
        
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = "llama3-8b-8192"
        
    def detect_intent(self, user_input, has_visual_context=False):
        """
        Detect user intent using Groq
        
        Args:
            user_input: User's text input
            has_visual_context: Whether visual context is available
            
        Returns:
            str: Intent category
        """
        system_prompt = """You are an intent classifier. Classify the user's query into one of these categories:
        - vision_query: Questions about what's in front of them or visual scene
        - object_location: Asking where a specific object is
        - general_chat: General conversation or knowledge questions
        - web_search: Questions about current events, news, or facts needing internet search
        - danger_check: Asking about safety or potential dangers
        - memory_query: Questions about previous conversations or detected objects
        
        Respond with ONLY the category name, nothing else."""
        
        context_hint = " (camera is active)" if has_visual_context else " (no camera)"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_input}{context_hint}"}
                ],
                temperature=0.3,
                max_tokens=20
            )
            
            intent = response.choices[0].message.content.strip().lower()
            return intent
            
        except Exception as e:
            print(f"Intent detection error: {e}")
            return "general_chat"  # Fallback
    
    def route_request(self, user_input, frame=None):
        """
        Route user request to appropriate agent
        
        Args:
            user_input: User's text input
            frame: Optional camera frame for vision queries
            
        Returns:
            dict: Response with answer and metadata
        """
        # Detect intent
        has_visual = frame is not None
        intent = self.detect_intent(user_input, has_visual)
        
        # Get context from memory
        memory_context = self.memory_agent.get_context_for_query(user_input)
        
        response_data = {
            'intent': intent,
            'response': '',
            'visual_data': None,
            'warnings': []
        }
        
        # Route based on intent
        if intent == 'vision_query' and has_visual:
            response_data = self._handle_vision_query(user_input, frame)
            
        elif intent == 'object_location' and has_visual:
            response_data = self._handle_object_location(user_input, frame)
            
        elif intent == 'danger_check' and has_visual:
            response_data = self._handle_danger_check(user_input, frame)
            
        elif intent == 'memory_query':
            response_data = self._handle_memory_query(user_input)
            
        elif intent == 'web_search':
            response_data = self._handle_web_search(user_input)
            
        else:  # general_chat or fallback
            response_data = self._handle_general_chat(user_input, memory_context)
        
        # Store conversation in memory
        if response_data['response']:
            self.memory_agent.add_conversation(user_input, response_data['response'])
        
        return response_data
    
    def _handle_vision_query(self, user_input, frame):
        """Handle queries about the visual scene"""
        # Process frame through vision agent
        vision_data = self.vision_agent.process_frame(frame)
        
        # Store detections in memory
        if vision_data['detections']:
            self.memory_agent.add_object_detection(vision_data['detections'])
        
        # Generate natural description
        description = self.chat_agent.generate_description(vision_data)
        
        # Check for dangers
        warnings = []
        if vision_data['dangers']:
            for danger in vision_data['dangers']:
                if 'fire' in danger.lower() or 'smoke' in danger.lower():
                    warnings.append(self.chat_agent.create_warning('fire'))
                else:
                    warnings.append(self.chat_agent.create_warning('default', danger))
        
        return {
            'intent': 'vision_query',
            'response': description,
            'visual_data': vision_data,
            'warnings': warnings
        }
    
    def _handle_object_location(self, user_input, frame):
        """Handle queries about object positions"""
        vision_data = self.vision_agent.process_frame(frame)
        
        if vision_data['detections']:
            self.memory_agent.add_object_detection(vision_data['detections'])
        
        # Extract object name from query (simple approach)
        positions = vision_data['positions']
        
        # Build location response
        location_info = []
        for position, objects in positions.items():
            if objects:
                location_info.append(f"{position}: {', '.join(objects)}")
        
        if location_info:
            response = "I can see: " + ". ".join(location_info) + "."
        else:
            response = "I don't see any recognizable objects right now."
        
        return {
            'intent': 'object_location',
            'response': response,
            'visual_data': vision_data,
            'warnings': []
        }
    
    def _handle_danger_check(self, user_input, frame):
        """Handle safety/danger queries"""
        vision_data = self.vision_agent.process_frame(frame)
        
        if vision_data['detections']:
            self.memory_agent.add_object_detection(vision_data['detections'])
        
        warnings = []
        if vision_data['dangers']:
            for danger in vision_data['dangers']:
                warnings.append(self.chat_agent.create_warning('default', danger))
            response = "⚠️ " + " ".join(warnings)
        else:
            response = "No immediate dangers detected. Your surroundings appear safe."
        
        return {
            'intent': 'danger_check',
            'response': response,
            'visual_data': vision_data,
            'warnings': warnings
        }
    
    def _handle_memory_query(self, user_input):
        """Handle queries about past conversations or objects"""
        # Use chat agent with memory context
        memory_context = self.memory_agent.get_context_for_query(user_input)
        response = self.chat_agent.chat(user_input, memory_context)
        
        return {
            'intent': 'memory_query',
            'response': response,
            'visual_data': None,
            'warnings': []
        }
    
    def _handle_web_search(self, user_input):
        """Handle queries requiring internet search"""
        answer = self.web_agent.get_answer(user_input)
        
        return {
            'intent': 'web_search',
            'response': answer,
            'visual_data': None,
            'warnings': []
        }
    
    def _handle_general_chat(self, user_input, memory_context):
        """Handle general conversation"""
        response = self.chat_agent.chat(user_input, memory_context)
        
        return {
            'intent': 'general_chat',
            'response': response,
            'visual_data': None,
            'warnings': []
        }
    
    def process_with_frame(self, user_input, frame):
        """
        Convenience method for processing with visual context
        
        Args:
            user_input: User's text
            frame: Camera frame
            
        Returns:
            dict: Complete response
        """
        return self.route_request(user_input, frame)
    
    def process_without_frame(self, user_input):
        """
        Convenience method for processing without visual context
        
        Args:
            user_input: User's text
            
        Returns:
            dict: Complete response
        """
        return self.route_request(user_input, frame=None)
