"""
Chat Agent - General conversation using Groq + Llama3
Fast responses for non-vision queries
"""
from groq import Groq
from src.config.settings import Config

class ChatAgent:
    """Handles general conversation and Q&A"""
    
    def __init__(self):
        """Initialize Chat Agent with Groq"""
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        # Use mixtral-8x7b-32768 instead of deprecated llama3-8b-8192
        self.model = "mixtral-8x7b-32768"
        
    def chat(self, message, context=""):
        """
        Generate a conversational response
        
        Args:
            message: User's message
            context: Optional context from memory
            
        Returns:
            str: AI response
        """
        system_prompt = """You are a helpful AI vision assistant for visually impaired people.
        You are knowledgeable, empathetic, and provide clear, concise information.
        Always prioritize safety and be proactive about potential dangers.
        Keep responses brief but informative (2-3 sentences max)."""
        
        if context:
            system_prompt += f"\n\nContext from memory:\n{context}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"I apologize, I'm having trouble responding right now. Error: {str(e)}"
    
    def answer_question(self, question):
        """
        Answer a specific question
        
        Args:
            question: User's question
            
        Returns:
            str: Answer
        """
        return self.chat(question)
    
    def generate_description(self, scene_info):
        """
        Generate a natural description of the scene
        
        Args:
            scene_info: Dictionary with detections, positions, scene_description
            
        Returns:
            str: Natural language description
        """
        prompt = f"""Based on this vision analysis, create a clear, natural description 
        for a visually impaired person:

        Objects detected: {[d['label'] for d in scene_info.get('detections', [])]}
        Positions: {scene_info.get('positions', {})}
        Scene analysis: {scene_info.get('scene_description', 'No detailed analysis')}
        
        Make it conversational and helpful. Mention spatial relationships clearly."""
        
        return self.chat(prompt)
    
    def create_warning(self, danger_type, details=""):
        """
        Create an appropriate warning message
        
        Args:
            danger_type: Type of danger detected
            details: Additional details
            
        Returns:
            str: Warning message
        """
        warnings = {
            'fire': "⚠️ WARNING: I detect fire or smoke in your vicinity. Please move to safety immediately!",
            'obstacle': "⚠️ CAUTION: There's an obstacle in your path.",
            'person_close': "⚠️ NOTICE: Someone is standing very close to you.",
            'default': "⚠️ ATTENTION: Potential hazard detected."
        }
        
        base_warning = warnings.get(danger_type, warnings['default'])
        
        if details:
            return f"{base_warning} {details}"
        
        return base_warning
