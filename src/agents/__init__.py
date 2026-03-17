"""
Agent initialization module
"""
from src.config.settings import Config

def initialize_agents():
    """Initialize all AI agents"""
    
    # Validate configuration first
    errors = Config.validate()
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    # Import agents here to avoid circular imports
    from src.agents.vision_agent import VisionAgent
    from src.agents.memory_agent import MemoryAgent
    from src.agents.chat_agent import ChatAgent
    from src.agents.web_agent import WebAgent
    from src.agents.orchestrator import MasterOrchestrator
    
    # Initialize agents
    vision_agent = VisionAgent()
    memory_agent = MemoryAgent()
    chat_agent = ChatAgent()
    web_agent = WebAgent()
    
    # Initialize master orchestrator with all agents
    orchestrator = MasterOrchestrator(
        vision_agent=vision_agent,
        memory_agent=memory_agent,
        chat_agent=chat_agent,
        web_agent=web_agent
    )
    
    return orchestrator
