"""
Web Search Agent - Internet queries using DuckDuckGo
For questions that require current information
"""
from duckduckgo_search import DDGS
from src.config.settings import Config

class WebAgent:
    """Handles web search queries"""
    
    def __init__(self):
        """Initialize Web Search Agent"""
        self.ddgs = DDGS()
        
    def search(self, query, max_results=3):
        """
        Search the web for information
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            list: Search results
        """
        try:
            results = []
            ddg_results = self.ddgs.text(query, max_results=max_results)
            
            for result in ddg_results:
                results.append({
                    'title': result.get('title', 'No title'),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
            
            return results
            
        except Exception as e:
            return [{'error': f"Search failed: {str(e)}"}]
    
    def get_answer(self, question):
        """
        Get a concise answer from web search
        
        Args:
            question: User's question
            
        Returns:
            str: Formatted answer
        """
        results = self.search(question, max_results=2)
        
        if not results or 'error' in results[0]:
            return "I couldn't find information about that on the internet."
        
        # Format results into a natural response
        answer_parts = []
        
        for i, result in enumerate(results, 1):
            if 'title' in result and 'snippet' in result:
                answer_parts.append(f"{result['snippet']}")
        
        if answer_parts:
            return "Here's what I found: " + " ".join(answer_parts[:2])
        else:
            return "I found some information but couldn't extract a clear answer."
    
    def get_latest_news(self, topic="AI"):
        """
        Get latest news about a topic
        
        Args:
            topic: News topic
            
        Returns:
            str: Latest news summary
        """
        query = f"latest news about {topic}"
        results = self.search(query, max_results=3)
        
        if not results:
            return f"I couldn't find recent news about {topic}."
        
        news_items = []
        for result in results:
            if 'title' in result:
                news_items.append(result['title'])
        
        return "Recent headlines: " + ", ".join(news_items)
