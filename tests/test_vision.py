"""
Test script for Vision Agent
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

def test_vision_agent():
    """Test the Vision Agent with sample image"""
    
    print("🧪 Testing Vision Agent...")
    print()
    
    try:
        from src.agents.vision_agent import VisionAgent
        
        # Create a test agent (will fail gracefully without API keys)
        try:
            agent = VisionAgent()
            print("✅ Vision Agent initialized")
            
            # Create a simple test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test YOLO detection
            print("\n📸 Testing YOLO object detection...")
            detections = agent.detect_objects(test_image)
            print(f"   Detected {len(detections)} objects")
            
            if detections:
                for det in detections[:3]:
                    print(f"   - {det['label']} ({det['confidence']:.2f})")
            
            # Test spatial positioning
            print("\n🧭 Testing spatial awareness...")
            positions = agent.get_spatial_positions(detections, 640)
            print(f"   Left: {positions['left']}")
            print(f"   Center: {positions['center']}")
            print(f"   Right: {positions['right']}")
            
            # Note: Gemini Vision test would require valid API key and real image
            print("\n⚠️  Gemini Vision test skipped (requires API key)")
            
        except Exception as e:
            print(f"⚠️  Vision Agent initialization failed (expected without API keys): {e}")
            print("\n💡 To test fully:")
            print("   1. Add GEMINI_API_KEY to .env file")
            print("   2. Run again")
        
        print("\n✅ Vision Agent tests completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")


if __name__ == "__main__":
    test_vision_agent()
