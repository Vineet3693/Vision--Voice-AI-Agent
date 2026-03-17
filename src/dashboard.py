"""
Streamlit Dashboard - Visual interface for the AI Vision Assistant
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Vision + Voice AI Agent",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1E88E5;
    text-align: center;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.warning-box {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 1rem;
    margin: 1rem 0;
}
.success-box {
    background-color: #e8f5e9;
    border-left: 5px solid #4caf50;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">👁️ Vision + Voice AI Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Multi-Agent System for Visually Impaired Assistance</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Camera control
        camera_enabled = st.checkbox("Enable Camera", value=True)
        
        # Voice control
        voice_enabled = st.checkbox("Enable Voice Input", value=False)
        
        # TTS control
        tts_enabled = st.checkbox("Enable Text-to-Speech", value=True)
        
        st.divider()
        
        # Settings
        st.subheader("🎛️ Settings")
        language = st.selectbox(
            "Language",
            ["English", "Hindi", "Spanish", "French"],
            index=0
        )
        
        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.1
        )
        
        st.divider()
        
        # Session info
        st.subheader("📊 Session Info")
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
        
        st.metric("Session Duration", 
                  str(datetime.now() - st.session_state.session_start).split('.')[0])
        
        if 'interaction_count' in st.session_state:
            st.metric("Interactions", st.session_state.interaction_count)
        
        if 'objects_detected' in st.session_state:
            st.metric("Objects Detected", st.session_state.objects_detected)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        if camera_enabled:
            # Camera placeholder
            camera_placeholder = st.empty()
            
            # Try to initialize camera
            try:
                cap = cv2.VideoCapture(0)
                
                if cap.isOpened():
                    st.success("✅ Camera connected")
                    
                    # Create placeholder for detections
                    detection_placeholder = st.empty()
                    
                    # Note: In real implementation, this would run in a loop
                    # For demo, we show a static frame
                    ret, frame = cap.read()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, use_column_width=True)
                        
                        # Sample detection overlay (would be dynamic in real app)
                        st.info("🔍 Real-time object detection active")
                    
                    cap.release()
                else:
                    st.error("❌ Could not access camera")
                    camera_placeholder.image(
                        np.zeros((480, 640, 3), dtype=np.uint8),
                        use_column_width=True
                    )
                    
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
        else:
            st.info("📷 Camera is disabled")
            st.image(
                np.zeros((480, 640, 3), dtype=np.uint8),
                caption="Camera disabled"
            )
    
    with col2:
        st.subheader("🎯 Detected Objects")
        
        # Sample detections (would be dynamic in real app)
        if 'current_detections' not in st.session_state:
            st.session_state.current_detections = []
        
        if st.session_state.current_detections:
            for det in st.session_state.current_detections:
                st.markdown(f"- **{det['label']}** ({det['confidence']:.2f})")
        else:
            st.info("No objects detected yet")
        
        st.divider()
        
        st.subheader("⚠️ Danger Alerts")
        
        if 'active_warnings' not in st.session_state or not st.session_state.active_warnings:
            st.success("✅ No dangers detected")
        else:
            for warning in st.session_state.active_warnings:
                st.markdown(f'<div class="warning-box">{warning}</div>', 
                           unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("🧭 Spatial Awareness")
        
        if 'positions' not in st.session_state:
            st.info("Position data will appear here")
        else:
            positions = st.session_state.positions
            if positions.get('left'):
                st.write(f"⬅️ Left: {', '.join(positions['left'])}")
            if positions.get('center'):
                st.write(f"⬆️ Center: {', '.join(positions['center'])}")
            if positions.get('right'):
                st.write(f"➡️ Right: {', '.join(positions['right'])}")
    
    # Conversation section
    st.divider()
    
    st.subheader("💬 Conversation")
    
    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Display conversation
    conversation_container = st.container()
    with conversation_container:
        for msg in st.session_state.conversation_history[-10:]:  # Last 10 messages
            if msg['role'] == 'user':
                st.markdown(f"**👤 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 AI:** {msg['content']}")
    
    # Input area
    input_col1, input_col2 = st.columns([4, 1])
    
    with input_col1:
        user_input = st.text_input(
            "Type your message:",
            key="user_input_field",
            placeholder="Ask about what's in front of you..."
        )
    
    with input_col2:
        if st.button("📤 Send", use_container_width=True):
            if user_input:
                # Add to conversation history
                st.session_state.conversation_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
                })
                
                # In real app, this would call the orchestrator
                # For demo, we show a placeholder response
                st.session_state.conversation_history.append({
                    'role': 'ai',
                    'content': "Processing your request... (Connect API keys for full functionality)",
                    'timestamp': datetime.now()
                })
                
                st.session_state.interaction_count = st.session_state.get('interaction_count', 0) + 1
                
                st.rerun()
    
    # Footer
    st.divider()
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**🏗 Architecture**")
        st.markdown("- Groq + Llama3 (Fast)")
        st.markdown("- Gemini Vision (Smart)")
        st.markdown("- YOLOv8 (Detection)")
    
    with footer_col2:
        st.markdown("**🤖 Agents**")
        st.markdown("- Vision Agent")
        st.markdown("- Memory Agent")
        st.markdown("- Chat Agent")
        st.markdown("- Web Agent")
    
    with footer_col3:
        st.markdown("**🎯 Purpose**")
        st.markdown("AI assistance for")
        st.markdown("visually impaired")
        st.markdown("individuals")
    
    # Demo mode notice
    if not os.path.exists('.env'):
        st.warning("⚠️ Demo Mode: Copy .env.example to .env and add your API keys for full functionality")


if __name__ == "__main__":
    main()
