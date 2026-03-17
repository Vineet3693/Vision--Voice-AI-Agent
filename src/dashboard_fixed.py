"""
FIXED Vision + Voice AI Dashboard with Proper Camera Management
Works reliably with real camera or demo mode
"""
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import time

st.set_page_config(
    page_title="Vision AI - LIVE",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
.live-badge {
    display: inline-block;
    background: #ff0000;
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 12px;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
.object-box {
    background: #e3f2fd;
    border-left: 4px solid #2196F3;
    padding: 10px;
    margin: 5px 0;
    border-radius: 3px;
}
.detection-count {
    font-size: 24px;
    font-weight: bold;
    color: #667eea;
}
.info-card {
    background: #f5f5f5;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'detections_all_time' not in st.session_state:
    st.session_state.detections_all_time = defaultdict(int)
if 'current_detections' not in st.session_state:
    st.session_state.current_detections = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# YOLO SETUP
# ============================================================================
@st.cache_resource
def load_yolo(model_name):
    """Load YOLO model once"""
    try:
        from ultralytics import YOLO
        return YOLO(model_name)
    except Exception as e:
        st.error(f"Failed to load YOLO: {e}")
        return None

def detect_objects(frame, model, conf_threshold):
    """Run YOLO detection"""
    if model is None:
        return []
    
    try:
        results = model(frame, verbose=False, conf=conf_threshold)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    
                    detections.append({
                        'label': label,
                        'confidence': conf,
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2)
                    })
                    
                    st.session_state.detections_all_time[label] += 1
        
        return detections
    except Exception as e:
        return []

def draw_boxes_on_frame(frame, detections):
    """Draw bounding boxes on frame"""
    frame_copy = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        label = det['label']
        conf = det['confidence']
        
        # Color based on confidence
        if conf > 0.75:
            color = (0, 255, 0)  # Green
        elif conf > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange
        
        # Draw box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw text
        text = f"{label}: {conf:.2f}"
        cv2.putText(frame_copy, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame_copy

# ============================================================================
# MAIN LAYOUT
# ============================================================================

# Header
col_header = st.container()
with col_header:
    st.markdown("<h1 style='text-align: center; color: #667eea;'>👁️ VISION + VOICE AI AGENT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Real-time object detection with live chat</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Mode selection
    mode = st.radio(
        "🎬 Mode",
        ["Live Camera", "Demo (Test Images)"],
        help="Choose between real camera or demo with test images"
    )
    
    # Model selection
    st.subheader("🎯 YOLO Model")
    model_choice = st.selectbox(
        "Select model:",
        ["yolov8n.pt", "yolov8m.pt", "yolov8l.pt"],
        index=1,
        help="Nano=Fast, Medium=Balanced, Large=Accurate"
    )
    
    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold",
        0.1, 0.95, 0.4, 0.05,
        help="Lower = more detections, Higher = fewer but more confident"
    )
    
    st.divider()
    
    # Stats
    st.subheader("📊 Statistics")
    duration = str(datetime.now() - st.session_state.start_time).split('.')[0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Frames", st.session_state.frame_count)
    with col2:
        st.metric("Objects", len(st.session_state.detections_all_time))
    
    st.metric("Total Detections", sum(st.session_state.detections_all_time.values()))
    st.metric("Duration", duration)

# ============================================================================
# MAIN CONTENT - 3 COLUMNS
# ============================================================================

col_video, col_objects, col_chat = st.columns([2, 1, 1.2])

# ============================================================================
# LEFT COLUMN - VIDEO FEED
# ============================================================================
with col_video:
    st.subheader("📹 Live Detection Feed")
    
    video_placeholder = st.empty()
    info_placeholder = st.empty()
    error_placeholder = st.empty()
    
    # Load model
    with st.spinner(f"Loading {model_choice}..."):
        model = load_yolo(model_choice)
    
    if model is None:
        error_placeholder.error("❌ Failed to load YOLO model")
    else:
        st.success(f"✅ {model_choice} loaded")
        
        if mode == "Live Camera":
            # Live camera mode
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                error_placeholder.error("❌ Cannot access camera. Try Demo mode below.")
                
                # Show placeholder
                placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_frame, "Camera not available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                video_placeholder.image(cv2.cvtColor(placeholder_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                st.success("✅ Camera connected")
                
                # Read frames
                for i in range(100):  # Limit to prevent infinite loop
                    ret, frame = cap.read()
                    
                    if not ret:
                        error_placeholder.error("⚠️ Failed to read frame from camera")
                        break
                    
                    st.session_state.frame_count += 1
                    
                    # Detect every other frame for performance
                    if i % 2 == 0:
                        detections = detect_objects(frame, model, confidence)
                        st.session_state.current_detections = detections
                    
                    # Draw boxes
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated = draw_boxes_on_frame(frame_rgb, st.session_state.current_detections)
                    
                    # Add frame info
                    cv2.putText(annotated, f"Frame: {st.session_state.frame_count} | Objects: {len(st.session_state.current_detections)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display
                    video_placeholder.image(annotated, use_container_width=True)
                    info_placeholder.write(f"**Detected:** {len(st.session_state.current_detections)} objects in frame | **Session total:** {sum(st.session_state.detections_all_time.values())} detections")
                    
                    time.sleep(0.01)
                
                cap.release()
        
        else:
            # Demo mode with test images
            st.info("🎭 Demo Mode - Using generated test images")
            
            for frame_num in range(100):
                st.session_state.frame_count += 1
                
                # Create a test frame with some patterns
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
                
                # Add some colored rectangles to simulate objects
                if frame_num % 30 < 10:
                    cv2.rectangle(frame, (100, 100), (250, 250), (0, 255, 0), -1)
                if frame_num % 30 < 20:
                    cv2.rectangle(frame, (350, 100), (550, 250), (255, 0, 0), -1)
                if frame_num % 20 < 15:
                    cv2.circle(frame, (320, 350), 80, (0, 0, 255), -1)
                
                # Run YOLO on test frame
                detections = detect_objects(frame, model, confidence)
                st.session_state.current_detections = detections
                
                # Draw boxes
                annotated = draw_boxes_on_frame(frame, detections)
                cv2.putText(annotated, f"Frame: {st.session_state.frame_count} | Objects: {len(detections)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display
                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                info_placeholder.write(f"**Detected:** {len(detections)} objects | **Session total:** {sum(st.session_state.detections_all_time.values())} detections")
                
                time.sleep(0.02)

# ============================================================================
# MIDDLE COLUMN - OBJECTS
# ============================================================================
with col_objects:
    st.subheader("🎯 Detected Objects")
    
    if st.session_state.current_detections:
        st.markdown("<span class='live-badge'>🔴 LIVE</span>", unsafe_allow_html=True)
        for det in st.session_state.current_detections:
            conf_pct = int(det['confidence'] * 100)
            st.markdown(f"<div class='object-box'><b>{det['label']}</b><br>{conf_pct}% confidence</div>", unsafe_allow_html=True)
    else:
        st.info("No objects detected")
    
    st.divider()
    
    st.subheader("📊 All Session Objects")
    if st.session_state.detections_all_time:
        for obj, count in sorted(st.session_state.detections_all_time.items(), key=lambda x: x[1], reverse=True)[:10]:
            st.write(f"**{obj}:** {count}")
    else:
        st.info("Waiting for detections...")

# ============================================================================
# RIGHT COLUMN - CHAT
# ============================================================================
with col_chat:
    st.subheader("💬 AI Chat")
    
    # Display messages
    for msg in st.session_state.messages[-6:]:
        if msg['role'] == 'user':
            st.markdown(f"**👤 You:** {msg['text']}")
        else:
            st.markdown(f"**🤖 AI:** {msg['text']}")
    
    st.divider()
    
    # Input
    user_input = st.text_area("Ask about detected objects:", height=80, key="chat_input")
    
    if st.button("Send 🚀", use_container_width=True):
        if user_input:
            # Add user message
            st.session_state.messages.append({
                'role': 'user',
                'text': user_input
            })
            
            # Generate response
            if st.session_state.current_detections:
                obj_list = ", ".join([det['label'] for det in st.session_state.current_detections])
                response = f"I see: {obj_list}. Total detections this session: {sum(st.session_state.detections_all_time.values())}."
            else:
                response = "No objects in current frame. Try repositioning the camera."
            
            st.session_state.messages.append({
                'role': 'ai',
                'text': response
            })
            
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("**🎯 YOLO Capabilities**")
    st.caption("Detects 80+ object classes")
    st.caption("Person, vehicle, animal, etc.")

with col_f2:
    st.markdown("**⚡ Features**")
    st.caption("✅ Real-time detection")
    st.caption("✅ Live bounding boxes")
    st.caption("✅ AI chat")

with col_f3:
    st.markdown("**🔧 Options**")
    st.caption("→ Switch to Demo mode")
    st.caption("→ Adjust confidence")
    st.caption("→ Change YOLO model")
