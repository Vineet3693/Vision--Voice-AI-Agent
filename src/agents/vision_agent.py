"""
Vision Agent - Dual-layer vision processing
Layer 1: YOLOv8 for fast object detection
Layer 2: Gemini Vision for deep scene understanding
"""
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
from src.config.settings import Config

class VisionAgent:
    """Handles all vision-related tasks"""
    
    def __init__(self):
        """Initialize Vision Agent with YOLO and Gemini"""
        # Configure Gemini
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
        
        # Load YOLO model
        self.yolo_model = YOLO(Config.YOLO_MODEL)
        
        # Class names for YOLO
        self.class_names = self.yolo_model.names
        
    def detect_objects(self, frame):
        """
        Layer 1: Fast object detection using YOLOv8
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            dict: Detected objects with bounding boxes and confidence
        """
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    detections.append({
                        'label': self.class_names[cls_id],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2
                    })
        
        return detections
    
    def analyze_scene(self, frame, detections=None):
        """
        Layer 2: Deep scene understanding using Gemini Vision
        
        Args:
            frame: OpenCV image frame
            detections: Optional YOLO detections for context
            
        Returns:
            str: Detailed scene description
        """
        # Convert BGR to RGB for Gemini
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Build prompt
        prompt = self._build_scene_prompt(detections)
        
        try:
            response = self.gemini_model.generate_content([prompt, rgb_frame])
            return response.text
        except Exception as e:
            return f"Scene analysis error: {str(e)}"
    
    def _build_scene_prompt(self, detections):
        """Build intelligent prompt based on detected objects"""
        base_prompt = """Analyze this image in detail. Describe:
        1. What objects are present and their spatial relationships
        2. The overall scene context
        3. Any potential dangers or important elements
        4. If people are present, describe their apparent emotional state
        
        Be concise but informative. This is for a visually impaired assistant."""
        
        if detections:
            objects_list = ", ".join([d['label'] for d in detections[:5]])
            base_prompt += f"\n\nI've detected these objects: {objects_list}. Provide context about them."
        
        return base_prompt
    
    def detect_dangers(self, frame, scene_description):
        """
        Check for dangerous situations
        
        Args:
            frame: Image frame
            scene_description: Gemini's scene analysis
            
        Returns:
            list: List of detected dangers
        """
        dangers = []
        scene_lower = scene_description.lower()
        
        # Check for danger keywords
        for keyword in Config.DANGER_KEYWORDS:
            if keyword in scene_lower:
                dangers.append(keyword)
        
        # Check for fire/smoke specifically
        if 'fire' in scene_lower or 'smoke' in scene_lower:
            dangers.insert(0, "⚠️ CRITICAL: Fire or smoke detected!")
        
        return dangers
    
    def get_spatial_positions(self, detections, frame_width):
        """
        Determine spatial positions of objects (left, center, right)
        
        Args:
            detections: YOLO detections
            frame_width: Width of the frame
            
        Returns:
            dict: Objects organized by position
        """
        positions = {'left': [], 'center': [], 'right': []}
        
        third = frame_width / 3
        
        for det in detections:
            cx = det['center_x']
            if cx < third:
                positions['left'].append(det['label'])
            elif cx > 2 * third:
                positions['right'].append(det['label'])
            else:
                positions['center'].append(det['label'])
        
        return positions
    
    def process_frame(self, frame):
        """
        Complete frame processing pipeline
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            dict: Complete vision analysis
        """
        # Layer 1: Fast detection
        detections = self.detect_objects(frame)
        
        # Get spatial positions
        h, w = frame.shape[:2]
        positions = self.get_spatial_positions(detections, w)
        
        # Layer 2: Deep analysis (only if objects detected)
        scene_description = ""
        dangers = []
        
        if detections:
            scene_description = self.analyze_scene(frame, detections)
            dangers = self.detect_dangers(frame, scene_description)
        
        return {
            'detections': detections,
            'positions': positions,
            'scene_description': scene_description,
            'dangers': dangers,
            'object_count': len(detections)
        }
