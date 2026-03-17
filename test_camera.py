"""
Camera Diagnostic Script - Check what's wrong with camera access
"""
import cv2
import sys

print("╔" + "="*70 + "╗")
print("║" + "  CAMERA DIAGNOSTIC TOOL  ".center(70) + "║")
print("╚" + "="*70 + "╝")
print()

# Test 1: Check OpenCV
print("1️⃣  OpenCV Version:")
print(f"   {cv2.__version__}")
print()

# Test 2: Try to access camera
print("2️⃣  Testing Camera Access:")
print()

for camera_idx in range(5):
    print(f"   Trying camera index {camera_idx}...", end=" ")
    try:
        cap = cv2.VideoCapture(camera_idx)
        
        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                print(f"✅ SUCCESS")
                print(f"      Resolution: {width}x{height}")
                print(f"      FPS: {fps}")
                
                cap.release()
                break
            else:
                print("❌ Cannot read frame")
                cap.release()
        else:
            print("❌ Cannot open")
    
    except Exception as e:
        print(f"❌ Error: {str(e)[:40]}")

print()
print("3️⃣  Settings:")
print("   If camera test failed:")
print("   - Check if another app is using the camera")
print("   - Try restarting your computer")
print("   - Check webcam is not disabled in device settings")
print("   - Run dashboard with test mode enabled")
print()
