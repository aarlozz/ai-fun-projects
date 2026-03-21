import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

# Download the hand landmarker model if not present
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Done.")

# Setup Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Load sunflower image — with fallback if file not found
sunflower = cv2.imread("sunflower.jpg", cv2.IMREAD_UNCHANGED)

if sunflower is None:
    print("⚠️  sunflower.png not found — using a yellow circle as fallback.")
    # Create a simple yellow circle on transparent background (BGRA)
    sunflower = np.zeros((80, 80, 4), dtype=np.uint8)
    cv2.circle(sunflower, (40, 40), 38, (0, 215, 255, 255), -1)   # yellow fill
    cv2.circle(sunflower, (40, 40), 38, (0, 140, 255, 255), 3)    # orange outline
else:
    sunflower = cv2.resize(sunflower, (80, 80))
    # If image has no alpha channel, add one
    if sunflower.shape[2] == 3:
        sunflower = cv2.cvtColor(sunflower, cv2.COLOR_BGR2BGRA)

def overlay_image(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    for i in range(h):
        for j in range(w):
            if overlay[i][j][3] != 0:  # alpha channel
                if 0 <= y+i < bg.shape[0] and 0 <= x+j < bg.shape[1]:
                    bg[y+i][x+j] = overlay[i][j][:3]
    return bg

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB and wrap in MediaPipe Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    finger_x, finger_y = None, None

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            lm = hand[8]  # index finger tip
            finger_x = int(lm.x * w)
            finger_y = int(lm.y * h)

            for landmark in hand:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    if finger_x is not None and finger_y is not None:
        frame = overlay_image(frame, sunflower, finger_x - 40, finger_y - 40)

    cv2.imshow("Sunflower Pointer 🌻", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
