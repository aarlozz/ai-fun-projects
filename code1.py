import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import random

# ── Model download ────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Done.")

# ── Hand Landmarker setup ─────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# Fingertip landmarks
FINGERTIPS = {
    "index":  8,
    "middle": 12,
    "ring":   16,
    "pinky":  20,
}
# Corresponding PIP (knuckle) landmarks for extended-finger check
KNUCKLES = {
    "index":  6,
    "middle": 10,
    "ring":   14,
    "pinky":  18,
}

# ── Helper: make a solid circle sprite (BGRA) ────────────────────────────────
def make_circle_sprite(size, bgr_color, outline_bgr=None):
    img = np.zeros((size, size, 4), dtype=np.uint8)
    r = size // 2
    cv2.circle(img, (r, r), r - 2, (*bgr_color, 255), -1)
    if outline_bgr:
        cv2.circle(img, (r, r), r - 2, (*outline_bgr, 255), 3)
    return img

# ── Load or create flower sprites ────────────────────────────────────────────
def load_or_fallback(path, size, fallback_bgr, outline_bgr=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️  '{path}' not found — using coloured circle fallback.")
        return make_circle_sprite(size, fallback_bgr, outline_bgr)
    img = cv2.resize(img, (size, size))
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

SPRITE_SIZE = 80
CAPY_SIZE   = 90   # capybaras are a bit bigger for maximum chaos

sprites = {
    "index":  load_or_fallback("sunflower.jpg",   SPRITE_SIZE, (0, 215, 255), (0, 140, 255)),   # yellow
    "ring":   load_or_fallback("cherry.jpg",      SPRITE_SIZE, (147, 112, 219), (180, 60, 200)), # pink/purple
    "pinky":  load_or_fallback("hibiscus.jpg",    SPRITE_SIZE, (0, 0, 220),   (0, 0, 160)),      # red
    "capy":   load_or_fallback("capybara.jpg",    CAPY_SIZE,   (34, 85, 102), (20, 60, 80)),     # brown
}

# ── Overlay helper (pixel-by-pixel alpha blend) ───────────────────────────────
def overlay_image(bg, overlay, x, y):
    oh, ow = overlay.shape[:2]
    bh, bw = bg.shape[:2]
    # Clip to frame bounds
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, bw), min(y + oh, bh)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return bg
    roi     = bg[y1:y2, x1:x2]
    crop    = overlay[oy1:oy2, ox1:ox2]
    alpha   = crop[:, :, 3:4] / 255.0
    bg[y1:y2, x1:x2] = (crop[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
    return bg

# ── Finger-extended check ─────────────────────────────────────────────────────
def is_extended(hand, tip_idx, knuckle_idx):
    """True when fingertip is above its knuckle (in normalised y, smaller = higher)."""
    return hand[tip_idx].y < hand[knuckle_idx].y

# ── Capybara rain state ───────────────────────────────────────────────────────
class Capybara:
    def __init__(self, frame_w, frame_h):
        self.x     = random.randint(0, frame_w - CAPY_SIZE)
        self.y     = random.randint(-frame_h, -CAPY_SIZE)   # start above screen
        self.speed = random.uniform(2.5, 6.0)
        self.frame_h = frame_h

    def update(self):
        self.y += self.speed

    def is_off_screen(self):
        return self.y > self.frame_h

capybaras_on_screen = []   # active falling capys
settled_capys       = []   # ones that have landed (permanent overlay)
SPAWN_RATE          = 4    # spawn one new capy every N frames while middle is up
frame_count         = 0

# ── Webcam loop ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Press ESC to quit.  Press C to clear capybaras.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_count += 1

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    # ── Draw settled capybaras first (background layer) ──────────────────────
    for sx, sy in settled_capys:
        frame = overlay_image(frame, sprites["capy"], sx, sy)

    # ── Detect which fingers are up ───────────────────────────────────────────
    middle_up = False
    active_fingers = []   # list of (finger_name, tip_x, tip_y)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for name in ("index", "middle", "ring", "pinky"):
                if is_extended(hand, FINGERTIPS[name], KNUCKLES[name]):
                    lm  = hand[FINGERTIPS[name]]
                    tx  = int(lm.x * w)
                    ty  = int(lm.y * h)
                    active_fingers.append((name, tx, ty))
                    if name == "middle":
                        middle_up = True

            # Draw all landmarks
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # ── Spawn capybaras when middle finger is raised ──────────────────────────
    if middle_up and frame_count % SPAWN_RATE == 0:
        capybaras_on_screen.append(Capybara(w, h))

    # ── Update & draw falling capybaras ──────────────────────────────────────
    still_falling = []
    for capy in capybaras_on_screen:
        capy.update()
        if capy.is_off_screen():
            settled_capys.append((capy.x, h - CAPY_SIZE))   # settle at bottom
        else:
            frame = overlay_image(frame, sprites["capy"], capy.x, int(capy.y))
            still_falling.append(capy)
    capybaras_on_screen = still_falling

    # ── Draw flower sprites for non-middle fingers ────────────────────────────
    for name, tx, ty in active_fingers:
        if name != "middle":
            sp = sprites[name]
            sz = sp.shape[0]
            frame = overlay_image(frame, sp, tx - sz // 2, ty - sz // 2)
        else:
            # Fun label above middle finger
            cv2.putText(frame, "CAPY MODE 🦫", (tx - 60, ty - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Capybaras: {len(settled_capys) + len(capybaras_on_screen)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "C=clear  ESC=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Finger Flowers & Capybara Rain 🌻🦫", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:          # ESC → quit
        break
    elif key == ord('c'):  # C   → clear all capybaras
        capybaras_on_screen.clear()
        settled_capys.clear()

cap.release()
cv2.destroyAllWindows()