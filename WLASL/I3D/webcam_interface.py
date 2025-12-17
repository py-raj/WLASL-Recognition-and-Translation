# webcam_inference.py 13.12.25

import cv2
import torch
import numpy as np
from collections import deque

from inference_i3d import (
    load_i3d_model,
    load_label_map,
    predict_sign,
    get_video_transforms
)

# -----------------------------
# CONFIG
# -----------------------------
NUM_FRAMES = 16          # Temporal window (must match training)
FRAME_SIZE = 224
LABEL_MAP_PATH = "preprocess/nslt_2000.json"


# -----------------------------
# FRAME BUFFER
# -----------------------------
class FrameBuffer:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def add(self, frame):
        self.buffer.append(frame)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def get(self):
        return list(self.buffer)


# -----------------------------
# PREPROCESS FRAMES
# -----------------------------
def preprocess_frames(frames, transform):
    processed = []

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frame = frame / 255.0

        frame = torch.tensor(frame).permute(2, 0, 1)  # C,H,W
        processed.append(frame)

    video = torch.stack(processed, dim=1)  # C,T,H,W
    video = transform(video)
    video = video.unsqueeze(0)  # 1,C,T,H,W

    return video

#-----------------------------
# Prediction Smoother
#-----------------------------

from collections import Counter, deque

class PredictionSmoother:
    def __init__(self, window_size=10, confidence_threshold=0.3):
        self.window_size = window_size
        self.conf_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)

    def add(self, word, confidence):
        if confidence >= self.conf_threshold:
            self.buffer.append(word)

    def get_stable_prediction(self):
        if len(self.buffer) == 0:
            return None

        counter = Counter(self.buffer)
        stable_word, count = counter.most_common(1)[0]

        # Optional: ensure stability
        if count >= len(self.buffer) // 2:
            return stable_word

        return None

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    print("Loading model...")
    model = load_i3d_model()

    print("Loading label map...")
    label_map = load_label_map(LABEL_MAP_PATH)

    transform = get_video_transforms()
    buffer = FrameBuffer(NUM_FRAMES)

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "❌ Webcam not accessible"

    last_prediction = "..."
    last_confidence = 0.0

    print("Starting webcam inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        buffer.add(frame)

        if buffer.is_full():
            video_tensor = preprocess_frames(buffer.get(), transform)
            word, conf = predict_sign(model, video_tensor, label_map)

            last_prediction = word
            last_confidence = conf

        # Display output
        cv2.putText(
            frame,
            f"Prediction: {last_prediction} ({last_confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
