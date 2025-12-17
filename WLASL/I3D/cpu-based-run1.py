# जय श्री राम
# note it does not use NGram model.
"""
CPU-only LIVE translation script for WLASL I3D model.

- Uses your PC webcam (cv2.VideoCapture(0))
- Runs I3D on CPU only (no CUDA, no DataParallel)
- Uses KeyToText (k2t-new) to turn gloss tokens into a sentence
- DOES NOT use NGram (no nlp_data_processed / nlp_gram_counts)

Press 'q' in the video window to quit.
"""

import os
import argparse
from itertools import chain  # kept in case you extend later

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_i3d import InceptionI3d
from keytotext import pipeline
from dotenv import load_dotenv

# ---------------------------------------------------------
# ENV & DEVICE (CPU ONLY)
# ---------------------------------------------------------

load_dotenv("posts/nlp/.env", override=True)

# Force CPU
DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

parser = argparse.ArgumentParser()
parser.add_argument("-mode", type=str, default="rgb", help="rgb or flow (unused)")
parser.add_argument("-save_model", type=str)
parser.add_argument("-root", type=str)
args = parser.parse_args()

# ---------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------

i3d = None
wlasl_dict = None
nlp = None

# threshold for accepting a gloss (lower than 0.5 so it actually fires)
CONF_THRESHOLD = 0.2

# ---------------------------------------------------------
# LABEL DICTIONARY
# ---------------------------------------------------------

def create_WLASL_dictionary():
    global wlasl_dict
    wlasl_dict = {}

    with open("preprocess/wlasl_class_list.txt") as file:
        for line in file:
            split_list = line.split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value

# ---------------------------------------------------------
# MODEL + NLP LOADING (CPU)
# ---------------------------------------------------------

def load_model(weights, num_classes):
    """
    Load I3D and KeyToText on CPU only.
    """
    global i3d, nlp

    # I3D
    print("🔁 Loading I3D on CPU...")
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(num_classes)

    state_dict = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(state_dict)
    del state_dict

    model.to(DEVICE)
    model.eval()
    i3d = model
    print("✅ I3D loaded (CPU).")

    # KeyToText
    print("⏬ Loading KeyToText model (k2t-new)...")
    nlp_local = pipeline("k2t-new")  # no extra kwargs to avoid TypeError
    nlp = nlp_local
    print("✅ KeyToText loaded.")

    # After model + NLP are ready, start webcam loop
    load_rgb_frames_from_video()

# ---------------------------------------------------------
# RUN I3D ON A CLIP
# ---------------------------------------------------------

def run_on_tensor(ip_tensor):
    """
    ip_tensor: torch tensor of shape (C, T, H, W) on CPU
    Returns: (gloss, confidence)
    """

    # Add batch dimension -> (1, C, T, H, W)
    ip_tensor = ip_tensor.unsqueeze(0).to(DEVICE)
    t = ip_tensor.shape[2]

    with torch.no_grad():
        per_frame_logits = i3d(ip_tensor)          # (1, num_classes, T')
        predictions = F.interpolate(per_frame_logits, t, mode="linear")
        predictions = predictions.transpose(2, 1)  # (1, T, num_classes)

    arr = predictions[0].detach().cpu().numpy()    # (T, num_classes)
    out_labels = np.argsort(arr)
    probs = F.softmax(torch.from_numpy(arr[0]), dim=0)
    max_prob = float(torch.max(probs))

    pred_idx = out_labels[0][-1]
    gloss = wlasl_dict.get(pred_idx, "")

    print("Frame max prob:", round(max_prob, 4), "| gloss:", gloss)
    return gloss, max_prob

# ---------------------------------------------------------
# LIVE WEBCAM LOOP (CPU)
# ---------------------------------------------------------

def load_rgb_frames_from_video():
    """
    Capture from webcam (device 0), maintain a sliding window of frames,
    run I3D every few frames, and show live sentence on screen.

    Press 'q' to quit.
    """

    vidcap = cv2.VideoCapture(0)
    if not vidcap.isOpened():
        print("❌ Could not open webcam.")
        return

    frames = []
    offset = 0
    batch = 40   # number of frames per window
    stride = 20  # run inference every N frames

    gloss_tokens = []
    sentence = ""

    font = cv2.FONT_HERSHEY_TRIPLEX

    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame1 = vidcap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        offset += 1

        # Resize frame for model
        h, w, c = frame1.shape
        sc = 224 / h
        sx = 224 / w
        frame = cv2.resize(frame1, dsize=(0, 0), fx=sx, fy=sc)

        # Also resize display frame to something nice
        frame_display = cv2.resize(frame1, (1280, 720))

        # Normalize to [-1, 1]
        frame_norm = (frame / 255.0) * 2.0 - 1.0
        frame_norm = frame_norm.astype(np.float32)

        # Maintain sliding window of 'batch' frames
        if len(frames) < batch:
            frames.append(frame_norm)
        else:
            frames.pop(0)
            frames.append(frame_norm)

        # Only run model when we have a full window and at stride steps
        if len(frames) == batch and (offset % stride == 0):
            window_np = np.asarray(frames, dtype=np.float32)  # (T, H, W, C)
            ip_tensor = torch.from_numpy(window_np.transpose(3, 0, 1, 2))  # (C, T, H, W)

            gloss, conf = run_on_tensor(ip_tensor)

            if conf >= CONF_THRESHOLD and gloss:
                # Avoid repeating the same gloss back-to-back
                if not gloss_tokens or gloss_tokens[-1] != gloss:
                    gloss_tokens.append(gloss)
                    print("Current gloss tokens:", gloss_tokens)

                    # Simple sentence:
                    #  - if we have at least 2–3 tokens, ask KeyToText
                    #  - otherwise just join tokens
                    if len(gloss_tokens) >= 3:
                        try:
                            sentence = nlp(gloss_tokens)
                        except Exception as e:
                            print("KeyToText error, falling back to raw tokens:", e)
                            sentence = " ".join(gloss_tokens)
                    else:
                        sentence = " ".join(gloss_tokens)

                # Keep the gloss history from growing too big
                if len(gloss_tokens) > 15:
                    gloss_tokens = gloss_tokens[-15:]

        # Draw sentence on frame
        if sentence:
            cv2.putText(
                frame_display,
                sentence,
                (50, 600),
                font,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_4,
            )

        cv2.imshow("Sign Language Translation (CPU)", frame_display)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vidcap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed.")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    # 2000-class WLASL model (same as original)
    num_classes = 2000
    weights = "archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt"

    print("📚 Loading label dictionary...")
    create_WLASL_dictionary()

    print("🧠 Loading I3D + KeyToText on CPU (no NGram)...")
    load_model(weights, num_classes)
    print("✅ Ready.")
