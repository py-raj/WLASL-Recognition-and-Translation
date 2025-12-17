# जय श्री राम
# more efficient than cpu-based-interface.py & requires less RAM.
# CPU-only offline inference script for WLASL I3D model (NO NLP, LOW RAM)

import os
from itertools import chain

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_i3d import InceptionI3d
from dotenv import load_dotenv

# ---------------------------------------------------------
# ENV & GLOBALS
# ---------------------------------------------------------

load_dotenv("posts/nlp/.env", override=True)

DEVICE = torch.device("cpu")   # CPU ONLY

i3d = None
wlasl_dict = None

# ---------------------------------------------------------
# LOAD LABEL DICTIONARY
# ---------------------------------------------------------

def create_WLASL_dictionary():
    """
    Build a mapping from class index to gloss (word/phrase).
    """
    global wlasl_dict
    wlasl_dict = {}

    with open('preprocess/wlasl_class_list.txt') as file:
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
# LOAD I3D MODEL (CPU)
# ---------------------------------------------------------

def load_model(weights, num_classes):
    """
    Load I3D model on CPU only.
    """
    global i3d

    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)

    state_dict = torch.load(weights, map_location=DEVICE)
    i3d.load_state_dict(state_dict)

    i3d.to(DEVICE)
    i3d.eval()

# ---------------------------------------------------------
# RUN I3D ON A WINDOW OF FRAMES
# ---------------------------------------------------------

def run_on_tensor(ip_tensor, threshold=0.5):
    """
    ip_tensor: torch tensor of shape (C, T, H, W)
    Returns: (gloss, max_prob)
    """

    # Add batch dimension -> (1, C, T, H, W)
    ip_tensor = ip_tensor.unsqueeze(0).to(DEVICE)

    t = ip_tensor.shape[2]

    with torch.no_grad():
        per_frame_logits = i3d(ip_tensor)              # (1, num_classes, T')
        predictions = F.interpolate(per_frame_logits, t, mode='linear')
        predictions = predictions.transpose(2, 1)      # (1, T, num_classes)

    arr = predictions[0].cpu().numpy()                # (T, num_classes)
    out_labels = np.argsort(arr)                      # (T, num_classes)

    probs = F.softmax(torch.from_numpy(arr[0]), dim=0)
    max_prob = float(torch.max(probs))

    pred_idx = out_labels[0][-1]
    pred_word = wlasl_dict.get(pred_idx, "")

    print("Frame max prob:", round(max_prob, 4), "| gloss:", pred_word)

    if max_prob > threshold:
        return pred_word, max_prob
    else:
        return "", max_prob

# ---------------------------------------------------------
# STREAM VIDEO FROM FILE (NO FULL LOAD INTO RAM)
# ---------------------------------------------------------

def process_video_stream(video_path, batch=40, stride=20, threshold=0.5):
    """
    Stream the video from disk, process in sliding windows.
    Only keeps `batch` frames in RAM at a time.

    Returns: final "sentence" built from predicted glosses.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames_buffer = []
    sentence_tokens = []
    offset = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        offset += 1

        # Resize + normalize like original code
        h, w, c = frame.shape
        scale_h = 224 / h
        scale_w = 224 / w
        frame_resized = cv2.resize(frame, dsize=(0, 0), fx=scale_w, fy=scale_h)
        frame_norm = (frame_resized / 255.0) * 2.0 - 1.0
        frame_norm = frame_norm.astype(np.float32)

        if len(frames_buffer) < batch:
            frames_buffer.append(frame_norm)
        else:
            frames_buffer.pop(0)
            frames_buffer.append(frame_norm)

        # When we have a full window, and at each `stride` step, run inference
        if len(frames_buffer) == batch and (offset % stride == 0):
            window_np = np.asarray(frames_buffer, dtype=np.float32)  # (T, H, W, C)

            # (T, H, W, C) -> (C, T, H, W)
            window_tensor = torch.from_numpy(window_np.transpose(3, 0, 1, 2))

            gloss, prob = run_on_tensor(window_tensor, threshold=threshold)

            if gloss:
                # Avoid repeating the same gloss back-to-back
                if not sentence_tokens or sentence_tokens[-1] != gloss:
                    sentence_tokens.append(gloss)
                    print("Current tokens:", sentence_tokens)

    cap.release()

    if not sentence_tokens:
        return "(No confident prediction)"

    # Simple "sentence" = join unique predicted glosses
    final_sentence = " ".join(sentence_tokens)
    return final_sentence

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == '__main__':
    # use smaller 300-class model for lighter head
    num_classes = 300
    # choose only ONE weight file
    weights = 'archived/asl300/FINAL_nslt_300_iters=2997_top1=56.14_top5=79.94_top10=86.98.pt'
    # weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'

    print("📚 Loading label dictionary...")
    create_WLASL_dictionary()

    print("🧠 Loading I3D model on CPU (this may take a bit)...")
    load_model(weights, num_classes)
    print("✅ Model loaded.")

    video_path = input("\nEnter path to your recorded sign-language video file: ").strip()

    print(f"\n🎬 Processing video (streaming): {video_path}")
    final_sentence = process_video_stream(video_path, batch=40, stride=20, threshold=0.5)

    print("\n==============================")
    print("📝 FINAL PREDICTED SENTENCE (no NLP):")
    print(final_sentence)
    print("==============================")
