# जय श्री राम
# note this program requires more RAM due to CPU processing, its crassing in my system with 12 GB RAM.
# it might work in systems with 16 GB RAM or more.
# i suggest using cpu-based-interface2.0.py instead, for systems with less RAM.

"""
CPU-only offline inference script for WLASL I3D model.

Workflow:
1. User records a ~15s sign video (e.g. MP4).
2. User gives the path of that video to this script.
3. Script processes the video with I3D + NGram + KeyToText.
4. Prints the final predicted sentence.
"""

import os
import math
from itertools import chain
import pickle

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_i3d import InceptionI3d
from keytotext import pipeline
import language
from dotenv import load_dotenv

# ------------------------------------------------------------------
# ENV & GLOBALS
# ------------------------------------------------------------------

load_dotenv("posts/nlp/.env", override=True)

DEVICE = torch.device("cpu")   # CPU ONLY

# Globals that will be filled in load_model / create_WLASL_dictionary
i3d = None
wlasl_dict = None
nlp = None
params = None
n_gram_counts_list = None
vocabulary = None


# ------------------------------------------------------------------
# UTILS: LOAD VIDEO FRAMES (FROM FILE, NOT WEBCAM)
# ------------------------------------------------------------------

def load_rgb_frames_from_file(video_path):
    """
    Load all RGB frames from a video file, resize to 224x224,
    normalize to [-1, 1], and return as a numpy array of shape:
        (num_frames, 224, 224, 3) float32
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to 224x224 (like original code)
        h, w, c = frame.shape
        scale_h = 224 / h
        scale_w = 224 / w
        frame_resized = cv2.resize(frame, dsize=(0, 0), fx=scale_w, fy=scale_h)

        # Normalize to [-1, 1]
        frame_norm = (frame_resized / 255.0) * 2.0 - 1.0
        frames.append(frame_norm.astype(np.float32))

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames read from video. Check the file/path.")

    return np.asarray(frames, dtype=np.float32)  # (T, H, W, C)


# ------------------------------------------------------------------
# MODEL & NLP LOADING (CPU)
# ------------------------------------------------------------------

def load_model(weights, num_classes):
    """
    Load I3D model, KeyToText NLP, and N-gram language model on CPU.
    """
    global i3d, nlp, params, n_gram_counts_list, vocabulary

    # ---- I3D (CPU) ----
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)

    state_dict = torch.load(weights, map_location=DEVICE)
    i3d.load_state_dict(state_dict)

    i3d.to(DEVICE)
    i3d.eval()

    # ---- KeyToText (NLP) ----
    # Pre-trained options in original code: 'k2t', 'k2t-base',
    # 'mrm8488/t5-base-finetuned-common_gen', 'k2t-new'
    nlp = pipeline("k2t-new")
    params = {
        "do_sample": True,
        "num_beams": 5,
        "no_repeat_ngram_size": 2,
        "early_stopping": True
    }

    # ---- NGram model ----
    with open("NLP/nlp_data_processed", "rb") as fp:
        train_data_processed = pickle.load(fp)

    with open("NLP/nlp_gram_counts", "rb") as fp:
        n_gram_counts_list = pickle.load(fp)

    vocabulary = list(set(chain.from_iterable(train_data_processed)))


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


# ------------------------------------------------------------------
# INFERENCE ON A WINDOW OF FRAMES
# ------------------------------------------------------------------

def run_on_tensor(ip_tensor, threshold=0.5):
    """
    ip_tensor: torch tensor of shape (C, T, H, W)
    Returns: predicted gloss (string) or " " if below threshold.
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

    print("Frame-level max prob:", max_prob)
    print("Predicted gloss:", pred_word)

    if max_prob > threshold:
        return pred_word
    else:
        return " "


# ------------------------------------------------------------------
# PROCESS A FULL VIDEO (OFFLINE) -> SENTENCE
# ------------------------------------------------------------------

def process_video(video_path, batch=40, stride=20, threshold=0.5):
    """
    Process the entire video in sliding windows.

    - video_path: path to .mp4/.avi video (about 15 seconds recommended)
    - batch: number of frames per window (like original code)
    - stride: how often to run prediction (every N frames)
    - threshold: confidence threshold for accepting a gloss

    Returns: final sentence (string)
    """
    frames_np = load_rgb_frames_from_file(video_path)  # (T, H, W, C)
    total_frames = frames_np.shape[0]

    print(f"Total frames read: {total_frames}")

    frames_buffer = []
    text_list = []
    word_list = []
    sentence = ""
    text_count = 0
    offset = 0

    for idx in range(total_frames):
        frame = frames_np[idx]  # (H, W, C), already normalized [-1, 1]
        offset += 1

        if len(frames_buffer) < batch:
            frames_buffer.append(frame)
        else:
            # Keep a rolling buffer of latest 'batch' frames
            frames_buffer.pop(0)
            frames_buffer.append(frame)

        if len(frames_buffer) == batch and (offset % stride == 0):
            # Window ready for inference
            window_np = np.asarray(frames_buffer, dtype=np.float32)  # (batch, H, W, C)

            # (T, H, W, C) -> (C, T, H, W)
            window_tensor = torch.from_numpy(window_np.transpose(3, 0, 1, 2))

            gloss = run_on_tensor(window_tensor, threshold=threshold)

            if gloss != " ":
                text_count += 1

                # Avoid repeating same gloss back-to-back
                if (text_list and word_list and
                        text_list[-1] != gloss and word_list[-1] != gloss) or not text_list:
                    text_list.append(gloss)
                    word_list.append(gloss)
                    sentence = sentence + " " + gloss

                # NGram suggestion
                word = language.get_suggestions(text_list, n_gram_counts_list, vocabulary, k=1.0)
                if word != " .":
                    sentence += word
                    text_list.append(word)

                # After we have a few tokens, use KeyToText
                if text_count > 2:
                    sentence = nlp(text_list, **params)

                print("Current sentence:", sentence)

    if not sentence.strip():
        sentence = "(No confident prediction)"

    return sentence.strip()


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == '__main__':
    num_classes = 300
    #weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    weights = 'archived/asl300/FINAL_nslt_300_iters=2997_top1=56.14_top5=79.94_top10=86.98.pt'
    
    print("📚 Loading label dictionary...")
    create_WLASL_dictionary()

    print("🧠 Loading I3D + NLP models on CPU (this may take a bit)...")
    load_model(weights, num_classes)
    print("✅ Models loaded.")

    # 1) Ask user for path OR
    # 2) You can hard-code it, e.g. video_path = "samples/my_sign.mp4"
    video_path = input("\nEnter path to your recorded sign-language video file: ").strip()

    print(f"\n🎬 Processing video: {video_path}")
    final_sentence = process_video(video_path)

    print("\n==============================")
    print("📝 FINAL PREDICTED SENTENCE:")
    print(final_sentence)
    print("==============================")
