# inference_i3d.py 13.12.25

import torch
import json
import os

from pytorch_i3d import InceptionI3d
from videotransforms import CenterCrop, Normalize
import torchvision.transforms as transforms


# -----------------------------
# CONFIG
# -----------------------------
NUM_CLASSES = 2000
MODEL_PATH = "archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt"
LABEL_MAP_PATH = "preprocess/nslt_2000.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# LOAD LABEL MAP
# -----------------------------
def load_label_map(path):
    with open(path, "r") as f:
        label_map = json.load(f)
    return label_map


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_i3d_model():
    model = InceptionI3d(
        num_classes=NUM_CLASSES,
        in_channels=3
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# -----------------------------
# PREPROCESS TRANSFORMS
# -----------------------------
def get_video_transforms():
    return transforms.Compose([
        CenterCrop(224),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
    ])


# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
@torch.no_grad()
def predict_sign(model, video_tensor, label_map):
    """
    video_tensor shape: (1, 3, T, 224, 224)
    """

    video_tensor = video_tensor.to(DEVICE)

    logits = model(video_tensor)
    probs = torch.softmax(logits, dim=1)

    top_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, top_class].item()

    predicted_word = label_map[str(top_class)]

    return predicted_word, confidence


# -----------------------------
# TEST WITH DUMMY INPUT
# -----------------------------
if __name__ == "__main__":
    print("Loading label map...")
    label_map = load_label_map(LABEL_MAP_PATH)

    print("Loading I3D model...")
    model = load_i3d_model()

    print("Running dummy inference...")

    # Dummy input: batch=1, channels=3, frames=16, H=224, W=224
    dummy_video = torch.randn(1, 3, 16, 224, 224)

    word, conf = predict_sign(model, dummy_video, label_map)

    print(f"Predicted sign: {word}")
    print(f"Confidence: {conf:.4f}")
