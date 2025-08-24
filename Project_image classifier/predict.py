#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Udacity Image Classifier - Predict script (predict.py)

import argparse
import json
from typing import Tuple, List

import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained checkpoint.")
    parser.add_argument("input", type=str, help="Path to input image.")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pth).")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes.")
    parser.add_argument("--category_names", type=str, default=None, help="Path to JSON mapping of categories to names.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available.")
    return parser.parse_args()

def rebuild_model_from_checkpoint(ckpt_path: str) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    arch = checkpoint.get("arch", "densenet121")
    hidden_units = checkpoint.get("hidden_units", 512)
    class_to_idx = checkpoint["class_to_idx"]

    if arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, len(class_to_idx)),
        )
    elif arch == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, len(class_to_idx)),
        )
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, len(class_to_idx)),
        )
    else:
        raise ValueError(f"Unsupported architecture in checkpoint: {arch}")

    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = class_to_idx
    return model

def process_image(image_path: str) -> torch.Tensor:
    im = Image.open(image_path).convert("RGB")
    width, height = im.size
    if width < height:
        new_size = (256, int(256 * height / width))
    else:
        new_size = (int(256 * width / height), 256)
    im = im.resize(new_size)
    left = (im.width - 224) / 2
    top = (im.height - 224) / 2
    right = left + 224
    bottom = top + 224
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np.transpose(np_image, (2, 0, 1))
    tensor = torch.from_numpy(np_image).type(torch.FloatTensor)
    return tensor

def predict(image_path: str, model: nn.Module, topk: int = 5, device: torch.device = torch.device("cpu")) -> Tuple[List[float], List[str]]:
    model.eval()
    model = model.to(device)
    img = process_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        ps = torch.softmax(output, dim=1)
        top_p, top_class_idx = torch.topk(ps, topk, dim=1)
    top_p = top_p.squeeze(0).cpu().numpy().tolist()
    top_class_idx = top_class_idx.squeeze(0).cpu().numpy().tolist()
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_class_idx]
    return top_p, top_classes

def main():
    args = get_args()
    device = torch.device("mps" if args.gpu and torch.mps.is_available() else "cpu")
    cat_to_name = None
    if args.category_names:
        with open(args.category_names, "r") as f:
            cat_to_name = json.load(f)
    model = rebuild_model_from_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, topk=args.top_k, device=device)
    if cat_to_name is not None:
        names = [cat_to_name.get(c, c) for c in classes]
        for rank, (p, c, n) in enumerate(zip(probs, classes, names), start=1):
            print(f"Top {rank}: class={c} name={n} prob={p:.4f}")
    else:
        for rank, (p, c) in enumerate(zip(probs, classes), start=1):
            print(f"Top {rank}: class={c} prob={p:.4f}")

if __name__ == "__main__":
    main()
