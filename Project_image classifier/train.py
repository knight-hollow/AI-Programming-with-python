#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Udacity Image Classifier - Train script (train.py)

import argparse
import os
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

def get_args():
    parser = argparse.ArgumentParser(description="Train a deep network on a dataset of images and save the checkpoint.")
    parser.add_argument("data_dir", type=str, help="Root directory of dataset with train/valid/test subfolders.")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save checkpoint (.pth).")
    parser.add_argument("--arch", type=str, default="densenet121", choices=["densenet121", "vgg13", "resnet50"],
                        help="Model architecture to use.")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate for optimizer.")
    parser.add_argument("--hidden_units", type=int, default=512, help="Hidden units for classifier head.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    return parser.parse_args()

def build_model(arch: str, hidden_units: int, output_size: int = 102) -> Tuple[nn.Module, str]:
    if arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, output_size),
        )
        param_prefix = "classifier"
    elif arch == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        in_features = model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, output_size),
        )
        model.classifier = classifier
        param_prefix = "classifier"
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, output_size),
        )
        param_prefix = "fc"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model, param_prefix

def get_dataloaders(data_dir: str):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir  = os.path.join(data_dir, "test")

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=eval_transforms)
    test_data  = datasets.ImageFolder(test_dir,  transform=eval_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=64)

    return train_loader, valid_loader, test_loader, train_data

def validate(model, dataloader, criterion, device) -> Tuple[float, float]:
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            ps = torch.softmax(outputs, dim=1)
            top_p, top_class = ps.topk(1, dim=1)
            acc_sum += (top_class.view(-1) == labels).sum().item()
            n += labels.size(0)
    model.train()
    return loss_sum / n, acc_sum / n

def train():
    args = get_args()
    device = torch.device("mps" if args.gpu and torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, train_data = get_dataloaders(args.data_dir)

    model, param_prefix = build_model(args.arch, args.hidden_units, output_size=len(train_data.classes))
    model = model.to(device)

    for name, param in model.named_parameters():
        if not name.startswith(param_prefix):
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=args.learning_rate)

    epochs = args.epochs
    step = 0
    print_every = 40

    for e in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            step += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % print_every == 0:
                val_loss, val_acc = validate(model, valid_loader, criterion, device)
                print(f"Epoch: {e+1}/{epochs}.. Train loss: {running_loss/print_every:.3f}.. Valid loss: {val_loss:.3f}.. Valid acc: {val_acc:.3f}")
                running_loss = 0.0

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.3f}  Test acc: {test_acc:.3f}")

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        "arch": args.arch,
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
        "hidden_units": args.hidden_units,
        "head": "fc" if args.arch == "resnet50" else "classifier",
    }

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "checkpoint.pth"
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to: {ckpt_path.resolve()}")

if __name__ == "__main__":
    train()
