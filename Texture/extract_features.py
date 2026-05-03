import os
import sys
import timm
import torch
import pandas as pd
import argparse
from torchvision import datasets
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.extract_utils import extract_features

DATASET    = "texture"
MODELS     = {
    "vit_s16": "vit_small_patch16_224.augreg_in1k",
    "swin_s":  "swin_small_patch4_window7_224.ms_in1k",
    "gcvit_s": "gcvit_small.in1k",
}
BATCH_SIZE      = 32
TRAIN_DIR       = "./dataset/train"
VALID_DIR       = "./dataset/valid"
FEATURES_FOLDER = "./texture_features"


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder that also returns the file path alongside image and label."""
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path


def extract_split(model, loader, dataset_root, device):
    """Extract features, labels, and relative file paths for one split."""
    all_features, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for i, (imgs, labels, paths) in enumerate(loader):
            imgs  = imgs.to(device)
            feats = model(imgs)
            all_features.append(feats.cpu())
            all_labels.extend(labels.tolist())
            rel_paths = [os.path.relpath(p, dataset_root).replace(os.sep, "/") for p in paths]
            all_paths.extend(rel_paths)
            if i % 20 == 0:
                print(f"  batch {i+1}/{len(loader)}")

    return torch.cat(all_features, dim=0), all_labels, all_paths


def main(model_name, weights):
    os.makedirs(FEATURES_FOLDER, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    if weights is not None:
        model = timm.create_model(MODELS[model_name], num_classes=0, checkpoint_path=weights)
    else:
        model = timm.create_model(MODELS[model_name], pretrained=True, num_classes=0)
    model.eval()

    transform    = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
    model        = model.to(device)
    dataset_root = os.path.abspath("./dataset")

    train_dataset = ImageFolderWithPaths(root=TRAIN_DIR, transform=transform)
    valid_dataset = ImageFolderWithPaths(root=VALID_DIR, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    valid_loader  = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Model: {model_name}")
    print(f"Training set extraction ({model_name})  — {len(train_dataset)} images:")
    train_feats, train_labels, train_paths = extract_split(model, train_loader, dataset_root, device)
    train_df = pd.DataFrame({
        model_name: train_feats.numpy().tolist(),
        "labels":   train_labels,
        "filepath": train_paths,
    })

    print(f"Validation set extraction ({model_name})  — {len(valid_dataset)} images:")
    valid_feats, valid_labels, valid_paths = extract_split(model, valid_loader, dataset_root, device)
    valid_df = pd.DataFrame({
        model_name: valid_feats.numpy().tolist(),
        "labels":   valid_labels,
        "filepath": valid_paths,
    })

    train_path = os.path.join(FEATURES_FOLDER, f"{model_name}_{DATASET}_train.pkl")
    valid_path = os.path.join(FEATURES_FOLDER, f"{model_name}_{DATASET}_valid.pkl")
    train_df.to_pickle(train_path)
    valid_df.to_pickle(valid_path)
    print(f"Saved features to {FEATURES_FOLDER}")
    print(f"  {train_path}  ({len(train_df)} images)")
    print(f"  {valid_path}  ({len(valid_df)} images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",  required=True,  help="Model name: vit_s16 | swin_s | gcvit_s")
    parser.add_argument("-w", "--weight", required=False, help="Path to fine-tuned weights (optional)")
    args = parser.parse_args()

    main(args.model, args.weight)
