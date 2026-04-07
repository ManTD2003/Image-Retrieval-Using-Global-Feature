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
DATASET_DIR     = "./dataset/train"
FEATURES_FOLDER = "./texture_features"


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder that also returns the file path alongside image and label."""
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path


def main(model_name, weights):
    os.makedirs(FEATURES_FOLDER, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else ""))

    if weights is not None:
        model = timm.create_model(MODELS[model_name], num_classes=0, checkpoint_path=weights)
    else:
        model = timm.create_model(MODELS[model_name], pretrained=True, num_classes=0)
    model.eval()

    transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
    model = model.to(device)

    dataset = ImageFolderWithPaths(root=DATASET_DIR, transform=transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    dataset_root = os.path.abspath(os.path.dirname(DATASET_DIR))  # points to ./dataset/

    print(f"Dataset: {len(dataset)} images | {len(dataset.classes)} classes")
    print(f"Extracting features with {model_name}...")

    all_features, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for i, (imgs, labels, paths) in enumerate(loader):
            imgs  = imgs.to(device)
            feats = model(imgs)
            all_features.append(feats.cpu())
            all_labels.extend(labels.tolist())
            # Store path relative to ./dataset/ so it matches groundtruth.json format
            rel_paths = [os.path.relpath(p, dataset_root).replace(os.sep, "/") for p in paths]
            all_paths.extend(rel_paths)
            if i % 20 == 0:
                print(f"  batch {i+1}/{len(loader)}")

    all_features = torch.cat(all_features, dim=0).numpy().tolist()

    df = pd.DataFrame({
        model_name: all_features,
        "labels":   all_labels,
        "filepath": all_paths,
    })

    out_path = os.path.join(FEATURES_FOLDER, f"{model_name}_{DATASET}.pkl")
    df.to_pickle(out_path)
    print(f"Saved {len(df)} feature vectors to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",  required=True,  help="Model name: vit_s16 | swin_s | gcvit_s")
    parser.add_argument("-w", "--weight", required=False, help="Path to fine-tuned weights (optional)")
    args = parser.parse_args()

    main(args.model, args.weight)
