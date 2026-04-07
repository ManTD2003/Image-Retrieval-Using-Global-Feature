import os
import timm
import torch
import torchvision
import pandas as pd
import argparse
from utils.extract_utils import extract_features

MODELS = {
    "vit_s16": "vit_small_patch16_224.augreg_in1k",
    "swin_s":  "swin_small_patch4_window7_224.ms_in1k",
    "gcvit_s": "gcvit_small.in1k",
}
DATASET      = "cifar100"
BATCH_SIZE   = 32
DATASET_ROOT = "../datasets/CIFAR-100"


def main(model_name, weights):
    os.makedirs(features_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    if weights is not None:
        model = timm.create_model(MODELS[model_name], num_classes=0, checkpoint_path=weights)
    else:
        model = timm.create_model(MODELS[model_name], pretrained=True, num_classes=0)
    model.eval()

    transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
    model     = model.to(device)

    train_set = torchvision.datasets.CIFAR100(root=DATASET_ROOT, train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR100(root=DATASET_ROOT, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Model: {model_name}")

    with torch.no_grad():
        print(f"Training set extraction ({model_name}):")
        train_features, train_labels = extract_features(model, train_loader, device)
        train_df = pd.DataFrame({
            model_name: train_features.cpu().numpy().tolist(),
            "labels":   train_labels.cpu().numpy(),
        })

        print(f"Test set extraction ({model_name}):")
        test_features, test_labels = extract_features(model, test_loader, device)
        test_df = pd.DataFrame({
            model_name: test_features.cpu().numpy().tolist(),
            "labels":   test_labels.cpu().numpy(),
        })

    train_df.to_pickle(train_features_path)
    test_df.to_pickle(test_features_path)
    print(f"Saved features to {features_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",  required=True,  help="Model name: vit_s16 | swin_s | gcvit_s")
    parser.add_argument("-w", "--weight", required=False, help="Path to fine-tuned weights (optional)")
    args = parser.parse_args()

    features_folder     = "./cifar100_features/"
    train_features_path = features_folder + f"{args.model}_{DATASET}_train.pkl"
    test_features_path  = features_folder + f"{args.model}_{DATASET}_test.pkl"

    main(args.model, args.weight)
