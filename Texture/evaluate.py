import os
import sys
import argparse
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.eval_utils import get_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

DATASET = "texture"


def main(model, train_path, valid_path, K, savedir):
    os.makedirs(savedir, exist_ok=True)

    print(f"Loading train features from {train_path}...")
    train_df = pd.read_pickle(train_path)
    print(f"Loading valid features from {valid_path}...")
    valid_df = pd.read_pickle(valid_path)
    print(f"Gallery (train): {len(train_df):,} images | Queries (valid): {len(valid_df):,} images")

    print("Computing retrieval metrics...")
    get_metrics(
        model_str   = model,
        train_df    = train_df,
        test_df     = valid_df,
        K           = K,
        filename    = filename,
        save_folder = savedir,
        device      = device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True,                      help="Model name (vit_s16, swin_s, gcvit_s)")
    parser.add_argument("--train",   required=True,                      help="Path to train features .pkl")
    parser.add_argument("--valid",   required=True,                      help="Path to valid features .pkl")
    parser.add_argument("--K",       required=True, nargs="+", type=int, help="Values of K for P@K")
    parser.add_argument("--savedir", required=True,                      help="Directory to save results")
    args = parser.parse_args()

    filename = f"{args.model}_{DATASET}_metrics"

    main(args.model, args.train, args.valid, args.K, args.savedir)
