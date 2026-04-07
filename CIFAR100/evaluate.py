import os
import pandas as pd
import torch
import argparse
from utils.eval_utils import get_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

DATASET = "cifar100"


def main(model, train_path, test_path, K, savedir):
    os.makedirs(savedir, exist_ok=True)

    print(f"Loading train features from {train_path}...")
    database_train_df = pd.read_pickle(train_path)
    print(f"Loading test features from {test_path}...")
    database_test_df = pd.read_pickle(test_path)
    print(f"Database: {len(database_train_df):,} train | {len(database_test_df):,} test")

    print("Computing retrieval metrics...")
    get_metrics(
        model_str   = model,
        train_df    = database_train_df,
        test_df     = database_test_df,
        K           = K,
        filename    = filename,
        save_folder = savedir,
        device      = device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True,                      help="Model name (vit_s16, swin_s, gcvit_s)")
    parser.add_argument("--train",   required=True,                      help="Path to train features .pkl")
    parser.add_argument("--test",    required=True,                      help="Path to test features .pkl")
    parser.add_argument("--K",       required=True, nargs="+", type=int, help="Values of K for P@K")
    parser.add_argument("--savedir", required=True,                      help="Directory to save results")
    args = parser.parse_args()

    filename = f"{args.model}_{DATASET}_metrics"

    main(args.model, args.train, args.test, args.K, args.savedir)
