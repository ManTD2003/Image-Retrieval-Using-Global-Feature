import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchmetrics.functional import (
    retrieval_average_precision,
    retrieval_precision,
    retrieval_recall,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

DATASET          = "texture"
GROUNDTRUTH_PATH = "./groundtruth.json"


def get_metrics(model_str, db_df, groundtruth, K, save_folder, filename):
    """
    Evaluate retrieval performance on the Texture dataset using groundtruth.json.

    Relevance is defined by the groundtruth file: each query has exactly 5 similar images.
    The query image itself is excluded from the ranked list.

    Args:
        model_str   : column name for feature vectors in db_df
        db_df       : DataFrame with columns [model_str, 'labels', 'filepath']
        groundtruth : list of dicts with keys 'query' and 'similar'
        K           : list of integers for precision/recall@K
        save_folder : directory to save result files
        filename    : base name for the output .npy file
    """
    all_filepaths = db_df["filepath"].tolist()
    filepath_to_idx = {fp: i for i, fp in enumerate(all_filepaths)}

    db_matrix = torch.tensor(
        np.array(db_df[model_str].tolist()), dtype=torch.float32
    ).to(device)                                      # [N, D]
    db_norm = F.normalize(db_matrix, dim=1)           # normalize once for cosine sim

    num_queries = len(groundtruth)
    avg_prec  = torch.zeros(num_queries, device=device)
    precision = {k: torch.zeros(num_queries, device=device) for k in K}
    recall    = {k: torch.zeros(num_queries, device=device) for k in K}

    skipped = 0
    for i, entry in enumerate(groundtruth):
        query_path   = entry["query"]
        similar_set  = set(entry["similar"])

        if query_path not in filepath_to_idx:
            skipped += 1
            continue

        query_idx  = filepath_to_idx[query_path]
        query_norm = db_norm[query_idx].unsqueeze(0)          # [1, D]
        scores     = (db_norm @ query_norm.T).squeeze(1)      # [N] cosine similarity

        # Exclude the query itself from the ranked list
        scores[query_idx] = -2.0

        targets = torch.tensor(
            [fp in similar_set for fp in all_filepaths],
            dtype=torch.bool, device=device
        )                                                      # [N]

        avg_prec[i] = retrieval_average_precision(scores, targets)
        for k in K:
            precision[k][i] = retrieval_precision(scores, targets, k)
            recall[k][i]    = retrieval_recall(scores, targets, k)

        if (i + 1) % 500 == 0:
            print(f"  evaluated {i+1}/{num_queries} queries...")

    if skipped > 0:
        print(f"Warning: {skipped} queries not found in features file (skipped).")

    avg_prec_np  = avg_prec.cpu().numpy()
    precision_np = {k: precision[k].cpu().numpy() for k in K}
    recall_np    = {k: recall[k].cpu().numpy()    for k in K}

    print(f"\nmAP: {avg_prec_np.mean():.4f}")
    for k in K:
        print(f"  P@{k}: {precision_np[k].mean():.4f}  |  R@{k}: {recall_np[k].mean():.4f}")

    # Histogram of average precision values
    plt.figure()
    plt.hist(avg_prec_np, bins=50, range=(0, 1), edgecolor="k")
    plt.xlabel("Average Precision")
    plt.ylabel("Count")
    plt.title(f"Histogram of AP values [Texture, {model_str}]")
    plt.axvline(avg_prec_np.mean(), color="r", linestyle="dashed", linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(avg_prec_np.mean() * 0.65, max_ylim * 0.9, f"Mean: {avg_prec_np.mean():.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"Texture_AP_hist_{model_str}.png"), dpi=300)
    plt.close()

    result_dict = {
        "model":       model_str,
        "AP":          avg_prec_np,
        "precision@k": precision_np,
        "recall@k":    recall_np,
    }
    np.save(os.path.join(save_folder, f"{filename}.npy"), result_dict)
    print(f"Results saved to {save_folder}")


def main(model, features_path, K, savedir):
    os.makedirs(savedir, exist_ok=True)

    print(f"Loading features from {features_path}...")
    db_df = pd.read_pickle(features_path)
    print(f"Database: {len(db_df)} images | feature dim: {len(db_df[model].iloc[0])}")

    print(f"Loading groundtruth from {GROUNDTRUTH_PATH}...")
    with open(GROUNDTRUTH_PATH, "r") as f:
        gt_raw = json.load(f)
    groundtruth = list(gt_raw.values())
    print(f"Groundtruth: {len(groundtruth)} queries")

    print("Computing retrieval metrics...")
    filename = f"{model}_{DATASET}_metrics"
    get_metrics(
        model_str   = model,
        db_df       = db_df,
        groundtruth = groundtruth,
        K           = K,
        save_folder = savedir,
        filename    = filename,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True,                 help="Model name (vit_s16, swin_s, gcvit_s)")
    parser.add_argument("--features", required=True,                 help="Path to the extracted features .pkl file")
    parser.add_argument("--K",        required=True, nargs="+", type=int, help="Values of K for precision/recall@K")
    parser.add_argument("--savedir",  required=True,                 help="Directory to save results")
    args = parser.parse_args()

    main(args.model, args.features, args.K, args.savedir)
