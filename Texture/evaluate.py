import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchmetrics.functional import (
    retrieval_average_precision,
    retrieval_precision,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

DATASET = "texture"


def get_metrics(model_str, db_df, K, save_folder, filename):
    """
    Evaluate retrieval using class labels as relevance (same class = relevant).

    Every image acts as a query against the full database.
    The query itself is excluded from its own ranked list.

    Args:
        model_str   : column name for feature vectors in db_df
        db_df       : DataFrame with columns [model_str, 'labels']
        K           : list of integers for precision/recall@K
        save_folder : directory to save result files
        filename    : base name for the output .npy file
    """
    db_matrix = torch.tensor(
        np.array(db_df[model_str].tolist()), dtype=torch.float32
    ).to(device)                                   # [N, D]
    db_norm   = F.normalize(db_matrix, dim=1)      # normalize once; dot product = cosine sim
    db_labels = torch.tensor(db_df["labels"].values, dtype=torch.long, device=device)  # [N]

    N = len(db_df)
    avg_prec  = torch.zeros(N, device=device)
    precision = {k: torch.zeros(N, device=device) for k in K}

    for i in range(N):
        query_norm  = db_norm[i].unsqueeze(0)              # [1, D]
        scores      = (db_norm @ query_norm.T).squeeze(1)  # [N] cosine similarity
        scores[i]   = -2.0                                 # exclude query itself

        targets = (db_labels == db_labels[i])              # [N] bool: same class = relevant

        avg_prec[i] = retrieval_average_precision(scores, targets)
        for k in K:
            precision[k][i] = retrieval_precision(scores, targets, k)

        if (i + 1) % 500 == 0:
            print(f"  evaluated {i+1}/{N} queries...")

    avg_prec_np  = avg_prec.cpu().numpy()
    precision_np = {k: precision[k].cpu().numpy() for k in K}

    print(f"\nmAP: {avg_prec_np.mean():.4f}")
    for k in K:
        print(f"  P@{k}: {precision_np[k].mean():.4f}")

    # ── Histogram ─────────────────────────────────────────────────────────────
    plt.figure()
    plt.hist(avg_prec_np, bins=100, range=(0, 1), edgecolor="k")
    plt.xlabel("Average Precision")
    plt.ylabel("Count")
    plt.title(f"Histogram of AP values [Texture, {model_str}]")
    plt.axvline(avg_prec_np.mean(), color="r", linestyle="dashed", linewidth=1)
    _, max_ylim = plt.ylim()
    plt.text(avg_prec_np.mean() * 0.65, max_ylim * 0.9, f"Mean: {avg_prec_np.mean():.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"Texture_AP_hist_{model_str}.png"), dpi=300)
    plt.close()

    #  Summary (.txt) ─────────────────────────────────────────
    summary_lines = [
        f"Model    : {model_str}",
        f"Dataset  : Texture  |  Queries: {len(avg_prec_np)}",
        "",
        f"{'Metric':<18} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}",
        "-" * 56,
        f"{'mAP':<18} {avg_prec_np.mean():>8.4f}  {avg_prec_np.std():>8.4f}"
        f"  {avg_prec_np.min():>8.4f}  {avg_prec_np.max():>8.4f}",
    ]
    for k in K:
        p = precision_np[k]
        summary_lines.append(
            f"{'P@'+str(k):<18} {p.mean():>8.4f}  {p.std():>8.4f}  {p.min():>8.4f}  {p.max():>8.4f}"
        )

    summary_path = os.path.join(save_folder, f"{filename}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # ── Per-image AP (.csv) ───────────────────────────────────────────────────
    if "filepath" in db_df.columns:
        row_labels = db_df["filepath"].str.replace(r"^train/", "", regex=True).tolist()
    else:
        row_labels = list(range(len(avg_prec_np)))

    csv_data = {"AP": avg_prec_np}
    for k in K:
        csv_data[f"P@{k}"] = precision_np[k]
    pd.DataFrame(csv_data, index=row_labels).to_csv(
        os.path.join(save_folder, f"{filename}_per_image.csv"), index_label="image"
    )

    # ── Raw numpy dict (.npy) ─────────────────────────────────────────────────
    result_dict = {
        "model":       model_str,
        "AP":          avg_prec_np,
        "precision@k": precision_np,
    }
    np.save(os.path.join(save_folder, f"{filename}.npy"), result_dict)

    print(f"\nSaved to {save_folder}:")
    print(f"  {filename}_summary.txt    ← human-readable metrics summary")
    print(f"  {filename}_per_image.csv  ← per-image AP, P@K")
    print(f"  Texture_AP_hist_{model_str}.png   ← histogram plot")
    print(f"  {filename}.npy            ← raw numpy dict")


def main(model, features_path, K, savedir):
    os.makedirs(savedir, exist_ok=True)

    print(f"Loading features from {features_path}...")
    db_df = pd.read_pickle(features_path)
    print(f"Database: {len(db_df)} images | feature dim: {len(db_df[model].iloc[0])} | classes: {db_df['labels'].nunique()}")

    print("Computing retrieval metrics...")
    filename = f"{model}_{DATASET}_metrics"
    get_metrics(
        model_str   = model,
        db_df       = db_df,
        K           = K,
        save_folder = savedir,
        filename    = filename,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True,                      help="Model name (vit_s16, swin_s, gcvit_s)")
    parser.add_argument("--features", required=True,                      help="Path to the extracted features .pkl file")
    parser.add_argument("--K",        required=True, nargs="+", type=int, help="Values of K for precision/recall@K")
    parser.add_argument("--savedir",  required=True,                      help="Directory to save results")
    args = parser.parse_args()

    main(args.model, args.features, args.K, args.savedir)
