import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchmetrics.functional import retrieval_average_precision, retrieval_precision
from torch.nn import CosineSimilarity


def get_metrics(model_str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                K: list, save_folder: str, filename: str, device):
    """
    Evaluate retrieval performance using class labels as relevance (same class = relevant).

    Args:
        model_str   : column name for feature vectors in the DataFrames
        train_df    : database (train) features DataFrame
        test_df     : query (test) features DataFrame
        K           : list of integers for P@K
        save_folder : directory to save result files
        filename    : base name for output files
        device      : torch device
    """
    sim_measure = CosineSimilarity()

    num_queries = len(test_df)

    database_matrix = torch.Tensor(np.array(train_df[model_str].tolist())).to(device)
    database_labels = torch.Tensor(train_df["labels"].values).to(device)

    avg_prec  = torch.zeros(num_queries).to(device)
    precision = {k: torch.zeros(num_queries).to(device) for k in K}

    for i in range(num_queries):
        query_ftrs  = torch.Tensor(test_df.iloc[i][model_str]).to(device)
        query_label = test_df.iloc[i]["labels"]

        targets = torch.where(database_labels == query_label, True, False).to(device)
        preds   = sim_measure(database_matrix, query_ftrs)

        avg_prec[i] = retrieval_average_precision(preds, targets)
        for k in K:
            precision[k][i] = retrieval_precision(preds, targets, k)

        if (i + 1) % 1000 == 0:
            print(f"  evaluated {i+1}/{num_queries} queries...")

    avg_prec  = avg_prec.cpu().numpy()
    for k in precision:
        precision[k] = precision[k].cpu().numpy()

    print(f"\nmAP: {avg_prec.mean():.4f}")
    for k in K:
        print(f"  P@{k}: {precision[k].mean():.4f}")

    # Histogram
    plt.figure()
    plt.hist(avg_prec, bins=100, range=(0, 1), edgecolor="k")
    plt.xlabel("Average Precision")
    plt.ylabel("Count")
    plt.title(f"Histogram of AP values [CIFAR-100, {model_str}]")
    plt.axvline(avg_prec.mean(), color="r", linestyle="dashed", linewidth=1)
    _, max_ylim = plt.ylim()
    plt.text(avg_prec.mean() * 0.65, max_ylim * 0.9, f"Mean: {avg_prec.mean():.4f}")
    plt.tight_layout()
    plt.savefig(save_folder + f"CIFAR100_AP_hist_{model_str}.png", dpi=300)
    plt.close()

    # Human-readable summary (.txt)
    summary_lines = [
        f"Model    : {model_str}",
        f"Dataset  : CIFAR-100  |  Queries: {num_queries}",
        "",
        f"{'Metric':<18} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}",
        "-" * 56,
        f"{'mAP':<18} {avg_prec.mean():>8.4f}  {avg_prec.std():>8.4f}"
        f"  {avg_prec.min():>8.4f}  {avg_prec.max():>8.4f}",
    ]
    for k in K:
        p = precision[k]
        summary_lines.append(
            f"{'P@'+str(k):<18} {p.mean():>8.4f}  {p.std():>8.4f}  {p.min():>8.4f}  {p.max():>8.4f}"
        )

    summary_path = save_folder + f"{filename}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nSaved to {save_folder}:")
    print(f"  {filename}_summary.txt        ← human-readable metrics summary")
    print(f"  CIFAR100_AP_hist_{model_str}.png  ← histogram plot")
    print(f"  {filename}.npy                ← raw numpy dict")

    # Raw numpy dict (.npy)
    result_dict = {"model": model_str, "AP": avg_prec, "precision@k": precision}
    np.save(save_folder + f"{filename}.npy", result_dict)
