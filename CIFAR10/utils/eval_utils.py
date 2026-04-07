import pandas as pd
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
import cv2 as cv
import random
from matplotlib import pyplot as plt
from torchmetrics.functional import retrieval_average_precision, retrieval_precision
from torch.nn import CosineSimilarity

def random_query(model_str, num_queries:int, topK:int, train_df:pd.DataFrame, test_df:pd.DataFrame, save_folder:str, device):
    num_images_test = len(test_df)

    sel_idx = random.sample(range(num_images_test), num_queries)

    big_grid = []

    database_matrix = torch.Tensor(train_df[model_str]).to(device)

    for i in sel_idx:
        # Extract a random test query image from the database
        query_img = test_df.iloc[i]['image']
        query_label = test_df.iloc[i]['label']
        query_ftrs = test_df.iloc[i][model_str]
        query_ftrs = torch.Tensor(query_ftrs).to(device) # transfer to CUDA for faster computations

        # Multiply the database matrix and the query feature vector (to be done for each query vector)
        # match_scores = database_matrix @ query_ftrs # inner product
        match_scores = CosineSimilarity()(database_matrix, query_ftrs) # cosine similarity
        sorted_scores, sorted_ind = torch.sort(match_scores, descending=True)

        # Take K best matches
        k_best_ind = sorted_ind[0:topK].cpu().numpy()

        # Grid for visualization
        grid_list = list(train_df.iloc[k_best_ind]['image'])

        query_img = cv.copyMakeBorder(query_img.transpose(1,2,0), 1, 1, 1, 1, cv.BORDER_CONSTANT, None, value=0)
        query_img = query_img.transpose(2,0,1)

        for j, img in enumerate(grid_list):
            img = img.transpose(1,2,0)
            
            color = (0,1,0) if query_label == train_df.iloc[k_best_ind[j]]['label'] else (1,0,0)
            img_with_border = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, value=color)

            grid_list[j] = torch.Tensor(img_with_border.transpose(2,0,1))

        grid_list = [torch.Tensor(query_img)] + grid_list # add query as first image
        grid = make_grid(grid_list, nrow=topK+1)

        big_grid.append(grid)

    plt.figure()
    big_grid = make_grid(big_grid, nrow=1, normalize=True)
    plt.imshow(big_grid.permute(1,2,0))
    plt.title(f'10 random queries and\n top 10 retrieved images [CIFAR-10, {model_str}]')
    plt.axis('off')
    plt.savefig(save_folder + f'CIFAR10_randQueries_{model_str}.png')


def get_metrics(model_str, train_df:pd.DataFrame, test_df:pd.DataFrame, K: list[int], save_folder:str, filename: str, device):
    sim_measure = CosineSimilarity()

    num_images_test = len(test_df)

    database_matrix = torch.Tensor(train_df[model_str]).to(device)
    database_labels = torch.Tensor(train_df['labels']).to(device)

    avg_prec  = torch.zeros(num_images_test).to(device)
    precision = {k: torch.zeros(num_images_test).to(device) for k in K}

    for i in range(num_images_test):
        query_ftrs  = torch.Tensor(test_df.iloc[i][model_str]).to(device)
        query_label = test_df.iloc[i]['labels']

        targets = torch.where(database_labels == query_label, True, False).to(device)
        preds   = sim_measure(database_matrix, query_ftrs)

        avg_prec[i] = retrieval_average_precision(preds, targets)
        for k in K:
            precision[k][i] = retrieval_precision(preds, targets, k)

    avg_prec  = avg_prec.cpu().numpy()
    for k in precision:
        precision[k] = precision[k].cpu().numpy()

    print(f"\nmAP: {avg_prec.mean():.4f}")
    for k in K:
        print(f"  P@{k}: {precision[k].mean():.4f}")

    # Histogram
    plt.figure()
    plt.hist(avg_prec, bins=100, range=(0, 1), edgecolor='k')
    plt.xlabel('Average Precision')
    plt.ylabel('Count')
    plt.title(f'Histogram of AP values [CIFAR-10, {model_str}]')
    plt.axvline(avg_prec.mean(), color='r', linestyle='dashed', linewidth=1)
    _, max_ylim = plt.ylim()
    plt.text(avg_prec.mean() * 0.65, max_ylim * 0.9, f'Mean: {avg_prec.mean():.4f}')
    plt.tight_layout()
    plt.savefig(save_folder + f'CIFAR10_AP_hist_{model_str}.png', dpi=300)
    plt.close()

    # Human-readable summary (.txt)
    summary_lines = [
        f"Model    : {model_str}",
        f"Dataset  : CIFAR-10  |  Queries: {num_images_test}",
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

    summary_path = save_folder + f'{filename}_summary.txt'
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSaved to {save_folder}:")
    print(f"  {filename}_summary.txt  ← human-readable metrics summary")
    print(f"  CIFAR10_AP_hist_{model_str}.png  ← histogram plot")
    print(f"  {filename}.npy          ← raw numpy dict")

    # Raw numpy dict (.npy)
    result_dict = {'model': model_str, 'AP': avg_prec, 'precision@k': precision}
    np.save(save_folder + f'{filename}.npy', result_dict)