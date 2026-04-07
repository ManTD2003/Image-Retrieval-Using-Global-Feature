import os
import sys
import argparse
import multiprocessing
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
import timm
from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.reducers import MultipleReducers, ThresholdReducer, MeanReducer
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metric_train_utils import train, get_all_embeddings, test

torch.backends.cudnn.benchmark = True

MODELS = {
    "vit_s16": "vit_small_patch16_224.augreg_in1k",
    "swin_s":  "swin_small_patch4_window7_224.ms_in1k",
    "gcvit_s": "gcvit_small.in1k",
}
DATASET_DIR      = "./dataset/train"
MODEL_SAVE_FOLDER = "./model_save"
TRAIN_RATIO      = 0.8   # fraction of data used for training; rest is used for evaluation


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else ""))

    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)

    use_amp = torch.cuda.is_available()
    scaler  = GradScaler("cuda", enabled=use_amp)

    reducer_dict = {"pos_loss": ThresholdReducer(0.1), "neg_loss": MeanReducer()}
    reducer      = MultipleReducers(reducer_dict)
    distance     = distances.CosineSimilarity()
    loss_func    = losses.ContrastiveLoss(pos_margin=1.0, neg_margin=0, distance=distance, reducer=reducer)
    mining_func  = miners.MultiSimilarityMiner(epsilon=0.1)
    accuracy_calculator = AccuracyCalculator(include=("mean_average_precision",), k=128)

    num_workers = min(8, multiprocessing.cpu_count())

    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Skipping {model_name}: not in the supported model list.")
            continue

        print(f"\nInitializing model: {model_name}")
        model_save_path = os.path.join(MODEL_SAVE_FOLDER, f"{model_name}_texture.pth")

        model     = timm.create_model(MODELS[model_name], pretrained=True, num_classes=0)
        transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg, model=model))
        model     = model.to(device)

        if torch.cuda.is_available():
            model = torch.compile(model)

        # Load full dataset and split into train / val
        full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
        n_train = int(len(full_dataset) * TRAIN_RATIO)
        n_val   = len(full_dataset) - n_train
        train_set, val_set = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Dataset split: {n_train} train / {n_val} val | {len(full_dataset.classes)} classes")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, persistent_workers=True)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        print(f"Training {model_name} for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            train(model, loss_func, mining_func, device, train_loader, optimizer, epoch, scaler=scaler)
            scheduler.step()

        print(f"Saving {model_name} to {model_save_path}...")
        # Unwrap torch.compile wrapper before saving
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(raw_model.state_dict(), model_save_path)

        print(f"Evaluating {model_name}...")
        with torch.no_grad():
            test(train_set, val_set, model, accuracy_calculator)

        del model, optimizer, scheduler, train_loader, val_loader, train_set, val_set, full_dataset
        if torch.cuda.is_available():
            torch._dynamo.reset()
            torch.cuda.empty_cache()

    print("\nFinished all training and evaluation processes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric learning fine-tuning on the Texture dataset")
    parser.add_argument("-m", "--models",       nargs="+", default=["vit_s16", "swin_s", "gcvit_s"],
                        help="Models to train")
    parser.add_argument("--lr",                 type=float, default=3e-5,  help="Learning rate")
    parser.add_argument("--weight_decay",       type=float, default=5e-4,  help="Weight decay")
    parser.add_argument("-b", "--batch_size",   type=int,   default=32,    help="Batch size")
    parser.add_argument("-n", "--epochs",       type=int,   default=10,    help="Number of epochs")
    args = parser.parse_args()

    main(args)
