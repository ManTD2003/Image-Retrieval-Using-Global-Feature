import os
import argparse
import multiprocessing
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from torch.amp import GradScaler
import timm
from ..utils.metric_train_utils import *

torch.backends.cudnn.benchmark = True

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")

    models_dict = {
        "vit_s16": 'vit_small_patch16_224.augreg_in1k',
        "swin_s": 'swin_small_patch4_window7_224.ms_in1k',
        "gcvit_s": 'gcvit_small.in1k'
    }

    model_save_folder = './model_save/'
    os.makedirs(model_save_folder, exist_ok=True)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)

    # Initialize loss, miner, reducer outside the loop if they don't depend on the model
    reducer_dict = {"pos_loss": ThresholdReducer(0.1), "neg_loss": MeanReducer()}
    reducer = MultipleReducers(reducer_dict)
    distance = distances.CosineSimilarity()
    loss_func = losses.ContrastiveLoss(pos_margin=1.0, neg_margin=0, distance=distance, reducer=reducer)
    mining_func = miners.MultiSimilarityMiner(epsilon=0.1)
    accuracy_calculator = AccuracyCalculator(include=("mean_average_precision",), k=128)

    # Calculate optimal number of workers
    num_workers = min(8, multiprocessing.cpu_count())

    for model_name in args.models:
        if model_name not in models_dict:
            print(f"Skipping {model_name}: Not in the supported list.")
            continue
            
        print(f'Initializing model: {model_name}')
        
        # Hardcoded to cifar10 for the save path
        model_save_path = os.path.join(model_save_folder, f'{model_name}_cifar10.pth')

        # CRITICAL: num_classes=0 to remove the final Linear layer, extracting only Feature Embeddings
        model = timm.create_model(models_dict[model_name], pretrained=True, num_classes=0)

        # Get transform config
        transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg, model=model))

        model = model.to(device)

        if torch.cuda.is_available():
            model = torch.compile(model)

        # Optimize param groups
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        
        # Add Learning Rate Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        # Load CIFAR-10 Dataset directly
        dataset1 = datasets.CIFAR10(root='../datasets/CIFAR-10', train=True, download=True, transform=transform)
        dataset2 = datasets.CIFAR10(root='../datasets/CIFAR-10', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

        print(f'Starting training of {model_name} for {args.epochs} epochs:')
        for epoch in range(1, args.epochs + 1):
            # Note: In utils.metric_train_utils, train() should use optimizer.zero_grad(set_to_none=True) for speedup
            train(model, loss_func, mining_func, device, train_loader, optimizer, epoch, scaler=scaler)
            
            # Update learning rate after each epoch
            scheduler.step() 

        print(f'Saving {model_name}...')
        torch.save(model.state_dict(), model_save_path)

        print(f'Evaluating {model_name} on test set...')
        with torch.no_grad():
            test(dataset1, dataset2, model, accuracy_calculator)

        # Thoroughly clean VRAM for the next model
        del model, optimizer, scheduler, train_loader, test_loader, dataset1, dataset2
        if torch.cuda.is_available():
            torch._dynamo.reset() # Clear torch.compile cache
            torch.cuda.empty_cache()

    print('\nFinished all training and evaluation processes!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric Learning Fine-tuning for Vision Models")

    parser.add_argument('-m', '--models', nargs='+', default=["vit_s16", "swin_s", "gcvit_s"], help='Models to train')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-n', '--epochs', type=int, default=10, help='Number of epochs')

    args = parser.parse_args()
    main(args)