import numpy as np
import torch


def extract_features(model, dataloader, device):
    features      = torch.Tensor().to(device)
    labels_tensor = torch.Tensor().to(device)

    for i, (imgs, labels) in enumerate(dataloader):
        imgs   = imgs.to(device)
        output = model(imgs)

        features      = torch.concat([features, output])
        labels_tensor = torch.concat([labels_tensor, labels.to(device)])

        if i % 99 == 0:
            print(f"{i+1}/{len(dataloader)}")

    return features, labels_tensor
