import torch
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.reducers import MultipleReducers, ThresholdReducer, MeanReducer
from torch.amp import autocast

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch, scaler=None):
    model.train()
    num_batches = len(train_loader)
    use_amp = scaler is not None
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=use_amp):
            embeddings = model(data)
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
            
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} Iteration {batch_idx}/{num_batches}: Loss = {loss.item():.4f}")


def get_all_embeddings(dataset, model):
    # BaseTester automatically handles batching data through the model and gathering results
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, model, accuracy_calculator):
    print("Extracting features for the Database (CIFAR-10 Train set)...")
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    
    print("Extracting features for the Query (CIFAR-10 Test set)...")
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    
    print("Computing metrics...")
    # Calculate accuracy: Test set acts as Query, Train set acts as Database
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )