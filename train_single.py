import torch
from torch import nn, optim
from typing import Callable, Tuple
from torch.utils.data import DataLoader
import math
import time
import matplotlib.pyplot as plt
import os

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: Callable, device: str):
    acc = 0
    avg_loss = 0
    model.eval()
    for pt_clouds, depth_ims, labels, _ in dataloader:
        depth_ims = torch.unsqueeze(depth_ims, 2)
        pt_clouds, depth_ims, labels = pt_clouds.to(device), depth_ims.to(device), labels.to(device)

        output = model(pt_clouds, depth_ims)
        correct = output.max(1)[1].eq(labels).sum()
        acc += correct.item()
        loss = criterion(output, labels)
        avg_loss += loss.item()

    model.train()
    acc /= len(dataloader.dataset)
    avg_loss /= len(dataloader)
    return acc, avg_loss

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: Callable, optimizer: optim.Optimizer, n_epoch: int, device: str):
  
    best_acc = 0
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    model.train()

    for epoch in range(n_epoch):
        t0 = time.time()
        acc = 0
        avg_loss = 0
        
        for pt_clouds, depth_ims, labels, _ in train_loader:
            depth_ims = torch.unsqueeze(depth_ims, 2)
            pt_clouds, depth_ims, labels = pt_clouds.to(device), depth_ims.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(pt_clouds, depth_ims)
            correct = output.max(1)[1].eq(labels).sum()
            acc += correct.item()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        # after each epoch
        accuracies.append(acc/len(train_loader.dataset))
        losses.append(avg_loss/len(train_loader))
        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0, "train_acc": acc/(len(train_loader.dataset)), "avg_loss_per_ep": avg_loss/len(train_loader)}
        print(f'Finished epoch: {epoch + 1}')
        print(log_dict)

        if (epoch+1) % 1 == 0:
            val_acc, val_loss = evaluate(model, val_loader, criterion, device)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            print("Val loss: ",val_loss," Val acc: ",val_acc)
            if val_acc > best_acc:
                torch.save(model.state_dict(), './model_weights/')
                best_acc = val_acc

    return accuracies, losses,val_accuracies,val_losses
