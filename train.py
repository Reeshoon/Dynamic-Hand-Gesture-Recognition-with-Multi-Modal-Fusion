import torch
from torch import nn, optim
from typing import Callable, Tuple
from torch.utils.data import DataLoader
import math
import time
import matplotlib.pyplot as plt
import os
from utils.misc import log,calc_step,save_model

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

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: Callable, optimizer: optim.Optimizer, n_epoch: int, device: str,schedulers: dict,config):
    best_acc = 0
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    log_file = os.path.join('./saved_files/', "training_log.txt")
    model.train()
    n_batches = len(train_loader)

    for epoch in range(n_epoch):
        t0 = time.time()
        acc = 0
        avg_loss = 0
        
        for batch_index, (pt_clouds, depth_ims, labels, _ ) in enumerate(train_loader):
            step = calc_step(epoch, n_batches, batch_index)

            depth_ims = torch.unsqueeze(depth_ims, 2)
            pt_clouds, depth_ims, labels = pt_clouds.to(device), depth_ims.to(device), labels.to(device)

            optimizer.zero_grad()
            
            output = model(pt_clouds, depth_ims)
            correct = output.max(1)[1].eq(labels).sum()
            acc += correct.item()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # schedulers          
            if schedulers["warmup"] is not None and epoch < 10:
                schedulers["warmup"].step()
            
            elif schedulers["scheduler"] is not None and epoch >= 10:
                schedulers["scheduler"].step()        

            avg_loss += loss.item()

        print(acc)
        print(len(train_loader.dataset))

        # after each epoch
        accuracies.append(acc/len(train_loader.dataset))
        losses.append(avg_loss/len(train_loader))

        
        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0, "train_acc": acc/(len(train_loader.dataset)), "avg_loss_per_ep": avg_loss/len(train_loader)}
        log(log_dict, step,config)

        if (epoch+1) % 1 == 0:
            val_acc, val_loss = evaluate(model, val_loader, criterion, device)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            log_dict = {"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
            log(log_dict, step, config)
          
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, save_path, model, optimizer, log_file) # save best val ckpt

    return accuracies, losses,val_accuracies,val_losses,best_model


def test(model: nn.Module, criterion: Callable, test_loader: DataLoader,device: str):
    test_acc, test_loss = evaluate(model, test_loader, criterion, device)
    return test_acc,test_loss