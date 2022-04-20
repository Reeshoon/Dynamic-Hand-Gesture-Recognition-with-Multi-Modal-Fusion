import torch
from torch import nn, optim
from models.score_fusion_model import PointDepthScoreFusion
from models.feature_fusion_model import PointDepthFeatureFusion
from dataloader import SHRECLoader
from train import train,test
import math
import time
import matplotlib.pyplot as plt
import os
#import wandb
from utils.loss import LabelSmoothingLoss
from utils.scheduler import WarmUpLR, get_scheduler
from utils.misc import count_params,seed_everything
################################
# run for the model
################################

def training_pipeline():
    seed_everything(47)
    device = "cuda"
    model = PointDepthScoreFusion()
    #model = PointDepthFeatureFusion()
    model = model.to(device)

    print("Successfully created score fusion model")
    num_params = count_params(model)
    print(f"Score fusion model has {num_params} parameters.")

    

    ###########  Optimizer Definition  ###########
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)

    ###########  Criterion Definition  ###########
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingLoss(num_classes= 14 , smoothing=0.1)

    # dataloader = dummy_data_loader(N=10, batch_size=2)
    train_set = SHRECLoader(framerate=32)
    val_set = SHRECLoader(framerate=32, phase='validation')
    test_set = SHRECLoader(framerate=32,phase='test')
    
    # train_size = int(0.8 * len(shrec))
    # val_size = len(shrec) - train_size
    # train_set, val_set = torch.utils.data.random_split(shrec, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        shuffle=True,
        num_workers=0,
    )

    # lr scheduler
    schedulers = {
        "warmup": None,
        "scheduler": None
    }
    schedulers["warmup"] = WarmUpLR(optimizer, total_iters=len(train_loader) * 10)
    total_iters = len(train_loader) * max(1, (30 - 10))
    schedulers["scheduler"] = get_scheduler(optimizer, "cosine_annealing", total_iters)


    #os.environ["WANDB_API_KEY"] = "87fd3cc00cd83c882da8bf145ecc92d00dae8bf0"
    #Adjust confugs for each training
    config ={
        "learning_rate": 0.001,
        "epochs": 30,
        "batch_size": 4,
        "optimizer" : "adamW",
        "criterion" : "LabelSmoothingLoss"
    }

    #with wandb.init(project='thesis-test-1', name='Depth Image Quantization', config=config):
    accuracies, losses,val_accuracies,val_losses,best_model= train(model, train_loader, val_loader, criterion, optimizer, 30, device,schedulers,config)
    test_acc, test_loss = test(best_model, criterion, test_loader,device)

    print("\nTest Accuracy :",test_acc,"\nTest Loss : ",test_loss)

    plt.plot(accuracies,'r',label='train_acc')
    plt.plot(val_accuracies,'g',label='val_acc')
    plt.title('accuracies')
    plt.legend(loc="lower right")

    plt.show()
    plt.plot(val_losses,'b',label='val_loss')
    plt.plot(losses,'y',label='train_loss')
    plt.title('losses')
    plt.legend(loc="upper right")
    plt.show()

    print("Completed run.")

    # completed training



if __name__ == "__main__":
    print("Starting run:")
    training_pipeline()

    