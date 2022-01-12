"""Straightforward example on setting up the score fusion model."""

import torch
from torch import nn, optim
from models.motion import Motion
from models.depthcrnn import DepthCRNN
from dataloader import SHRECLoader
from train_single import train
import math
import time
import matplotlib.pyplot as plt

class PointDepthScoreFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.point_lstm_model=Motion(
            num_classes=14,
            pts_size=128,
            offsets=False,
            topk=16, downsample=(2, 2, 2),
            knn=[16, 24, 48, 12]
        )

        self.depth_crnn_model=DepthCRNN(
            num_classes=14,
            conv_blocks=[8, 16, 32],
            res_in=[50, 50],
            T=32,
            mlp_layers=[128],
            drop_prb=0.5,
            lstm_units=128,
            lstm_layers=2,
            use_bilstm=True,
            use_bn=True,
            actn_type="relu"
        )

    def forward(self, x_ptcloud, x_depth):

        x_ptcloud = self.point_lstm_model(x_ptcloud)
        x_depth = self.depth_crnn_model(x_depth)

        x_fused = (x_ptcloud + x_depth) / 2
        return x_fused

def count_params(model):
    return sum(map(lambda p: p.data.numel(), model.parameters()))

################################
# run for the model
################################

if __name__ == "__main__":
    print("Starting run:")

    device = "cuda"
    model = PointDepthScoreFusion()
    model = model.to(device)

    print("Successfully created score fusion model")
    num_params = count_params(model)

    print(f"Score fusion model has {num_params} parameters.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # dataloader = dummy_data_loader(N=10, batch_size=2)
    shrec = SHRECLoader(framerate=32)
    train_size = int(0.8 * len(shrec))
    val_size = len(shrec) - train_size
    train_set, val_set = torch.utils.data.random_split(shrec, [train_size, val_size])

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

    # dataloader = torch.utils.data.DataLoader(
    #     dataset=shrec,
    #     batch_size=4,
    #     shuffle=True,
    #     num_workers=0,
    # )

    accuracies, losses,val_accuracies,val_losses = train(model, train_loader, val_loader, criterion, optimizer, 30, device)
    plt.plot(losses)
    plt.show()
    plt.plot(accuracies)
    plt.show()
    plt.plot(val_losses)
    plt.show()
    plt.plot(val_accuracies)
    plt.show()

    with open('accuracies.txt', 'w') as f:
        for item in accuracies:
            f.write("%s\n" % item)
    with open('losses.txt', 'w') as f:
        for item in losses:
            f.write("%s\n" % item)
    print("Completed run.")

    # completed training
