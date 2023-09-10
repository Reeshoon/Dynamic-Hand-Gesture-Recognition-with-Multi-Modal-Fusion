import torch
import numpy as np
from torch import nn, optim
from models.motion import Motion
from models.depthcrnn import DepthCRNN
from dataloader import SHRECLoader


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

        #average score fusion
        x_fused = (x_ptcloud + x_depth) / 2

        #max score fusion
        #x_fused = torch.max(x_ptcloud,x_depth)

        return x_fused

