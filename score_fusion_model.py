"""Straightforward example on setting up the score fusion model."""

import torch
from torch import nn, optim
from models.motion import Motion
from models.depthcrnn import DepthCRNN
from dataloader import SHRECLoader
import math

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


def dummy_data_loader(N, batch_size):
    """Creates dummy data so that you can test the model."""
    num_batches = math.ceil(N/batch_size)
    num_frames = 32
    num_points = 128
    for i in range(num_batches):
        dummy_point_clouds = torch.randn(batch_size, num_frames, 128, 4)
        dummy_depth_images = torch.randn(batch_size, num_frames, 1, 50, 50)
        labels = torch.randint(14, (batch_size,)).long()
        yield dummy_point_clouds, dummy_depth_images, labels

        def insert(original, new, pos):
            '''Inserts new inside original at pos.'''
            return original[:pos] + new + original[pos:]


def count_params(model):
    return sum(map(lambda p: p.data.numel(), model.parameters()))

################################
# demo run for the model
################################

if __name__ == "__main__":
    print("Starting demo run:")
    
    device = "cpu"
    model = PointDepthScoreFusion()
    model = model.to(device)

    print("Successfully created score fusion model")
    num_params = count_params(model)

    print(f"Score fusion model has {num_params} parameters.")


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # dataloader = dummy_data_loader(N=10, batch_size=2)
    shrec = SHRECLoader(framerate=32)
    dataloader = torch.utils.data.DataLoader(
        dataset=shrec,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    for pt_clouds, depth_ims, labels, _ in dataloader:
        depth_ims = torch.unsqueeze(depth_ims, 2)
        pt_clouds, depth_ims, labels = pt_clouds.to(device), depth_ims.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(pt_clouds, depth_ims)
        print(f"Shape of output: {output.shape}")

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        print(f"[+] Successfully trained 1 step. Loss: {loss.item()}")

    print("Completed demo run.")