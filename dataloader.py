import re

from cv2 import phase
from train import train
import torch
import numpy as np
from utils.utils_ptcloud import *
from torch.utils import data
import albumentations as A

def insert(original, new, pos):
    '''Inserts new inside original at pos.'''
    return original[:pos] + new + original[pos:]

class SHRECLoader(data.Dataset):
    def __init__(self, framerate, phase="train", datatype="depth", inputs_type="pts"):
        self.phase = phase
        self.datatype = datatype
        self.inputs_type = inputs_type
        self.framerate = framerate
        self.inputs_list = self.get_inputs_list()
        self.prefix = "./SHREC2017/gesture_{}/finger_{}/subject_{}/essai_{}"
        self.r = re.compile('[ \t\n\r:]+')
        print(len(self.inputs_list))
        if phase == "train":
            self.transform_ptclouds = self.transform_init_ptclouds("train")
            self.transform_depthim = self.transform_init_depthim("train")
        elif phase == "test":
            self.transform_ptclouds = self.transform_init_ptclouds("test")
            self.transform_depthim = self.transform_init_depthim("test")

    def __getitem__(self, index):
        splitLine = self.r.split(self.inputs_list[index])
        # label28 = int(splitLine[-3]) - 1
        label14 = int(splitLine[-4]) - 1

        point_clouds = np.load(
            insert(self.prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3]), "Processed_", 2)
            + "/pts_label.npy")[:, :, :7]

        point_clouds = point_clouds[self.key_frame_sampling(len(point_clouds), self.framerate)]
        for i in range(self.framerate):
            point_clouds[i, :, 3] = i
        point_clouds = np.dstack((point_clouds, np.zeros_like(point_clouds)))[:, :, :7]
        point_clouds = self.normalize_ptclouds(point_clouds, self.framerate)

        depth_images = np.load(
            insert(self.prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3]), "DepthProcessed_", 2)
            + "/depth_video.npy").astype('float32')
        depth_images = self.normalize_depthim(depth_images)


        # label14 = torch.from_numpy(label14).long()
        # print(label14.dtype)
        return point_clouds, depth_images, label14, self.inputs_list[index]

    def get_inputs_list(self):
        prefix = "./SHREC2017"
        if self.phase == "train":
            inputs_path = prefix + "/train_gestures.txt"
        if self.phase == "test":
            inputs_path = prefix + "/test_gestures.txt"
        inputs_list = open(inputs_path).readlines()
        return inputs_list

    def __len__(self):
        return len(self.inputs_list)

    def normalize_ptclouds(self, pts, fs):
        timestep, pts_size, channels = pts.shape
        pts = pts.reshape(-1, channels)
        pts = pts.astype(float)
        pts[:, 0] = (pts[:, 0] - np.mean(pts[:, 0])) / 120
        pts[:, 1] = (pts[:, 1] - np.mean(pts[:, 1])) / 160
        pts[:, 3] = (pts[:, 3] - fs / 2) / fs * 2
        if (pts[:, 2].max() - pts[:, 2].min()) != 0:
            pts[:, 2] = (pts[:, 2] - np.mean(pts[:, 2])) / np.std(pts[:, 2])
        pts = self.transform_ptclouds(pts)
        pts = pts.reshape(timestep, pts_size, channels)
        return pts
    
    def normalize_depthim(self,image_sequence: np.ndarray):

        for i in range(0, image_sequence.shape[0]):
            for x in range(0,image_sequence.shape[1]):
                for y in range(0,image_sequence.shape[2]):
                    if image_sequence[i][x][y]<200:
                        image_sequence[i][x][y] = 0
                    else:
                        image_sequence[i][x][y] = 155 + (10 * (image_sequence[i][x][y] - np.amin(image_sequence[i])) / (np.amax(image_sequence[i]) - np.amin(image_sequence[i])) + 0.5) * ((255 - 155) / 10)
        image_sequence = (image_sequence / 255).astype(np.float32)

        if self.transform_depthim != None :
            data = self.transform_depthim(image=image_sequence[0])
            image_sequence[0] = data["image"]

            # Use same params for all frames
            for i in range(1, image_sequence.shape[0]):
                image_sequence[i] = A.ReplayCompose.replay(data['replay'], image=image_sequence[i])["image"]

        return image_sequence

        

    @staticmethod
    def key_frame_sampling(key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        index = [int(j / factor) for j in range(frame_size)]
        return index

    @staticmethod
    def transform_init_ptclouds(phase):
        if phase == 'train':
            transform = Compose([
                PointcloudToTensor(),
                PointcloudScale(lo=0.9, hi=1.1),
                PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
                # PointcloudJitter(std=0.01, clip=0.05),
                PointcloudRandomInputDropout(max_dropout_ratio=0.2),
            ])
        else:
            transform = Compose([
                PointcloudToTensor(),
            ])
        return transform
    
    @staticmethod
    def transform_init_depthim(phase,shift_limit: float=0.2, scale_limit:float=0.2, rotate_limit: int = 20, p : float = 0.5):
        if phase == 'train':
            transform = A.ReplayCompose([
            A.ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, p=p)
        ])

        else:
            transform = None
        return transform


