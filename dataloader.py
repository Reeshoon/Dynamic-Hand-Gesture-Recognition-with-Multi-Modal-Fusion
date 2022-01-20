import re
import torch
import numpy as np
from utils import *
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
            self.transform = self.transform_init("train")
        elif phase == "test":
            self.transform = self.transform_init("test")

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
        point_clouds = self.normalize(point_clouds, self.framerate)

        depth_images = np.load(
            insert(self.prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3]), "DepthProcessed_", 2)
            + "/depth_video.npy").astype('float32')
        depth_images = self.image_shift_scale_rotate(depth_images)

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

    def normalize(self, pts, fs):
        timestep, pts_size, channels = pts.shape
        pts = pts.reshape(-1, channels)
        pts = pts.astype(float)
        pts[:, 0] = (pts[:, 0] - np.mean(pts[:, 0])) / 120
        pts[:, 1] = (pts[:, 1] - np.mean(pts[:, 1])) / 160
        pts[:, 3] = (pts[:, 3] - fs / 2) / fs * 2
        if (pts[:, 2].max() - pts[:, 2].min()) != 0:
            pts[:, 2] = (pts[:, 2] - np.mean(pts[:, 2])) / np.std(pts[:, 2])
        pts = self.transform(pts)
        pts = pts.reshape(timestep, pts_size, channels)
        return pts

    @staticmethod
    def key_frame_sampling(key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        index = [int(j / factor) for j in range(frame_size)]
        return index

    @staticmethod
    def transform_init(phase):
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

    def image_shift_scale_rotate(image_sequence: np.ndarray, shift_limit: float=0.05, scale_limit:float=0.05, rotate_limit: int = 10, p : float = 0.5):
        """Shift scale and rotate image within a certain limit."""

        transform = A.ReplayCompose([
            A.ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, p=p)
        ])

        data = transform(image=image_sequence[0])
        image_sequence[0] = data["image"]

        # Use same params for all frames
        for i in range(1, image_sequence.shape[0]):
            image_sequence[i] = A.ReplayCompose.replay(data['replay'], image=image_sequence[i])["image"]

        return image_sequence
