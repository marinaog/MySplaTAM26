import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils import data
from natsort import natsorted

from .basedataset import GradSLAMDataset

class RawSLAMDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        self.raw = config_dict.get('raw')
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        translation = pvec[:3]
        euler_angles_deg = pvec[3:]

        # Create rotation matrix from Euler angles (in degrees)
        rotation = Rotation.from_euler('xyz', euler_angles_deg, degrees=True)
        pose[:3, :3] = rotation.as_matrix()
        pose[:3, 3] = translation
        return pose

    def get_filepaths(self):
        groundtruth_file = os.path.join(self.input_folder, 'groundtruth.txt')
        with open(groundtruth_file, 'r') as f:
            poses_lines = f.readlines()

        image_list = os.path.join(self.input_folder, 'sRGB')
        depth_list = os.path.join(self.input_folder, 'depth')

        color_paths, depth_paths = [], []
        for line in poses_lines[1:]:
            frame_name = line.strip().split()[0]
            frame_name += '.png'
            color_paths += [os.path.join(image_list, frame_name)]
            depth_paths += [os.path.join(depth_list, frame_name)]

        embedding_paths = None

        return color_paths, depth_paths, embedding_paths

    def load_poses(self):

        frame_rate = 32
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(self.input_folder, 'groundtruth.txt')):
            pose_list = os.path.join(self.input_folder, 'groundtruth.txt')
        else:
            print("No groundtruth file found in ", self.input_folder)

        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 2:].astype(np.float64)

        poses = []
        for pose in pose_vecs:
            c2w = self.pose_matrix_from_quaternion(pose)
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)