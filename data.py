# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import torch

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobVideoDataset(Dataset):
    def __init__(self, root, phase, image_size, ep_len=6, image_glob='*.png'):
        self.root = root
        self.image_size = image_size
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        else:
            pass

        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            frame_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, image_glob)))
            for path in image_paths:
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        for image_loc in self.episodes[idx]:
            image = Image.open(image_loc).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video
