import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from torchvision import transforms
import random
import decord
from pytorch_lightning import LightningDataModule
from pathlib import Path
import json

from talking_head.models.vae.util import instantiate_from_config


class VideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        metadata_paths,
        num_frames=16,
        sample_stride=4,
        is_image=False,
        height=512,
        width=512,
    ):
        self.data_dir = data_dir
        self.metadata = []
        for metadata_path in metadata_paths:
            for line in Path(metadata_path).read_text().splitlines():
                self.metadata.append(json.loads(line))
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.transform = transforms.Compose(
            [
                transforms.Resize((height, width), antialias=True),
                transforms.CenterCrop((height, width)),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.is_image = is_image

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        video_info = self.metadata[idx]
        video_path = os.path.join(self.data_dir, video_info["video"])

        # audio_emb = sample["audio_emb"]

        try:
            video_reader = decord.VideoReader(video_path, num_threads=0)
            video_length = len(video_reader)

            if not self.is_image:
                clip_length = min(video_length, (self.num_frames - 1) * self.sample_stride + 1)
                start_idx = random.randint(0, video_length - clip_length)
                batch_idx = np.linspace(start_idx, start_idx + clip_length - 1, self.num_frames, dtype=int)
            else:
                batch_idx = [random.randint(0, video_length - 1)]

            pixel_values = torch.from_numpy(video_reader.get_batch(batch_idx).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del video_reader

        except Exception:
            print_str = f"Warning: missing video file {video_path}."
            print_str += " Resampling another video."
            print(print_str)
            return self.__getitem__(np.random.choice(self.__len__()))

        if self.is_image:
            pixel_values = pixel_values[0]

        pixel_values = self.transform(pixel_values)
        sample = dict(pixel_values=pixel_values)

        return sample


class VideoDataLoader(LightningDataModule):
    def __init__(self, batch_size, train_config, validation_config=None, num_workers=0, prefetch_factor=2, shuffle=True):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else 0
        self.shuffle = shuffle

        self.train_dataset = instantiate_from_config(train_config)
        if validation_config is not None:
            self.val_dataset = instantiate_from_config(validation_config)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
