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
from PIL import Image
import sys
sys.path.append("/mnt/data/jiachunpan/talking-pose-vae/")
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
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.is_image = is_image

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_info = self.metadata[idx]
        file_path = '/'.join(image_info["video"].split("/")[:-1]) + '_dwpose'
        path = image_info["video"].split("/")[-1]
        path = path.split('.')[0]
        image_folder = os.path.join(file_path, path)
        
        # image_info = self.metadata[idx]
        # path = image_info["video"].split("/")[-2]
        # subpath = image_info["video"].split("/")[-1]
        # subpath = path.split('.')[0]
        # image_folder = os.path.join(self.data_dir, path+"_dwpose")
        # image_folder = os.path.join(image_folder, path)

        try:
            # List all images in the folder
            all_images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg'))]

            # Ensure there are enough images to sample
            if len(all_images) < self.num_frames:
                raise ValueError(f"Not enough images in folder {image_folder} to sample {self.num_frames} images.")
            
            # Randomly sample `num_images` from the folder
            selected_images = random.sample(all_images, self.num_frames)

            image_list = []
            [image_list.append(Image.open(img)) for img in selected_images]

            image_list = np.array(image_list)

        except Exception:
            print_str = f"Warning: missing video file {image_folder}."
            print_str += " Resampling another video."
            print(print_str)
            return self.__getitem__(np.random.choice(self.__len__()))
        
        pixel_values = torch.from_numpy(image_list).permute(0, 3, 1, 2).contiguous() 
        pixel_values = pixel_values / 255.
        pixel_values = self.transform(pixel_values)
        sample = dict(pixel_values=pixel_values)
        
        del all_images

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

if __name__ == "__main__":
    train_dataset = VideoDataset(
        data_dir="/mnt/data/public/dataset/talking_body/training_videos/luoxiang_512_dwpose",
        num_images=16,
        is_image=True,
        metadata_paths="/mnt/data/public/dataset/talking_body/embeddings/luoxiang_512/metadata.jsonl",
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for step, batch in enumerate(train_dataloader):
        import pdb; pdb.set_trace()
        print(batch["pixel_values"].shape)