import os, csv, random
import numpy as np
from decord import VideoReader
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPImageProcessor

# from talking_head.utils.util import zero_rank_print


class TalkingHeadDataset(Dataset):
    def __init__(
        self,
        csv_path,
        sample_size=512,
        sample_stride=4,
        sample_n_frames=16,
        is_image=False,
    ):
        # zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        # zero_rank_print(f"data scale: {self.length}")

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.clip_image_processor = CLIPImageProcessor()

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_dir = video_dict['videoid']
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        clip_inputs = self.clip_image_processor(pixel_values[0], return_tensors="pt").pixel_values[0]
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, clip_inputs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, clip_inputs = self.get_batch(idx)
                break
            except Exception:
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, clip_inputs=clip_inputs)
        return sample


if __name__ == "__main__":
    image_dataset = TalkingHeadDataset(
        csv_path="/mnt/data/public/dataset/talking_head/video_datasets/metadata.csv",
        sample_stride=4,
        sample_n_frames=16,
        is_image=True,
    )
    video_dataset = TalkingHeadDataset(
        csv_path="/mnt/data/public/dataset/talking_head/video_datasets/metadata.csv",
        sample_stride=4,
        sample_n_frames=16,
        is_image=False,
    )
    print(f"dataset size: {len(video_dataset)}")
    for dataset in [image_dataset, video_dataset]:
        for i, d in enumerate(dataset):
            if i > 3:
                break
            print(d["pixel_values"].shape)  # (c h w) if is_image=True, (f c h w) otherwise
            print(d["clip_inputs"].shape)  # (c 224 224)
