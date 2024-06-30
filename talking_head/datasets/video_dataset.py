import json
import random
import numpy as np
import torch
from pathlib import Path
from decord import VideoReader
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(
        self,
        n_motion_frames,
        n_sample_frames,
        img_size=(512, 512),
        sample_rate=16000,
        audio_margin=2,
        metadata_paths=None,
    ):
        super().__init__()
        self.n_motion_frames = n_motion_frames
        self.n_sample_frames = n_sample_frames
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.audio_margin = audio_margin

        self.metadata = []
        for metadata_path in metadata_paths:
            for line in Path(metadata_path).read_text().splitlines():
                self.metadata.append(json.loads(line))
        self.length = len(self.metadata)

        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.Normalize([0.5], [0.5], inplace=True),
            ]
        )

    def get_batch(self, idx):
        video_info = self.metadata[idx]
        video_path = video_info["video"]
        face_emb_path = video_info["face_emb"]
        audio_emb_path = video_info["audio_emb"]
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        assert (
            video_length
            > self.n_sample_frames + self.n_motion_frames + 2 * self.audio_margin
        )

        start_idx = random.randint(
            self.n_motion_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )

        batch_idx = np.arange(start_idx, start_idx + self.n_sample_frames)
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_idx).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.0  # (f, c, h, w)

        ref_img_idx = random.randint(
            self.n_motion_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )
        pixel_values_ref_img = torch.from_numpy(video_reader[ref_img_idx].asnumpy()).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        pixel_values_ref_img = pixel_values_ref_img / 255.0  # (1, c, h, w)

        face_emb = torch.load(face_emb_path)  # (512)
        full_audio_emb = torch.load(audio_emb_path)  # (n, 12, 768)
        assert full_audio_emb.shape[0] == video_length
        indices = torch.arange(2 * self.audio_margin + 1) - self.audio_margin  # [-2, -1, 0, 1, 2]
        center_indices = torch.arange(
            start_idx,
            start_idx + self.n_sample_frames,
        ).unsqueeze(1) + indices.unsqueeze(0)
        audio_emb = full_audio_emb[center_indices]  # (n, 5, 12, 768)

        if self.n_motion_frames > 0:
            batch_idx = np.arange(start_idx - self.n_motion_frames, start_idx)
            pixel_values_motion = torch.from_numpy(video_reader.get_batch(batch_idx).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values_motion = pixel_values_motion / 255.0  # (n_motion_frames, c, h, w)
            pixel_values_ref_img = torch.cat(
                [pixel_values_ref_img, pixel_values_motion], dim=0
            )
        
        del video_reader

        return pixel_values, pixel_values_ref_img, face_emb, audio_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, pixel_values_ref_img, face_emb, audio_emb = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img)

        sample = dict(
            pixel_values=pixel_values,
            pixel_values_ref_img=pixel_values_ref_img,
            face_emb=face_emb,
            audio_emb=audio_emb,
        )
        
        return sample
