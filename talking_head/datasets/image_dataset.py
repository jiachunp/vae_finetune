import random
import cv2
import numpy as np
import json
from PIL import Image

import torch
import torchvision.transforms as transforms
# from torch.utils.data.dataset import Dataset
from datasets import Dataset, Image
import torch.distributed as dist

from transformers import CLIPImageProcessor


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


# Function to convert bounding boxes to binary masks
def bounding_boxes_to_masks(box, resolution=(512, 512)):
    # Create an empty binary mask
    masks = np.zeros(resolution, dtype=np.uint8)
    
    # Convert normalized coordinates to pixel values
    x1, y1, x2, y2 = box
    x1_pixel, y1_pixel = int(x1 * resolution[1]), int(y1 * resolution[0])
    x2_pixel, y2_pixel = int(x2 * resolution[1]), int(y2 * resolution[0])
    
    # Draw the rectangle on the mask
    cv2.rectangle(masks, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), 1, thickness=-1)
    
    return masks


# def make_train_dataset(args, accelerator):
def make_train_dataset():
    clip_image_processor = CLIPImageProcessor()
    
    image_transforms = transforms.Compose(
        [
            # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["clip_inputs"] = clip_image_processor(images, return_tensors="pt").pixel_values
        images = [image_transforms(image) for image in images]
        examples["pixel_values"] = images

        ref_images = [image.convert("RGB") for image in examples["ref_image"]]
        examples["ref_clip_inputs"] = clip_image_processor(ref_images, return_tensors="pt").pixel_values
        ref_images = [image_transforms(image) for image in ref_images]
        examples["ref_pixel_values"] = ref_images

        face_masks = [bounding_boxes_to_masks(bbox) for bbox in examples["bbox"]]
        face_masks = np.array(face_masks)  # array -> tensor is much faster than list[array] to tensor
        examples["face_masks"] = torch.tensor(face_masks).unsqueeze(1)

        return examples

    # with accelerator.main_process_first():
        # train_dataset = dataset.with_transform(preprocess_train)
    # dataset = ImageDataset(json_path="/mnt/data/public/dataset/talking_head/image_datasets/HDTF/metadata.json")
    json_path="/mnt/data/public/dataset/talking_head/image_datasets/HDTF/metadata.json"
    with open(json_path, "r") as f:
        dataset_list = json.load(f)
    dataset_dict = {
        "image": [data_dict["image"] for data_dict in dataset_list],
        "ref_image": [data_dict["ref_image"] for data_dict in dataset_list],
        "bbox": [data_dict["annotation"]["bbox"] for data_dict in dataset_list],
    }
    dataset = Dataset.from_dict(dataset_dict).cast_column("image", Image()).cast_column("ref_image", Image())
    print(dataset)
    train_dataset = dataset.with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    clip_inputs = torch.stack([example["clip_inputs"] for example in examples])
    clip_inputs = clip_inputs.to(memory_format=torch.contiguous_format).float()

    ref_pixel_values = torch.stack([example["ref_pixel_values"] for example in examples])
    ref_pixel_values = ref_pixel_values.to(memory_format=torch.contiguous_format).float()

    ref_clip_inputs = torch.stack([example["ref_clip_inputs"] for example in examples])
    ref_clip_inputs = ref_clip_inputs.to(memory_format=torch.contiguous_format).float()

    face_masks = torch.stack([example["face_masks"] for example in examples])
    face_masks = face_masks.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "clip_inputs": clip_inputs,
        "ref_pixel_values": ref_pixel_values,
        "ref_clip_inputs": ref_clip_inputs,
        "face_masks": face_masks,
    }


if __name__ == "__main__":
    train_dataset = make_train_dataset()
    # train_dataset = ImageDataset(json_path="/public/datasets/talking_head/image_datasets/HDTF_v2/metadata.json")
    print(f"Lenght of the dataset: {len(train_dataset)}")
    sample = train_dataset[1]
    print(sample["pixel_values"].shape)  # [3, 512, 512]
    print(sample["clip_inputs"].shape)  # [3, 512, 512]
    # print(sample["clip_ref_images"].shape)  # [1, 3, 224, 224]
    # print(sample["ref_pixel_values"].shape)  # [3, 512, 512]
    # print(sample["drop_image_embeds"].shape)
    print(sample["face_masks"].shape)  # [1, 512, 512]
    # print(sample["face_masks"][0][200])
    # print("Test passed!")
