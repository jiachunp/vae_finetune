# pylint: disable=W1203,W0718
"""
This module is used to process videos to prepare data for training. It utilizes various libraries and models
to perform tasks such as video frame extraction, audio extraction, face mask generation, and face embedding extraction.
The script takes in command-line arguments to specify the input and output directories, GPU status, level of parallelism,
and rank for distributed processing.

Usage:
    python -m scripts.data_preprocess --input_dir /path/to/video_dir --dataset_name dataset_name --gpu_status --parallelism 4 --rank 0

Example:
    python -m scripts.data_preprocess -i data/videos -o data/output -g -p 4 -r 0
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List

import cv2
import torch
from tqdm import tqdm

from talking_head.datasets.audio_processor import AudioProcessor
from talking_head.datasets.image_processor import ImageProcessorForDataProcessing
from talking_head.utils.util import convert_video_to_images, extract_audio_from_videos

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def setup_directories(video_path: Path) -> dict:
    """
    Setup directories for storing processed files.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        dict: A dictionary containing paths for various directories.
    """
    base_dir = video_path.parent.parent
    dirs = {
        "face_mask": base_dir / "face_mask",
        "sep_pose_mask": base_dir / "sep_pose_mask",
        "sep_face_mask": base_dir / "sep_face_mask",
        "sep_lip_mask": base_dir / "sep_lip_mask",
        "face_emb": base_dir / "face_emb",
        "audio_emb": base_dir / "audio_emb"
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def process_single_video(video_path: Path,
                         output_dir: Path,
                         image_processor: ImageProcessorForDataProcessing,
                         audio_processor: AudioProcessor,
                         step: int) -> None:
    """
    Process a single video file.

    Args:
        video_path (Path): Path to the video file.
        output_dir (Path): Directory to save the output.
        image_processor (ImageProcessorForDataProcessing): Image processor object.
        audio_processor (AudioProcessor): Audio processor object.
        gpu_status (bool): Whether to use GPU for processing.
    """
    assert video_path.exists(), f"Video path {video_path} does not exist"
    dirs = setup_directories(video_path)
    logging.info(f"Processing video: {video_path}")

    try:
        if step == 1:
            images_output_dir = output_dir / 'images' / video_path.stem
            images_output_dir.mkdir(parents=True, exist_ok=True)
            images_output_dir = convert_video_to_images(
                video_path, images_output_dir)
            logging.info(f"Images saved to: {images_output_dir}")

            audio_output_dir = output_dir / 'audios'
            audio_output_dir.mkdir(parents=True, exist_ok=True)
            audio_output_path = audio_output_dir / f'{video_path.stem}.wav'
            audio_output_path = extract_audio_from_videos(
                video_path, audio_output_path)
            logging.info(f"Audio extracted to: {audio_output_path}")

            face_mask, _, sep_pose_mask, sep_face_mask, sep_lip_mask = image_processor.preprocess(
                images_output_dir)
            cv2.imwrite(
                str(dirs["face_mask"] / f"{video_path.stem}.png"), face_mask)
            cv2.imwrite(str(dirs["sep_pose_mask"] /
                        f"{video_path.stem}.png"), sep_pose_mask)
            cv2.imwrite(str(dirs["sep_face_mask"] /
                        f"{video_path.stem}.png"), sep_face_mask)
            cv2.imwrite(str(dirs["sep_lip_mask"] /
                        f"{video_path.stem}.png"), sep_lip_mask)
        else:
            images_dir = output_dir / "images" / video_path.stem
            audio_path = output_dir / "audios" / f"{video_path.stem}.wav"
            _, face_emb, _, _, _ = image_processor.preprocess(images_dir)
            torch.save(face_emb, str(
                dirs["face_emb"] / f"{video_path.stem}.pt"))
            audio_emb, _ = audio_processor.preprocess(audio_path)
            torch.save(audio_emb, str(
                dirs["audio_emb"] / f"{video_path.stem}.pt"))
    except Exception as e:
        logging.error(f"Failed to process video {video_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process videos for training.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing videos")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save results")

    args = parser.parse_args()

    video_paths = [item for item in sorted(args.input_dir.iterdir()) if item.is_file() and item.suffix == '.mp4']
    video_path_list = [video_paths[i] for i in range(len(video_paths)) if i % parallelism == rank]

    face_analysis_model_path = "pretrained_models/face_analysis"
    landmark_model_path = "pretrained_models/face_analysis/models/face_landmarker_v2_with_blendshapes.task"
    audio_separator_model_file = "pretrained_models/audio_separator/Kim_Vocal_2.onnx"
    wav2vec_model_path = 'pretrained_models/wav2vec/wav2vec2-base-960h'

    audio_processor = AudioProcessor(
        16000,
        25,
        wav2vec_model_path,
        False,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        os.path.join(args.output_dir, "vocals"),
    ) if step==2 else None

    image_processor = ImageProcessorForDataProcessing(
        face_analysis_model_path, landmark_model_path, step)

    for video_path in tqdm(video_path_list, desc="Processing videos"):
        process_single_video(video_path, args.output_dir, image_processor, audio_processor, step)


if __name__ == "__main__":
    main()
