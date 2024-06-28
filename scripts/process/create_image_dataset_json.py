import os
import argparse
import torch
from decord import VideoReader
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
import cv2

from emo.utils.blazeface.blazeface import BlazeFace
from emo.utils.face_annotation import process_single_batch


def generate_training_json_for_single_video(
    video_path,
    save_dir,
    gpu_index,
    sample_n_frames=16,
    min_clip_length=30,
):
    try:
        video_reader = VideoReader(video_path)
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return []
    
    video_length = len(video_reader)
    if video_length < min_clip_length:
        return []

    results = []
    batch_idx = np.linspace(0, video_length - 1, sample_n_frames, dtype=int)
    # Ensure the new batch_idx has different values at each position
    while True:
        ref_batch_idx = np.random.choice(video_length, sample_n_frames, replace=False)
        if np.all(ref_batch_idx != batch_idx):
            break

    device = f"cuda:{gpu_index}"
    blazeface = BlazeFace(back_model=True).to(device)
    blazeface.load_weights("/mnt/data/longtaozheng/talking_head/emo/utils/blazeface/blazefaceback.pth")
    blazeface.load_anchors("/mnt/data/longtaozheng/talking_head/emo/utils/blazeface/anchorsback.npy")

    frames = video_reader.get_batch(batch_idx)
    annotations = process_single_batch(frames, blazeface)

    ref_frames = video_reader.get_batch(ref_batch_idx)

    video_name = os.path.basename(video_path).split(".")[0]
    os.makedirs(os.path.join(save_dir, video_name), exist_ok=True)

    for frame, ref_frame, idx, ref_idx, annotation in zip(frames.asnumpy(), ref_frames.asnumpy(), batch_idx, ref_batch_idx, annotations):
        if annotation["num_detected_faces"] == 1:
            # save images to disk
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            frame_path = os.path.join(save_dir, f"{video_name}/{idx}.jpg")
            ref_frame_path = os.path.join(save_dir, f"{video_name}/{ref_idx}.jpg")
            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, frame)
            if not os.path.exists(ref_frame_path):
                cv2.imwrite(ref_frame_path, ref_frame)

            results.append({
                "image": frame_path,
                "ref_image": ref_frame_path,
                "annotation": annotation,
            })

    return results


def generate_training_json(
    video_dir,
    save_dir,
    start_idx,
    end_idx,
    # sample_stride=4,
    sample_n_frames=16,
    # sample_stride_aug=False,
    # min_clip_length=30,
):
    print(f"Reading videos from {video_dir}")
    if start_idx is None and end_idx is None:
        filenames = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    elif start_idx is None:
        filenames = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])[:end_idx]
    elif end_idx is None:
        filenames = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])[start_idx:]
    else:
        filenames = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])[start_idx:end_idx]
    print(f"Processing {len(filenames)} videos")
    video_paths = [os.path.join(video_dir, f) for f in filenames]

    start_time = time.time()
    processed_count = 0
    results = []
    
    # for i, video_path in tqdm(enumerate(video_paths)):
    #     result = generate_training_json_for_single_video(video_path, save_dir, i % torch.cuda.device_count(), sample_n_frames)
    #     if len(result) > 0:
    #         results.extend(result)
    #     processed_count += 1
    #     elapsed_time = time.time() - start_time
    #     average_time_per_video = elapsed_time / processed_count
    #     remaining_videos = len(video_paths) - processed_count
    #     eta = average_time_per_video * remaining_videos
    #     print(f"Processed {processed_count}/{len(video_paths)}. ETA: {eta:.2f} seconds")
    num_workers = torch.cuda.device_count()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, video_path in tqdm(enumerate(video_paths), desc="Adding videos to processing queue"):
            future = executor.submit(generate_training_json_for_single_video, video_path, save_dir, i % torch.cuda.device_count(), sample_n_frames)
            futures.append(future)
        
        total_videos = len(futures)
        for future in as_completed(futures):
            result = future.result()
            if len(result) > 0:
                results.extend(result)
            processed_count += 1
            elapsed_time = time.time() - start_time
            average_time_per_video = elapsed_time / processed_count
            remaining_videos = total_videos - processed_count
            eta = average_time_per_video * remaining_videos
            print(f"Processed {processed_count}/{total_videos}. ETA: {eta:.2f} seconds")

    with open(os.path.join(save_dir, "metadata.json"), 'w') as jsonfile:
        json.dump(results, jsonfile)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, help='path of video dataset')
    parser.add_argument("--save_dir", type=str, help='path to save images')
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    np.random.seed(0)

    generate_training_json(args.video_dir, args.save_dir, args.start_idx, args.end_idx)


if __name__ == "__main__":
    main()
