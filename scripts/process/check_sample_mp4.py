import argparse
import os
import random
import shutil


def sample_and_copy_mp4_files(src_dir, dest_dir, num_samples):
    os.makedirs(dest_dir, exist_ok=True)

    # List all mp4 files in the source directory
    mp4_files = [f for f in os.listdir(src_dir) if f.endswith('.mp4')]

    # Randomly sample the specified number of mp4 files
    sampled_files = random.sample(mp4_files, num_samples)
    
    # Copy each sampled file to the destination directory
    for file in sampled_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))
        
    print(f"Copied {num_samples} mp4 files to {dest_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--dest_dir', type=str)
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()

    sample_and_copy_mp4_files(args.src_dir, args.dest_dir, args.num_samples)
