import argparse
import multiprocessing as mp
import os
from functools import partial
from time import time as timer
import subprocess

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_list', type=str, required=True,
                    help='List of youtube video ids')
parser.add_argument('--output_dir', type=str, default='data/youtube_videos',
                    help='Location to download videos')
parser.add_argument('--num_workers', type=int, default=8,
                    help='How many multiprocessing workers?')
args = parser.parse_args()


def download_video(output_dir, video_id):
    r"""Download video."""
    video_path = '%s/%s.mp4' % (output_dir, video_id)
    if not os.path.isfile(video_path):
        try:
            # Download the highest quality mp4 stream.
            proxy_cmd = None
            if not os.path.exists(video_path):
                down_video = [
                    "youtube-dl",
                    '-f', "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
                    '--skip-unavailable-fragments',
                    '--merge-output-format', 'mp4',
                    "https://www.youtube.com/watch?v=" + video_id, "--output",
                    video_path,
                ]
                if proxy_cmd is not None:
                    down_video.insert(1, proxy_cmd)
                print(down_video)
                status = subprocess.call(down_video)
                if status != 0:
                    print(f"video not found: {video_id}")
        except Exception as e:
            print(e)
            print('Failed to download %s' % (video_id))
    else:
        print('File exists: %s' % (video_id))


if __name__ == '__main__':
    # Read list of videos.
    video_ids = []
    with open(args.input_list) as fin:
        for line in fin:
            video_ids.append(line.strip())

    # Create output folder.
    os.makedirs(args.output_dir, exist_ok=True)

    # Download videos.
    downloader = partial(download_video, args.output_dir)

    start = timer()
    pool_size = args.num_workers
    print('Using pool size of %d' % (pool_size))
    with mp.Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(downloader, video_ids), total=len(video_ids)))
    print('Elapsed time: %.2f' % (timer() - start))
