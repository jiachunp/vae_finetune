import os
import sys
import subprocess
from glob import glob
import concurrent.futures
import threading
from tqdm import tqdm
from pathlib import Path


def get_video_duration(file_path):
    """Use ffmpeg to get the video duration in seconds."""
    result = subprocess.run(['ffmpeg', '-i', file_path], stderr=subprocess.PIPE, text=True)
    for line in result.stderr.split('\n'):
        if "Duration" in line:
            duration = line.split('Duration: ')[1].split(',')[0]
            h, m, s = map(float, duration.split(':'))
            return int(h * 3600 + m * 60 + s)
    return 0


def main_parallel(folder_path):
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        sys.exit(1)

    print(f"Calculating total duration of videos in folder: {folder_path}")

    all_mp4_files = [str(file) for file in Path(folder_path).rglob('*.mp4')]
    files = list(all_mp4_files)
    print(f"Found {len(all_mp4_files)} mp4 files in the main directory.")

    total = len(files)
    print(f"Total files: {total}")

    # Shared variable for progress tracking
    progress_count = 0
    progress_lock = threading.Lock()

    def update_progress(duration):
        """Update the progress bar and count."""
        nonlocal progress_count
        with progress_lock:
            progress_count += 1
            percent = int((100 * progress_count) / total)
            bar = '#' * (percent // 2)
            sys.stdout.write(f"\r[{bar:<50}] {percent}%")
            sys.stdout.flush()
        return duration

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map get_video_duration to files and then apply update_progress
        durations = list(executor.map(lambda f: update_progress(get_video_duration(f)), files))

    total_sec = sum(durations)
    sys.stdout.write("\n")
    hours, remainder = divmod(total_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total Duration: {hours:02}:{minutes:02}:{seconds:02}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide a folder path as an argument.")
        sys.exit(1)
    main_parallel(sys.argv[1])
