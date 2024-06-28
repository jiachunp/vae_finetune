import os
import subprocess


def download(video_path, ytb_id):
    """
    ytb_id: youtube_id
    save_folder: save video folder
    """
    if not os.path.exists(video_path):
        down_video = [
            "youtube-dl",
            '-f', "bestvideo+bestaudio",
            '--skip-unavailable-fragments',
            '--merge-output-format', 'mp4',
            "https://www.youtube.com/watch?v=" + ytb_id, "--output",
            video_path, "--external-downloader", "aria2c",
            "--external-downloader-args", "-x 16 -k 1M",
        ]
        print(down_video)
        status = subprocess.call(down_video)
        if status != 0:
            print(f"video not found: {ytb_id}")


if __name__ == '__main__':
    video_path = "downloaded_vfhq"
    entries = os.listdir('vfhq_dataset')
    raw_vid_root = os.path.join(video_path, 'raw_videos')
    os.makedirs(raw_vid_root, exist_ok=True)
    for clipname in entries:
        _, videoid, pid, clip_idx, frame_rlt = clipname.split('+')
        # download youtube video
        raw_vid_path = os.path.join(raw_vid_root, videoid + ".mp4")
        # Check if the video is already downloaded
        if os.path.exists(raw_vid_path):
            print(f"video {videoid} already downloaded")
            continue
        download(raw_vid_path, videoid)
