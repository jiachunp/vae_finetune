import os
import json
import subprocess


def download(video_path, ytb_id):
    """
    ytb_id: youtube_id
    save_folder: save video folder
    """
    if not os.path.exists(video_path):
        down_video = [
            "youtube-dl",
            '-f', "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
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


def load_data(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)

    for key, val in data_dict['clips'].items():
        save_name = key+".mp4"
        ytb_id = val['ytb_id']
        time = val['duration']['start_sec'], val['duration']['end_sec']

        bbox = [val['bbox']['top'], val['bbox']['bottom'],
                val['bbox']['left'], val['bbox']['right']]
        yield ytb_id, save_name, time, bbox


if __name__ == '__main__':
    json_path = 'celebvhq_info.json'  # json file path
    raw_vid_root = './downloaded_celebvhq/raw/'  # download raw video path
    processed_vid_root = './downloaded_celebvhq/processed/'  # processed video path

    os.makedirs(raw_vid_root, exist_ok=True)
    os.makedirs(processed_vid_root, exist_ok=True)

    for vid_id, save_vid_name, time, bbox in load_data(json_path):
        raw_vid_path = os.path.join(raw_vid_root, vid_id + ".mp4")
        # Check if the video is already downloaded
        if os.path.exists(raw_vid_path):
            print(f"video {vid_id} already downloaded")
            continue
        download(raw_vid_path, vid_id)
