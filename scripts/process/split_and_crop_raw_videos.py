import os
import json
import argparse
from tqdm import tqdm
import ffmpeg
from concurrent.futures import ThreadPoolExecutor, as_completed


def split_and_crop_video(input_video_path, output_video_path, start_time, end_time, x, y, width, height, denorm=False):
    if denorm:
        probe = ffmpeg.probe(input_video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        video_width = int(video_info['width'])
        video_height = int(video_info['height'])
        x = round(x * video_width)
        y = round(y * video_height)
        width = round(width * video_width)
        height = round(height * video_height)
        c = min(height, width) // 2
        c_h = (2 * y + height) / 2
        c_w = (2 * x + width) / 2
        x, y, width, height = c_w - c, c_h - c, 2 * c, 2 * c

    input_stream = ffmpeg.input(input_video_path, ss=start_time, to=end_time)
    video = (
        input_stream
        .video
        .filter('crop', w=width, h=height, x=x, y=y)
    )
    audio = input_stream.audio

    output = ffmpeg.output(video, audio, output_video_path, vcodec='libx264', loglevel='error')
    ffmpeg.run(output, overwrite_output=True)

    return output_video_path


def split_video_to_1min_clips(input_video_path, output_video_path):
    input_stream = ffmpeg.input(input_video_path)
    output = ffmpeg.output(input_stream, output_video_path, c='copy', map='0', segment_time='00:01:00', loglevel='error', f='segment')
    ffmpeg.run(output, overwrite_output=True)

    return output_video_path


def split_video_to_30s_clips(input_video_path, output_video_path):
    input_stream = ffmpeg.input(input_video_path)
    output = ffmpeg.output(input_stream, output_video_path, c='copy', map='0', segment_time='00:00:30', loglevel='error', f='segment')
    ffmpeg.run(output, overwrite_output=True)

    return output_video_path


def split_and_crop_video_to_multiple_clips(input_video_path, output_video_paths, start_frames, end_frames, xs, ys, widths, heights):
    probe = ffmpeg.probe(input_video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    video_width = int(video_info['width'])
    video_height = int(video_info['height'])
    fps = eval(video_info['r_frame_rate'])

    for output_video_path, start_frame, end_frame, x, y, width, height in zip(output_video_paths, start_frames, end_frames, xs, ys, widths, heights):
        x = round(x * video_width)
        y = round(y * video_height)
        width = round(width * video_width)
        height = round(height * video_height)
        start_time = round(start_frame / fps, 5)
        end_time = round(end_frame / fps, 5)
        input_stream = ffmpeg.input(input_video_path, ss=start_time, to=end_time)
        video = (
            input_stream
            .video
            .filter('crop', w=width, h=height, x=x, y=y)
        )
        audio = input_stream.audio
        output = ffmpeg.output(video, audio, output_video_path, vcodec='libx264', loglevel='error')
        ffmpeg.run(output, overwrite_output=True)

    return input_video_path


def main(args):
    metadata_path = args.metadata_path
    input_video_root = args.input_video_root
    output_video_root = args.output_video_root
    os.makedirs(output_video_root, exist_ok=True)

    video_data = []
    if args.dataset == "celebvhq":
        with open(metadata_path) as f:
            data_dict = json.load(f)
        data = [
            (val['ytb_id'], key+".mp4", (val['duration']['start_sec'], val['duration']['end_sec']),
            [val['bbox']['top'], val['bbox']['bottom'], val['bbox']['left'], val['bbox']['right']])
            for key, val in data_dict['clips'].items()
        ]

        def expand(bbox, ratio):
            top, bottom = max(bbox[0] - ratio, 0), min(bbox[1] + ratio, 1)
            left, right = max(bbox[2] - ratio, 0), min(bbox[3] + ratio, 1)
            return top, bottom, left, right

        for ytb_id, save_vid_name, time, bbox in tqdm(data, desc="Preparing video data"):
            input_video_path = os.path.join(input_video_root, f"{ytb_id}.mp4")
            if not os.path.exists(input_video_path):
                print(f"Input video not found: {input_video_path}")
                continue
            else:
                output_video_path = os.path.join(output_video_root, save_vid_name)
                if os.path.exists(output_video_path):
                    print(f"Output video already exists: {output_video_path}")
                    continue

                top, bottom, left, right = expand(bbox, 0.02)
                start_sec, end_sec = time

                video_data.append((input_video_path, output_video_path, start_sec, end_sec, left, top, right-left, bottom-top, True))

    elif args.dataset == "vfhq":
        data = os.listdir(metadata_path)
        for clipname in tqdm(data, desc="Preparing video data"):
            clip_meta_file = os.path.join(metadata_path, clipname)
            clipname = os.path.splitext(clipname)[0]
            _, ytb_id, pid, clip_idx, frame_rlt = clipname.split('+')    
            input_video_path = os.path.join(input_video_root, f"{ytb_id}.mp4")
            if not os.path.exists(input_video_path):
                print(f"Input video not found: {input_video_path}")
                continue
            else:
                output_video_path = os.path.join(output_video_root, f'{clipname}.mp4')
                if os.path.exists(output_video_path):
                    print(f"Output video already exists: {output_video_path}")
                    continue

                # read the basic info
                with open(clip_meta_file, 'r') as f:
                    for line in f:
                        if line.startswith('FPS'):
                            clip_fps = float(line.strip().split(' ')[-1])
                        if line.startswith('CROP'):
                            clip_crop_bbox = line.strip().split(' ')[-4:]
                            left = int(clip_crop_bbox[0])
                            top = int(clip_crop_bbox[1])
                            right = int(clip_crop_bbox[2])
                            bottom = int(clip_crop_bbox[3])

                pid = int(pid.split('P')[1])
                clip_idx = int(clip_idx.split('C')[1])
                frame_start, frame_end = frame_rlt.replace('F', '').split('-')
                frame_start, frame_end = int(frame_start) + 1, int(frame_end) - 1
                start_sec = round(frame_start / float(clip_fps), 5)
                end_sec = round(frame_end / float(clip_fps), 5)

                video_data.append((input_video_path, output_video_path, start_sec, end_sec, left, top, right - left, bottom - top))

    elif args.dataset in ["hdtf", "talkinghead1kh-split"]:
        # Iterate over each mp4 file in the input directory
        for filename in os.listdir(input_video_root):
            if filename.endswith(".mp4"):
                input_video_path = os.path.join(input_video_root, filename)
                base_name = filename.replace(".mp4", "")
                output_video_path = os.path.join(output_video_root, f"{base_name}_%04d.mp4")
                if os.path.exists(output_video_path):
                    print(f"Already processed: {output_video_path}")
                    continue
                video_data.append((input_video_path, output_video_path))

    elif args.dataset == "talkinghead1kh":
        clip_info = []
        with open(metadata_path) as metadata:
            for line in metadata:
                clip_info.append(line.strip())
        print(f"Total clips: {len(clip_info)}")

        data = {}
        for clip_params in tqdm(clip_info, desc="Creating data dict"):
            video_name, H, W, S, E, L, T, R, B = clip_params.strip().split(',')
            H, W, S, E, L, T, R, B = int(H), int(W), int(S), int(E), int(L), int(T), int(R), int(B)
            output_video_path = os.path.join(output_video_root, f"{video_name}_S{S}_E{E}_L{L}_T{T}_R{R}_B{B}.mp4")
            if os.path.exists(output_video_path):
                # print(f"Already processed: {output_video_path}")
                continue
            input_video_path = os.path.join(input_video_root, f"{video_name}.mp4")
            if not os.path.exists(input_video_path):
                # print(f"Input video not found: {input_video_path}")
                continue
        
            if video_name not in data:
                data[video_name] = {
                    "input_video_path": input_video_path,
                    "output_video_path": [],
                    "start_frame": [],
                    "end_frame": [],
                    "x": [],
                    "y": [],
                    "width": [],
                    "height": [],
                }
            data[video_name]["output_video_path"].append(output_video_path)
            data[video_name]["start_frame"].append(int(S) + 1)
            data[video_name]["end_frame"].append(int(E) - 1)
            data[video_name]["x"].append(L / W)
            data[video_name]["y"].append(T / H)
            data[video_name]["width"].append((R - L) / W)
            data[video_name]["height"].append((B - T) / H)

        for v in tqdm(data.values(), desc="Preparing video data"):
            video_data.append((v["input_video_path"], v["output_video_path"], v["start_frame"], v["end_frame"], v["x"], v["y"], v["width"], v["height"]))

    else:
        raise ValueError("Unknown metadata path")

    with ThreadPoolExecutor() as executor:
        if args.dataset == "talkinghead1kh-split":
            futures = [executor.submit(split_video_to_1min_clips, *args) for args in video_data]
        elif args.dataset == "hdtf":
            futures = [executor.submit(split_video_to_30s_clips, *args) for args in video_data]
        elif args.dataset == "talkinghead1kh":
            futures = [executor.submit(split_and_crop_video_to_multiple_clips, *args) for args in video_data]
        else:
            futures = [executor.submit(split_and_crop_video, *args) for args in video_data]
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Processed video: {result}")
                except Exception as e:
                    print(f'Error processing video: {e}')
                pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["hdtf", "celebvhq", "vfhq", "talkinghead1kh-split", "talkinghead1kh"])
    parser.add_argument('--metadata_path', type=str, default=None)
    parser.add_argument('--input_video_root', type=str)
    parser.add_argument('--output_video_root', type=str)
    args = parser.parse_args()

    main(args)
