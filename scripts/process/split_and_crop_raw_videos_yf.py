import os
import argparse
import ffmpeg
from moviepy.video.io.VideoFileClip import VideoFileClip

def trim_video(input_path, temp_output_path, start_trim, end_trim):
    with VideoFileClip(input_path) as video:
        duration = video.duration
        if duration > start_trim + end_trim:
            # Create the subclip by excluding the first 'start_trim' seconds and last 'end_trim' seconds
            trimmed_video = video.subclip(start_trim, duration - end_trim)
            trimmed_video.write_videofile(temp_output_path, codec="libx264")
        else:
            print(f"Video {input_path} is too short to trim")
            return None
    return temp_output_path

def split_video_to_30s_clips(input_video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input_stream = ffmpeg.input(input_video_path)
    output_template = os.path.join(output_folder, 'output_%03d.mp4')
    output = ffmpeg.output(input_stream, output_template, c='copy', map='0', segment_time='30', f='segment', reset_timestamps='1', loglevel='error')
    ffmpeg.run(output, overwrite_output=True)

def process_videos(input_folder, output_folder, temp_folder, start_trim, end_trim):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            temp_output_path = os.path.join(temp_folder, filename)
            output_subfolder = os.path.join(output_folder, os.path.splitext(filename)[0])
            
            # Check if the video has already been processed
            if os.path.exists(output_subfolder) and any(fname.endswith('.mp4') for fname in os.listdir(output_subfolder)):
                print(f"Skipping {input_path} as it has already been processed")
                continue
            
            print(f"Processing {input_path}")
            trimmed_video_path = trim_video(input_path, temp_output_path, start_trim, end_trim)
            if trimmed_video_path:
                split_video_to_30s_clips(trimmed_video_path, output_subfolder)
                print(f"Saved split videos to {output_subfolder}")
            else:
                print(f"Skipping {input_path} due to insufficient length")

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_video_root', type=str, required=True)
    parser.add_argument('--output_video_root', type=str, required=True)
    parser.add_argument('--temp_video_root', type=str, required=True)
    parser.add_argument('--start_trim', type=int, default=12, help="Number of seconds to trim from the start of each video")
    parser.add_argument('--end_trim', type=int, default=8, help="Number of seconds to trim from the end of each video")
    args = parser.parse_args()

    process_videos(args.input_video_root, args.output_video_root, args.temp_video_root, args.start_trim, args.end_trim)
