import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import json

from talking_head.utils.video_utils import VideoProcessor
from talking_head.utils.audio_utils import AudioProcessor


def process_video(
    input_path,
    output_dir,
    video_processor,
    audio_processor,
):
    # prepare image embdeding
    face_emb, video_length = video_processor.get_face_embedding_from_video(input_path)
    if face_emb is None:
        print("Fail to extract face embedding")
        return None, None

    # prepare audio embedding
    audio_emb = audio_processor.get_audio_embedding_from_video(input_path, video_length)
    if audio_emb is None:
        print("Fail to extract audio embedding")
        return None, None

    face_emb_path = output_dir / "face_emb" / f"{input_path.stem}.pt"
    audio_emb_path = output_dir / "audio_emb" / f"{input_path.stem}.pt"
    torch.save(face_emb, face_emb_path)
    torch.save(audio_emb, audio_emb_path)

    return face_emb_path, audio_emb_path


def main():
    parser = argparse.ArgumentParser(description="Process videos for training.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing videos")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save results")
    parser.add_argument("--face_analysis_model_path", type=str, default="pretrained_models/face_analysis", help="Path to the face analysis model")
    parser.add_argument("--wav2vec_model_path", type=str, default="pretrained_models/wav2vec/wav2vec2-base-960h", help="Path to the wav2vec model")
    parser.add_argument("--audio_separator_model_file", type=str, default=None, help="Path to the audio separator model file")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio processing")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames for audio processing")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the processing on")

    args = parser.parse_args()

    # create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir_name in ["face_emb", "audio_emb", "vocal"]:
        subdir_path = output_dir / subdir_name
        subdir_path.mkdir(parents=True, exist_ok=True)

    video_processor = VideoProcessor(args.face_analysis_model_path)
    audio_processor = AudioProcessor(
        output_dir,
        args.wav2vec_model_path,
        args.audio_separator_model_file,
        sample_rate=args.sample_rate,
        num_frames=args.num_frames,
        device=args.device,
    )

    # process videos
    metadata_path = output_dir / "metadata.jsonl"
    processed_videos = []
    if metadata_path.exists():
        for line in metadata_path.read_text().splitlines():
            metadata = json.loads(line)
            processed_videos.append(Path(metadata["video"]))

    input_video_paths = [file for file in Path(args.input_dir).rglob('*.mp4')]
    for input_path in tqdm(input_video_paths, desc="Processing videos"):
        if not input_path.exists():
            print(f"Video path {input_path} does not exist")
            continue
        if input_path in processed_videos:
            print(f"Video {input_path} has already been processed")
            continue

        face_emb_path, audio_emb_path = process_video(input_path, output_dir, video_processor, audio_processor)
        if face_emb_path is not None and audio_emb_path is not None:
            processed_videos.append(input_path)
            # save metadata to jsonl
            metadata = {
                "video": str(input_path),
                "face_emb": str(face_emb_path),
                "audio_emb": str(audio_emb_path),
            }
            with open(metadata_path, "a") as f:
                f.write(json.dumps(metadata) + "\n")

    print(f"Saved {len(processed_videos)} embeddings to {metadata_path}")


if __name__ == "__main__":
    main()
