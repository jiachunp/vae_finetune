# Talking Head

## Installation

```bash
conda create -n talking_head python=3.10 -y
conda activate talking_head
pip install -r requirements.txt
pip install -e .
apt-get install ffmpeg
pip install ffmpeg-python
```

## Download Datasets

### HDTF

Clone `HDTF_dataset` from [here](https://github.com/MRzzm/HDTF).

Run the script to download videos with audios from Youtube.

```bash
python scripts/download/download_hdtf.py --output_dir /mnt/data/public/dataset/talking_head/HDTF --num_workers 8
```

### CelebV-HQ

Run the script to download metadata and visual-audio dataset from Youtube.

```bash
wget https://raw.githubusercontent.com/CelebV-HQ/CelebV-HQ/main/celebvhq_info.json
python scripts/download/download_celebvhq.py
```

### VFHQ

Download and unzip `meta_info.zip` from [here](https://onedrive.live.com/?authkey=%21ALmyA3IelxWV2iw&id=AACA1803F11F470D%211000&cid=AACA1803F11F470D&parId=root&parQt=sharedby&o=OneUp).

Or from terminal:
```bash
curl 'https://6in4cq.bn.files.1drv.com/y4mioTW0R1Ql8SoTZycvl17GUA1sO6xCpaTCbVFViDSoPyJB19sP_DqvQMrTXlYB3PxQaYOj4HsliixWFmF2y2Mr6Cs7q_BVwC7ihP_VwsFVUznJOKSQG6uUBPs1XWmCEyPC-bF34AvNIH6ixEi1dGw3r3wkgbbi9ifNT0fCs79fVm7vdLaLAGpc78xhgdrF7EobEq7aT18XClTPstegyaIgg' --output meta_info.zip
unzip meta_info.zip -d vfhq_dataset
```

Run the script to download videos with audios from YouTube.

```bash
python scripts/download/download_vfhq.py
```

### TalkingHead-1KH

```bash
wget https://raw.githubusercontent.com/tcwang0509/TalkingHead-1KH/master/data_list.zip
unzip data_list.zip
python scripts/download/download_talkinghead1kh.py --input_list data_list/small_video_ids.txt --output_dir small/raw_videos
python scripts/download/download_talkinghead1kh.py --input_list data_list/train_video_ids.txt --output_dir train/raw_videos
python scripts/download/download_talkinghead1kh.py --input_list data_list/val_video_ids.txt --output_dir val/raw_videos
```

### Voxceleb

```bash
./scripts/download/hfd.sh ProgramComputer/voxceleb --local-dir /mnt/data/public/dataset/talking_head/raw_data/voxceleb --dataset --tool aria2c -x 4
```

Joining zipped files and unzip:

```bash
cd vox2
cat vox2_dev_mp4* > vox2_mp4.zip
unzip vox2_mp4.zip
```

## Post-Download Checks

```bash
find . -type f ! -name "*.mp4" ! -name "*.ytdl" ! -name "*.part"
```

### If video and audio are unmerged

Probably due to incorrect installation of ffmpeg when downloading.

```bash
bash scripts/download/merge_video_audio.sh /mnt/data/public/dataset/talking_head/talkinghead_1kh/raw_videos /mnt/data/public/dataset/talking_head/talkinghead_1kh/merged_videos
```

### Check whether the audio exists

```bash
ffprobe -v error -show_entries stream=index,codec_type -of default=noprint_wrappers=1:nokey=1 file.mp4
ffprobe -v error -show_entries stream=index,codec_type -of default=noprint_wrappers=1:nokey=1 -- --aqjaJyZLk.mp4  # when filename starts with '--'
```

The output should look like this:

```bash
0
video
1
audio
```

### Check Total Durations

Run the following command in the video folder (we round the durations to the nearest second):

```bash
python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/video_datasets/HDTF
python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/raw_data/talkinghead_1kh/raw_videos
python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/raw_data/VFHQ_with_audio/raw_videos
python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/raw_data/CelebV-HQ/raw

python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/voxceleb/vox2/dev/mp4
```

Length of raw videos before processing:

- HDTF: 14:18:58 (354 videos, 4.7G, ~30 fps, 1636135 frames)
- VFHQ: 729:13:04 (2503 videos, 1.2T, full version without audio 2.9T)
- CelebV-HQ: 615:32:52 (9290 videos, 576G)
- TalkingHead-1KH (train): 1289:19:28 (2576 videos, 1.7T)

- Voxceleb: 1114:52:35 (545598 videos, 714G)

## Data Processing

HDTF and VoxCeleb2 are already processed.

VFHQ:

```bash
python scripts/process/split_and_crop_raw_videos.py --dataset vfhq --metadata_path /mnt/data/public/dataset/talking_head/metadata/vfhq_dataset --input_video_root /mnt/data/public/dataset/talking_head/raw_data/VFHQ_with_audio/raw_videos --output_video_root /mnt/data/public/dataset/talking_head/video_datasets/VFHQ
```

CelebV-HQ:

```bash
python scripts/process/split_and_crop_raw_videos.py --dataset celebvhq --metadata_path /mnt/data/public/dataset/talking_head/metadata/celebvhq_info.json --input_video_root /mnt/data/public/dataset/talking_head/raw_data/CelebV-HQ/raw --output_video_root /mnt/data/public/dataset/talking_head/video_datasets/CelebV-HQ
```

TalkingHead-1KH:

```bash
# Split the videos into 1-min chunks.
python scripts/process/split_and_crop_raw_videos.py --dataset talkinghead1kh-split --input_video_root /mnt/data/public/dataset/talking_head/raw_data/talkinghead_1kh/merged_videos --output_video_root /mnt/data/public/dataset/talking_head/raw_data/talkinghead_1kh/1min_clips

# Extract the talking head clips.
python scripts/process/split_and_crop_raw_videos.py --dataset talkinghead1kh --metadata_path /mnt/data/public/dataset/talking_head/metadata/data_list/train_video_tubes.txt --input_video_root /mnt/data/public/dataset/talking_head/raw_data/talkinghead_1kh/1min_clips --output_video_root /mnt/data/public/dataset/talking_head/video_datasets/talkinghead_1kh
```

### After Processing

Check durations:

```bash
python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/video_datasets/HDTF
python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/video_datasets/VFHQ
python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/video_datasets/CelebV-HQ
python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/video_datasets/talkinghead_1kh
python scripts/download/calculate_durations.py /mnt/data/public/dataset/talking_head/raw_data/voxceleb
```

- HDTF: 14:18:58 (354 clips, 4.7G)
- VFHQ: 15:19:01 (6764 videos, 15G)
- CelebV-HQ: 46:09:48 (28252 clips, 23G)
- TalkingHead-1KH (train): 534:49:12 (445964 clips, 144G)

- Voxceleb: (, 714G)

Manually examination:

```bash
python scripts/process/check_sample_mp4.py --src_dir /mnt/data/public/dataset/talking_head/video_datasets/VFHQ --dest_dir /mnt/data/longtaozheng/talking_head/check_samples/vfhq --num_samples 50
python scripts/process/check_sample_mp4.py --src_dir /mnt/data/public/dataset/talking_head/video_datasets/CelebV-HQ --dest_dir /mnt/data/longtaozheng/talking_head/check_samples/celebvhq --num_samples 50
python scripts/process/check_sample_mp4.py --src_dir /mnt/data/public/dataset/talking_head/video_datasets/talkinghead_1kh --dest_dir /mnt/data/longtaozheng/talking_head/check_samples/talkinghead_1kh --num_samples 50
```

## Inference

### Video Generation

```bash
python scripts/inference/inference.py --source_image examples/reference_images/5.jpg --driving_audio examples/driving_audios/4.wav
```

Inference with Reference UNet with Denoising UNet weights loaded:

```bash
python scripts/inference/no_ref_inference_load_weight.py --source_image examples/reference_images/5.jpg --driving_audio examples/driving_audios/4.wav
```

Inference with Denoising UNet only:

```bash
python scripts/inference/no_ref_inference.py --source_image examples/reference_images/5.jpg --driving_audio examples/driving_audios/4.wav
```

## Training

### Data Preparation

Organize the video directory into the following structure:

```text
video_datasets/
|-- CelebV-HQ/
|   └-- [video files].mp4
|
|-- HDTF/
|   └-- [video files].mp4
|
|-- talkinghead_1kh/
|   └-- [video files].mp4
|
└-- VFHQ/
    └-- [video files].mp4
```

Process the videos with the following commands:

```bash
python scripts/data_preprocess --input_dir /mnt/data/public/dataset/talking_head/video_datasets/HDTF
```

Generate the metadata JSON files with the following commands:

```bash
python scripts/extract_meta_info_stage1.py -r path/to/dataset -n dataset_name
python scripts/extract_meta_info_stage2.py -r path/to/dataset -n dataset_name
```

Replace `path/to/dataset` with the path to the parent directory of `videos`, such as `dataset_name` in the example above. This will generate `dataset_name_stage1.json` and `dataset_name_stage2.json` in the `./data` directory.

### Training

Update the data meta path settings in the configuration YAML files, `configs/train/stage1.yaml` and `configs/train/stage2.yaml`:


```yaml
#stage1.yaml
data:
  meta_paths:
    - ./data/dataset_name_stage1.json

#stage2.yaml
data:
  meta_paths:
    - ./data/dataset_name_stage2.json
```

Start training with the following command:

```shell
accelerate launch -m \
  --config_file accelerate_config.yaml \
  --machine_rank 0 \
  --main_process_ip 0.0.0.0 \
  --main_process_port 20055 \
  --num_machines 1 \
  --num_processes 8 \
  scripts.train_stage1 --config ./configs/train/stage1.yaml
```

#### Accelerate Usage Explanation

The `accelerate launch` command is used to start the training process with distributed settings.

```shell
accelerate launch [arguments] {training_script} --{training_script-argument-1} --{training_script-argument-2} ...
```

**Arguments for Accelerate:**

- `-m, --module`: Interpret the launch script as a Python module.
- `--config_file`: Configuration file for Hugging Face Accelerate.
- `--machine_rank`: Rank of the current machine in a multi-node setup.
- `--main_process_ip`: IP address of the master node.
- `--main_process_port`: Port of the master node.
- `--num_machines`: Total number of nodes participating in the training.
- `--num_processes`: Total number of processes for training, matching the total number of GPUs across all machines.

**Arguments for Training:**

- `{training_script}`: The training script, such as `scripts.train_stage1` or `scripts.train_stage2`.
- `--{training_script-argument-1}`: Arguments specific to the training script. Our training scripts accept one argument, `--config`, to specify the training configuration file.

For multi-node training, you need to manually run the command with different `machine_rank` on each node separately.
