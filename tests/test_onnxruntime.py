import os
from audio_separator.separator import Separator

audio_separator_model_file = "/mnt/data/longtaozheng/talking-head/pretrained_models/audio_separator/Kim_Vocal_2.onnx"
audio_separator_model_path = os.path.dirname(audio_separator_model_file)
audio_separator_model_name = os.path.basename(audio_separator_model_file)

audio_separator = Separator(
    output_dir="vocal",
    output_single_stem="vocal",
    model_file_dir=audio_separator_model_path,
)

audio_separator.load_model(audio_separator_model_name)
