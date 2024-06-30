import os
import librosa
import math
import numpy as np
import torch
import soundfile as sf
from einops import rearrange
from audio_separator.separator import Separator
from transformers import Wav2Vec2FeatureExtractor

from talking_head.models.wav2vec import Wav2VecModel


class AudioProcessor:
    def __init__(
        self,
        output_dir,
        wav2vec_model_path,
        audio_separator_model_file,
        sample_rate,
        fps,
        num_frames,
        device,
    ):
        self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).to(device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=True)

        if audio_separator_model_file is None:
            self.audio_separator = None
        else:
            audio_separator_model_path = os.path.dirname(audio_separator_model_file)
            audio_separator_model_name = os.path.basename(audio_separator_model_file)
            self.audio_separator = Separator(
                output_dir=output_dir / "vocal",
                output_single_stem="vocal",
                model_file_dir=audio_separator_model_path,
            )
            self.audio_separator.load_model(audio_separator_model_name)
            assert self.audio_separator.model_instance is not None, "Fail to load audio separate model."

        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.fps = fps
        self.num_frames = num_frames
        self.device = device

    def get_audio_embedding_from_video(self, input_path):
        speech_array, sr = librosa.load(input_path.as_posix(), sr=self.sample_rate)

        if self.audio_separator is not None:
            # if mono, duplicate the mono channel
            if len(speech_array.shape) == 1:
                y = np.tile(speech_array, (2, 1)).T
            # save the audio as wav
            raw_audio_path = self.output_dir / "vocal" / f"{input_path.stem}-raw.wav"
            sf.write(raw_audio_path, speech_array, sr)
            # separate vocal from audio
            outputs = self.audio_separator.separate(raw_audio_path)
            print(f"outputs: {outputs}")
            if len(outputs) <= 0:
                print("Audio separate failed. Using raw audio.")
            else:
                pass
                # # resample the vocal to the desired sample rate
                # vocal_path = self.output_dir / "vocal" / f"{input_path.stem}-16k.wav"
                # y, sr = librosa.load(str(input_audio_file), sr=None)
                # y_resampled = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                # speech_array = y_resampled
        
        audio_feature = np.squeeze(self.wav2vec_feature_extractor(speech_array, sampling_rate=self.sample_rate).input_values)
        seq_len = math.ceil(len(audio_feature) / self.sample_rate * self.fps)
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        if seq_len % self.num_frames != 0:
            audio_feature = torch.nn.functional.pad(audio_feature, (0, (self.num_frames - seq_len % self.num_frames) * (self.sample_rate // self.fps)), 'constant', 0.0)
            seq_len += self.num_frames - seq_len % self.num_frames
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)

        if len(embeddings) == 0:
            print("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(
            embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()

        return audio_emb