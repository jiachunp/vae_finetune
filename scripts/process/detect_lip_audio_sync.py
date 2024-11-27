from os.path import dirname, join, basename, isfile
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from os.path import join, basename, dirname, isfile, abspath
import numpy as np
import librosa
import audio

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        print(audio_sequences.shape)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding
        
device = "cuda"
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True,device="cuda"):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path,map_location=device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])

    return model
logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

import os
import random
import numpy as np
import torch
import cv2
from glob import glob
from os.path import join, basename, dirname, isfile
import subprocess
import librosa

# 你需要根据你的需求调整这些参数
class hparams:
    img_size = 128
    fps = 25
    sample_rate = 16000
    n_mels = 80

syncnet_T = 15
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.extract_frames_and_audio()

    def extract_frames_and_audio(self):
        # 获取当前运行目录
        current_dir = abspath('.')
        # 定义 temp 目录路径
        temp_dir = join(current_dir, 'temp')
        # 创建 temp 目录
        os.makedirs(temp_dir, exist_ok=True)
        
        # 在 temp 目录下创建 frames 目录保存帧
        self.frames_dir = join(temp_dir, 'frames')
        # 在 temp 目录下保存音频文件
        self.audio_path = join(temp_dir, 'audio.wav')
        
        os.makedirs(self.frames_dir, exist_ok=True)

        # Extract frames
        command = f"ffmpeg -i {self.video_path} -vf 'fps={hparams.fps}' {self.frames_dir}/%06d.jpg"
        subprocess.call(command, shell=True)

        # Extract audio
        command = f"ffmpeg -i {self.video_path} -q:a 0 -map a {self.audio_path}"
        subprocess.call(command, shell=True)
        
        self.all_videos = [self.frames_dir]  # Assuming single video for this case


    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{:06d}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        vidname = self.all_videos[0]
        img_names = list(glob(join(vidname, '*.jpg')))
        #if len(img_names) <= 3 * syncnet_T:
        #    continue
        
        img_name = random.choice(img_names)
        wrong_img_name = random.choice(img_names)
        while wrong_img_name == img_name:
            wrong_img_name = random.choice(img_names)

        window_fnames = self.get_window(img_name)
        wrong_window_fnames = self.get_window(wrong_img_name)
        #if window_fnames is None or wrong_window_fnames is None:
        #    continue

        window = self.read_window(window_fnames)
        #if window is None:
        #    continue

        wrong_window = self.read_window(wrong_window_fnames)
        #if wrong_window is None:
        #    continue


        wavpath = self.audio_path
        print("test")
        wav = audio.load_wav(wavpath, hparams.sample_rate)
        orig_mel = audio.melspectrogram(wav).T
        #wav, sr = librosa.load(wavpath, sr=hparams.sample_rate)
        #orig_mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=hparams.n_mels).T
        #wav, sr = librosa.load(wavpath, sr=hparams.sample_rate)
        #orig_mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=hparams.n_mels).T
        print(orig_mel.shape)
        #orig_mel = audio.melspectrogram(wav).T

        #print(window)
        mel = self.crop_audio_window(orig_mel.copy(), img_name)
        #print(mel.shape)
        
        #indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
        #print(indiv_mels)
        print(window)
        window = self.prepare_window(window)
        y = window.copy()
        window[:, :, window.shape[2]//2:] = 0.

        wrong_window = self.prepare_window(wrong_window)
        x = np.concatenate([window, wrong_window], axis=0)

        x = torch.FloatTensor(x)
        mel = torch.FloatTensor(mel.T).unsqueeze(0)
        #indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
        y = torch.FloatTensor(y)
        return x, mel, y

# 示例使用
#dataset = Dataset('path_to_video.mp4')
#data = dataset[0]


if __name__ == "__main__":
    recon_loss = nn.L1Loss()
    syncnet_checkpoint_path = "/mnt/data/longtaozheng/talking-head/lipsync_expert.pth"

    audio_path = ""
    syncnet_T = 15
    model = load_checkpoint(syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    dataset = Dataset('/mnt/data/longtaozheng/talking-head/deprecated/check_samples/talkinghead_1kh/hasaLmOLJ8k_0001_S1536_E1689_L973_T0_R1277_B304.mp4')
    (x, mel, gt) = dataset[0]
    x = x.to(device)
    mel = mel.to(device)
    #indiv_mels = indiv_mels.to(device)
    gt = gt.to(device)
    print(f"the shape is {x.shape,mel.shape}")
    a, v = model(mel, x)

    loss = cosine_loss(a, v, gt)
    print(loss)
    #g = model(indiv_mels, x)
    #sync_loss = get_sync_loss(mel, g)
