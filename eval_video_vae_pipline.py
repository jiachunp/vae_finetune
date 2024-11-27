import cv2
import os
import json
# import mmcv
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import os
from glob import glob
from pathlib import Path
from typing import Optional
from safetensors.torch import load_file as load_safetensors
import cv2
# import mmcv
import numpy as np
import torch
import random
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor 
from talking_head.models.vae.util import default, instantiate_from_config
import pdb
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import models
from scipy import linalg
from piq import psnr, ssim

import pandas as pd

def find_all_files(src, abs_path=True, suffixs=None):
    all_files = []
    if suffixs is not None and not isinstance(suffixs, list):
        suffixs = list(suffixs)

    for root, dirs, files in os.walk(src):
        for file in files:
            if suffixs is not None:
                for suffix_i in suffixs:
                    if file.endswith(suffix_i):
                        all_files.append(os.path.join(root, file))
            else:
                all_files.append(os.path.join(root, file))

    if not abs_path:
        all_files = [os.path.relpath(item, src) for item in all_files]
    else:
        all_files = [os.path.abspath(item) for item in all_files]

    return all_files

class VideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        metadata_paths,
        num_frames=16,
        sample_stride=4,
        is_image=False,
        height=512,
        width=512,
    ):
        self.data_dir = data_dir
        self.metadata = []
        for line in Path(metadata_paths).read_text().splitlines():
            self.metadata.append(json.loads(line))
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.transform = transforms.Compose(
            [
                transforms.Resize((height, width), antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.is_image = is_image

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_info = self.metadata[idx]
        path = image_info["video"].split("/")[-1]
        path = path.split('.')[0]
        image_folder = os.path.join(self.data_dir, path)
        v_name = path
        
        # image_info = self.metadata[idx]
        # path = image_info["video"].split("/")[-2]
        # subpath = image_info["video"].split("/")[-1]
        # subpath = path.split('.')[0]
        # image_folder = os.path.join(self.data_dir, path+"_dwpose")
        # image_folder = os.path.join(image_folder, path)

        try:
            # List all images in the folder
            all_images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg'))]

            # Ensure there are enough images to sample
            if len(all_images) < self.num_frames:
                raise ValueError(f"Not enough images in folder {image_folder} to sample {self.num_frames} images.")
            
            # Randomly sample `num_images` from the folder
            selected_images = random.sample(all_images, self.num_frames)

            image_list = []
            [image_list.append(Image.open(img)) for img in selected_images]

            image_list = np.array(image_list)

        except Exception:
            print_str = f"Warning: missing video file {image_folder}."
            print_str += " Resampling another video."
            print(print_str)
            return self.__getitem__(np.random.choice(self.__len__()))
        
        pixel_values = torch.from_numpy(image_list).permute(0, 3, 1, 2).contiguous() 
        pixel_values = pixel_values / 255.
        pixel_values = self.transform(pixel_values)
        # sample = dict(pixel_values=pixel_values)
        sample = pixel_values
        
        del all_images

        return sample, v_name


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4)

    return x

def normalize_features(features):
    mean = torch.mean(features, dim=0)
    std = torch.std(features, dim=0)
    return (features - mean) / (std + 1e-8)

from scripts.video_metric.fid import load_inception_v3_pretrained, get_fid_score
from scipy.linalg import sqrtm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate the Frechet Distance between two multivariate Gaussians.
    
    Args:
        mu1 (np.ndarray): Mean vector of the first Gaussian distribution.
        sigma1 (np.ndarray): Covariance matrix of the first Gaussian distribution.
        mu2 (np.ndarray): Mean vector of the second Gaussian distribution.
        sigma2 (np.ndarray): Covariance matrix of the second Gaussian distribution.
        eps (float): Small epsilon value added for numerical stability.

    Returns:
        float: The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors must have the same length."
    assert sigma1.shape == sigma2.shape, "Covariance matrices must have the same dimensions."

    diff = mu1 - mu2
    try:
        # Attempting to calculate the sqrt of sigma product
        covmean = sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
        
        # Check if the result is close to numerical error that might give slight imaginary components
        if np.iscomplexobj(covmean):
            if not np.isfinite(covmean).all():
                raise ValueError("Non-finite values found in square root of product matrix.")
            covmean = covmean.real  # Discard imaginary parts that should not exist

    except ValueError as e:
        # Handling potential numerical issues with the sqrtm calculation by adding epsilon to diagonal
        print(f"Numerical stability issue encountered: {e}. Adding epsilon to diagonals.")
        covmean = sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
        covmean = np.real(covmean)  # Ensure any residual complex part is discarded

    fd = (np.sum(diff**2) + 
          np.trace(sigma1) + 
          np.trace(sigma2) - 
          2 * np.trace(covmean))

    return fd

def calculate_fid(features1, features2):
    """
    Compute the Fréchet Inception Distance (FID) between two sets of feature vectors.
    
    Args:
        features1 (np.ndarray): Feature matrix for the first set of images.
        features2 (np.ndarray): Feature matrix for the second set of images.

    Returns:
        float: The calculated FID score.
    """
    features1 = (features1 - features1.min()) / (features1.max() - features1.min())
    features2 = (features2 - features2.min()) / (features2.max() - features2.min())
 
    m1, s1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    m2, s2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


 

def calculate_fvd(real_videos, fake_videos, device, method='styleganv'):
    if method == 'styleganv':
        from scripts.video_metric.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from scripts.video_metric.fvd.videogpt.fvd import load_i3d_pretrained
        from scripts.video_metric.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from scripts.video_metric.fvd.videogpt.fvd import frechet_distance

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    print('assert real_videos.shape == fake_videos.shape', real_videos.shape, fake_videos.shape)

    assert real_videos.shape == fake_videos.shape

    i3d = load_i3d_pretrained(device=device)

    inception_v3 = load_inception_v3_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    real_videos_fvd = trans(real_videos) # BTCHW -> BCTHW 
    fake_videos_fvd = trans(fake_videos)

    fvd_results = {}
    fake_embeddings = []
    real_recon_embeddings = []

    fid_list = []
    real_feat_list = []
    fake_feat_list = []

    #print('real_videos_fvd[0]', real_videos_fvd[0]) # real_videos_fvd[0] torch.Size([3, 32, 224, 224])
    real_feat = inception_v3(real_videos_fvd[0].permute(1, 0, 2, 3).to(device))
    fake_feat = inception_v3(fake_videos_fvd[0].permute(1, 0, 2, 3).to(device))
    #print('real_feat', real_feat.shape) #torch.Size([40, 2048])  
    #print('fake_feat', fake_feat.shape) #torch.Size([40, 2048])  

    # Normalize features to reduce numerical instability
    # real_feat = normalize_features(real_feat)
    # fake_feat = normalize_features(fake_feat)
    # print('real_feat', real_feat.shape)
    # print('fake_feat', fake_feat.shape)
    #fid = get_fid_score(real_feat.shape[-1], real_feat, fake_feat)
 
    fid = calculate_fid(real_feat.detach().cpu().numpy(), fake_feat.detach().cpu().numpy())
    print('fid', fid)
    # for calculate FVD, each clip_timestamp must >= 10
    for clip_timestamp in tqdm(range(10, real_videos_fvd.shape[-3] + 1)):
        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        print('clip_timestamp', clip_timestamp)
        videos_clip_real = real_videos_fvd[:, :, : clip_timestamp]
        videos_clip_fake = fake_videos_fvd[:, :, : clip_timestamp] 

        # get FVD features
        real = get_fvd_feats(videos_clip_real, i3d=i3d, device=device)
        fake = get_fvd_feats(videos_clip_fake, i3d=i3d, device=device)
        # print('real', real.shape)
        fake_embeddings.append(real)
        real_recon_embeddings.append(fake)
        

    real_recon_embeddings = torch.cat(real_recon_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)

    fvd = frechet_distance(real_recon_embeddings.clone(), fake_embeddings.clone())
    fvd_star = frechet_distance(real_recon_embeddings.clone(), fake_embeddings.clone())
    return fvd, fvd_star, fid



def get_txt_from_image(image_name):
    directory, name = os.path.split(image_name)
    file_name = name.split('.')[0] + '.txt'
    txt_path = os.path.join(directory, file_name)
    with open(txt_path, 'r') as f:
        txt = f.readline()
    print(f"txt:{txt}")
    return txt


def add_text_to_frame(frame, text, font_scale=0.6, font_thickness=1, text_color=(255, 255, 255)):
    # 设置字体和文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 获取文本大小
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # 计算文本的位置
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 10  # 偏移量，可以根据需要进行调整
    # 检查文本是否超出图像宽度，需要进行换行
    if text_size[0] > frame.shape[1]:
        # 按照最大宽度计算文本行数
        max_text_width = frame.shape[1]
        # max_chars_per_line = max_text_width // (text_size[0] / len(text))
        max_chars_per_line = len(text) * max_text_width // text_size[0]
        num_lines = len(text) // max_chars_per_line + 1
        chars_per_line = len(text) // num_lines

        # 计算换行后的文本
        lines = []
        for i in range(num_lines):
            if i == num_lines - 1:
                lines.append(text[i * chars_per_line:])
            else:
                lines.append(text[i * chars_per_line: (i + 1) * chars_per_line])

        # 计算每行文本的位置并绘制
        for i, line in enumerate(lines):
            line_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            line_x = (frame.shape[1] - line_size[0]) // 2
            line_y = text_y - (text_size[1] + 5) * (num_lines - i - 1)
            cv2.putText(frame, line, (line_x, line_y), font, font_scale, text_color, font_thickness)
    else:
        # 绘制单行文本
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    return frame


def find_all_files(src, abs_path=True, suffixs=None):
    all_files = []
    if suffixs is not None and not isinstance(suffixs, list):
        suffixs = list(suffixs)

    for root, dirs, files in os.walk(src):
        for file in files:
            if suffixs is not None:
                for suffix_i in suffixs:
                    if file.endswith(suffix_i):
                        all_files.append(os.path.join(root, file))
            else:
                all_files.append(os.path.join(root, file))

    if not abs_path:
        all_files = [os.path.relpath(item, src) for item in all_files]
    else:
        all_files = [os.path.abspath(item) for item in all_files]

    return all_files


class CenterCropWide(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if isinstance(img, list):
            scale = min(img[0].size[0] / self.size[0], img[0].size[1] / self.size[1])
            img = [u.resize((round(u.width // scale), round(u.height // scale)), resample=Image.BOX) for u in img]

            # center crop
            x1 = (img[0].width - self.size[0]) // 2
            y1 = (img[0].height - self.size[1]) // 2
            img = [u.crop((x1, y1, x1 + self.size[0], y1 + self.size[1])) for u in img]
            return img
        else:
            scale = min(img.size[0] / self.size[0], img.size[1] / self.size[1])
            img = img.resize((round(img.width // scale), round(img.height // scale)), resample=Image.BOX)
            x1 = (img.width - self.size[0]) // 2
            y1 = (img.height - self.size[1]) // 2
            img = img.crop((x1, y1, x1 + self.size[0], y1 + self.size[1]))
            return img


def sliding_window_sampling(video_imgs, window_size=17, overlap_frame=0):
    sampled_frames = []
    num_frames = len(video_imgs)

    # 遍历视频帧序列
    read_window_size = window_size - overlap_frame
    for i in range(overlap_frame, num_frames - read_window_size + 1, read_window_size):
        # print('i', i)
        # 从当前位置开始提取长度为 window_size 的子序列

        if overlap_frame>0:
            if i == overlap_frame:
                # np.concatenate([x1,x2])
                window = video_imgs[i:i + read_window_size]
                window = torch.concatenate([video_imgs[:overlap_frame], window])
                sampled_frames.append(window)
                front_frame = video_imgs[:overlap_frame]
                # if overlap_frame==1:
                #     front_frame = np.expand_dims(front_frame, axis=0)
            else:
                window = video_imgs[i:i + read_window_size]
                window = torch.concatenate([front_frame, window])
                # window = np.concatenate([front_frame, window])
                sampled_frames.append(window)
                front_frame = window[-overlap_frame:]
        else:
            window = video_imgs[i:i + read_window_size]
            sampled_frames.append(window)


        print('i:{}, window:{}'.format(i, window.shape))
 
    return sampled_frames, i + read_window_size


def imdenormalize(img, mean, std, to_bgr=False):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img

def sample(
        input_path: str = "/mnt/workspace/ai-story/public/datasets/k600/demo_test",
        metadata_paths: str = "/mnt/workspace/ai-story/public/datasets/k600/demo_test",
        # Can either be image file or folder with image files
        # input_path: str = "pics_imageonly/",  # Can either be image file or folder with image files
        # input_path: str = "/TrainData/ai-story/dawei.liu/data/my_pic/",  # Can either be image file or folder with image files
        input_txt: str = "Rocket launching",
        num_frames: Optional[int] = None,
        save_eval_file='eval_vae.log',
        overlap_frame=0, 
        version: str = "svd",
        fps_id: int = 8,
        sample_frame=16,
        
        seed: int = 23,
        decoding_t: int = 8,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
        device: str = "cuda",
        output_folder: Optional[str] = None,
        ckpt_path='',
        config_file='',
        is_vis=False,
        seg=10, 
        short_size=512
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 20) 
        model_config = config_file   
    else:
        raise ValueError(f"Version {version} does not exist.")

    torch.manual_seed(seed)


    # center_crop = CenterCropWide((512, 512))
    model, filter = load_model(
        ckpt_path,
        model_config,
        device,
        num_frames, 
    )
    model.eval()
    device = torch.device("cuda")
    model.encoder.to(device)
    model.decoder.to(device) 
    size = 512
 
    transform = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True),
            transforms.CenterCrop((size, size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = VideoDataset(data_dir=input_path, metadata_paths=metadata_paths, num_frames=sample_frame)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    fvds = []
    fvds_star = []
    fid_list = []
    psnr_list = []
    ssim_list = []
    with open(save_eval_file, 'w') as f:

    # 使用 DataLoader 迭代数据
        for batch_idx, item in tqdm(enumerate(data_loader)):
            video_imgs, v_name = item 
            print('video_imgs', video_imgs.shape)  # b, t, c, h, w
            t_frames = video_imgs.shape[1]
            v_name = v_name[0]
            

            all_samples = []
            z_mean_list = []
            z_var_list = []
            silding_window_frames, last_id = sliding_window_sampling(video_imgs[0], num_frames, overlap_frame)
            print('silding_window_frames', len(silding_window_frames))
            print('silding_window_frames[0].shape', silding_window_frames[0].shape)
            # video_imgs = video_imgs[:num_frames].to(device) # b,t,
            # print('video_imgs', video_imgs.shape)

            #batch = {"timesteps": num_frames, "num_video_frames": num_frames, "jpg": video_imgs}
            for silding_window_frame in silding_window_frames:
                with torch.no_grad():
                    # with torch.autocast(device):

                    #additional_decode_kwargs = {'timesteps': num_frames}
                    print('encode_video_imgs', silding_window_frame.shape)

                    # model.en_and_decode_n_samples_a_time = decoding_t
                    z, reg_log = model.encode(silding_window_frame.to(device), return_reg_log=True)
                    torch.cuda.empty_cache()
                    z_mean = z.mean()
                    z_var = z.var()
                    z_mean_list.append(z_mean.cpu().numpy())
                    z_var_list.append(z_var.cpu().numpy())
                    
                    batch_size = int(silding_window_frame.shape[0]/num_frames)

                    additional_decode_kwargs = {}
                    additional_decode_kwargs["timesteps"] = t_frames
                    samples = model.decode(z.to(device), **additional_decode_kwargs)

                    torch.cuda.empty_cache()
                    del z

                    # samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
                    #print('is_causal', type(is_causal), is_causal)
                    #print('samples', samples.shape)
                    
                    print('z_mean', z_mean)
                    print('z_var', z_var)
                    # print("samples.mean()", samples.mean())
                    # print("samples.max()", samples.max())
                    # print("samples.min()", samples.min())

                    all_samples.append(samples[overlap_frame:].cpu())
                    # if is_causal:
                    #     all_samples.append(samples[(3 + overlap_frame):].cpu())
                    # else:
                    #     all_samples.append(samples[overlap_frame:].cpu())
 
            gen_video_frames = torch.cat(all_samples, dim=0) 
            video_path = os.path.join(output_folder, "{}".format(v_name))
            os.makedirs(video_path, exist_ok=True) 

            vis_gen_video_frames = rearrange(gen_video_frames, "t c h w -> t h w c")
        
            fvd_video_imgs = video_imgs[:, :last_id]
            vis_fvd_video_imgs = rearrange(fvd_video_imgs[0], "t c h w -> t h w c") 

            
            if is_vis:
                for idx, frame in enumerate(vis_gen_video_frames): 
                    frame = torch.clamp((frame + 1.0) / 2.0, min=0.0, max=1.0)
                    frame = cv2.cvtColor((frame*255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(video_path, "{}.png".format(idx)), frame)

                # for idx, frame in enumerate(vis_gen_video_frames):
                #     # frame = torch.clamp((frame + 1.0) / 2.0, min=0.0, max=1.0)
                #     frame = cv2.cvtColor((frame*255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(os.path.join(video_path, "fvd_video_imgs_{}.png".format(idx)), frame)

                for idx, frame in enumerate(vis_fvd_video_imgs):
                    frame = torch.clamp((frame + 1.0) / 2.0, min=0.0, max=1.0)
                    frame = cv2.cvtColor((frame*255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(video_path, "gth_video_imgs_{}.png".format(idx)), frame)
 
            try:

                fvd, fvd_star, fid  = calculate_fvd(fvd_video_imgs, gen_video_frames.unsqueeze(0), device=torch.device('cpu'), method='videogpt')
                print('v_name, fvd', v_name, fvd)
                print('v_name, fvd_star', v_name, fvd_star)
                print('fid', fid)

                fvd_video_imgs = torch.clamp((fvd_video_imgs + 1.0) / 2.0, min=0.0, max=1.0)
                gen_video_frames = torch.clamp((gen_video_frames + 1.0) / 2.0, min=0.0, max=1.0)
                psnr_value = psnr(fvd_video_imgs.squeeze(0), gen_video_frames, data_range=1.0)
                ssim_value = ssim(fvd_video_imgs.squeeze(0), gen_video_frames, data_range=1.0)
                print('psnr_value', psnr_value) 
                print('ssim_value', ssim_value)

  
                fvds.append(fvd)
                fid_list.append(fid)
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                out_line = "vid:{}, fvd:{}, fid:{}, psnr:{}, ssim:{}, z_mean:{}, z_var:{}".format(v_name, fvd, fid, psnr_value, ssim_value,z_mean,z_var)
                f.write(out_line + '\n')
            except Exception as error:
                print(v_name, error)

        fvd_mean = np.mean(fvds)
        fvd_std = np.std(fvds)

        fvd_star_mean = np.mean(fvds_star)
        fvd_star_std = np.std(fvds_star)

        fid_mean = np.mean(fid_list)
        fid_std = np.std(fid_list)
 
        psnr_mean = np.mean(psnr_list)
        psnr_std = np.std(psnr_list)

        ssim_mean = np.mean(ssim_list)
        ssim_std = np.std(ssim_list)
         
        z_mean_mean = np.mean(z_mean_list)
        z_var_mean = np.mean(z_var_list)
                             

        print(f" FID {fid_mean:.2f} +/- {fid_std:.2f}, FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, PSNR {psnr_mean:.2f} +/- {psnr_std:.2f}, SSIM {ssim_mean:.2f} +/- {ssim_std:.2f}, z_mean {z_mean_mean:.2f}, z_var {z_var_mean:.2f}")
        f.write(f" FID {fid_mean:.2f} +/- {fid_std:.2f}, FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, PSNR {psnr_mean:.2f} +/- {psnr_std:.2f}, SSIM {ssim_mean:.2f} +/- {ssim_std:.2f}, z_mean {z_mean_mean:.2f}, z_var {z_var_mean:.2f}" + '\n')
        f.close()

        # print('result', result)
        # with open("{}.json".format(v_name), "w") as json_file:
        #     json.dump(result, json_file)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        elif key == "txt":
            batch[key] = [value_dict["txt"]]
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    batch_uc["txt"] = ""
    return batch, batch_uc


def model_load_ckpt(model, path):
    # TODO: how to load ema weights?
    if path.endswith("ckpt") or path.endswith("pt"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
    elif path.endswith("safetensors"):
        sd = load_safetensors(path)
    else:
        raise NotImplementedError(f"Unknown checkpoint format: {path}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
    )
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
    if len(unexpected) > 0:
        print(f"Unexpected Keys: {unexpected}")

    return model


def load_model(
        ckpt_path: str,
        config: str,
        device: str,
        num_frames: int, 
):
    config = OmegaConf.load(config) 
    # config.model.params.encoder_config.video_frame_num = num_frames
    # config.model.params.decoder_config.video_frame_num = num_frames

    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()
 
    if len(ckpt_path)>1:
        model = model_load_ckpt(model, path=ckpt_path)
 
    return model, filter


if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='/mnt/data/ai-story/zhuoqun.luo/model_zoo/svd.safetensors')
    parser.add_argument("--output_folder", type=str, default='vae_eval_out/svd_vae_baseline')
    parser.add_argument("--save_eval_file", type=str, default='vae_eval_out/svd_vae_baseline_eval.log')
    parser.add_argument("--config_file", type=str, default='/mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/svd_vae_infer.yaml')
    parser.add_argument("--sample_frame", type=int, default=32) 
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--seg", type=int, default=1)
    parser.add_argument("--overlap_frame", type=int, default=0)
    parser.add_argument("--is_vis", type=bool, default=False) 

    
    
    parser.add_argument("--input_path", type=str, default="/mnt/workspace/ai-story/zhuoqun.luo/dataset/mini_vae_test/demo")
    parser.add_argument("--metadata_paths", type=str, default="/mnt/workspace/ai-story/zhuoqun.luo/dataset/mini_vae_test/demo")
    # parser.add_argument("--short_size", type=str)

    args = parser.parse_args()



    # '/mnt/workspace/ai-story/zhuoqun.luo/workspace/exp_2dplus1d_v3/ti2v/configs/video_vae_config/v4_8z.yaml'

    sample(input_path=args.input_path,
           metadata_paths=args.metadata_paths,
           # Can either be image file or folder with image files
           # input_path: str = "pics_imageonly/",  # Can either be image file or folder with image files
           # input_path: str = "/TrainData/ai-story/dawei.liu/data/my_pic/",  # Can either be image file or folder with image files
           input_txt="Rocket launching",
           num_frames=args.num_frames,
           overlap_frame=args.overlap_frame,
           save_eval_file=args.save_eval_file, 
           version="svd", 
           sample_frame=args.sample_frame,
           fps_id=25,
           seed=23,
           seg=args.seg,
           decoding_t=8,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
           device="cuda",
           is_vis=args.is_vis,
           output_folder=args.output_folder,
           ckpt_path=args.ckpt_path,
           config_file=args.config_file)

#  CUDA_VISIBLE_DEVICES=0,1 python scripts/sampling/eval_video_vae_pipline.py 
# 
# /mnt/workspace/ai-story/zhuoqun.luo/workspace/exp_2dplus1d_v3/ti2v/logs/2024-03-30T10-14-19_video_vae_config-v4_8z/checkpoints/epoch=000002-v1.ckpt  outputs/ti2v_sample/vae_ori_res/ /mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/v4_8z.yaml 1 9  >vae_infer_9f_v4_8z.log  2>&1
#  CUDA_VISIBLE_DEVICES=0,1 python scripts/sampling/vae_infer.py /mnt/workspace/ai-story/zhuoqun.luo/workspace/exp_2dplus1d_v3/ti2v/logs/2024-03-30T10-14-19_video_vae_config-v4_8z/checkpoints/epoch=000002-v1.ckpt  outputs/ti2v_sample/vae_ori_res_5f_demo/ /mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/v4_8z_5f.yaml  1 5 >vae_infer_5f_v4_8z_ori_res.log  2>&1

#  CUDA_VISIBLE_DEVICES=0,1 python scripts/sampling/vae_infer.py /mnt/workspace/ai-story/zhuoqun.luo/workspace/exp_2dplus1d_v3/ti2v/logs/2024-03-30T10-14-19_video_vae_config-v4_8z/checkpoints/epoch=000002.ckpt  outputs/ti2v_sample/vae_ori_res_17f_demo_v1_320/ /mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/v4_8z.yaml  1 17 320


#  CUDA_VISIBLE_DEVICES=0,1 python scripts/sampling/vae_infer.py /mnt/workspace/ai-story/zhuoqun.luo/workspace/exp_2dplus1d_v3/ti2v/logs/2024-03-26T10-25-56_video_vae_config-debug_wlcb_vae_2dplus1d/checkpoints/epoch=000020.ckpt outputs/ti2v_sample/debug_wlcb_vae_2dplus1d_5f_demo_v1_ori_res/ /mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/debug_wlcb_vae_2dplus1d.yaml 0 17 1 512  > vae_infer_2d_1d_17f_sora.log  2>&1
#

# fvd 1175.1865234375
# fvd_star 4416.291015625