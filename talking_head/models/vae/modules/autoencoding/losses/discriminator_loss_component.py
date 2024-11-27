from typing import Dict, Iterator, List, Optional, Tuple, Union
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from matplotlib import colormaps
from matplotlib import pyplot as plt
from torchvision.ops import roi_align

from ....util import default, instantiate_from_config
from ..lpips.loss.lpips import LPIPS
from ..lpips.model.model import weights_init
from ..lpips.vqperceptual import hinge_d_loss, vanilla_d_loss
 
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.3)


# 定义辅助函数和类
def extract_facial_components(image, enlarge_ratio=1.4): 
 
    image = (image + 1) / 2.0
    image_uint8 = (image * 255).astype(np.uint8)
   

    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB) if image_uint8.shape[2] == 3 else image_uint8  
    results = face_mesh.process(image_rgb)
     
    if not results.multi_face_landmarks:
        # 保存无法检测到人脸的图像以便进一步分析
        cv2.imwrite('debug_image.png', image_rgb)
        print("No face detected in the image.")
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    component_locations = {}

    # 定义左眼、右眼和嘴巴的索引
    left_eye_idx = [33, 133, 160, 159, 158, 157, 173, 263, 362, 387, 386, 385, 384, 398]
    right_eye_idx = [263, 362, 387, 386, 385, 384, 398, 33, 133, 160, 159, 158, 157, 173]
    mouth_idx = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 42, 183]

    # 提取左眼位置
    left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_idx])
    mean_left_eye = np.mean(left_eye, axis=0)
    half_len_left_eye = np.max(np.ptp(left_eye, axis=0)) / 2 * enlarge_ratio
    component_locations['left_eye'] = [mean_left_eye[0], mean_left_eye[1], half_len_left_eye]

    # 提取右眼位置
    right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_idx])
    mean_right_eye = np.mean(right_eye, axis=0)
    half_len_right_eye = np.max(np.ptp(right_eye, axis=0)) / 2 * enlarge_ratio
    component_locations['right_eye'] = [mean_right_eye[0], mean_right_eye[1], half_len_right_eye]

    # 提取嘴巴位置
    mouth = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_idx])
    mean_mouth = np.mean(mouth, axis=0)
    half_len_mouth = np.max(np.ptp(mouth, axis=0)) / 2 * enlarge_ratio
    component_locations['mouth'] = [mean_mouth[0], mean_mouth[1], half_len_mouth]

    return component_locations
 

def get_roi_regions(image_batch: torch.Tensor, component_locations_batch: List[Dict[str, List[float]]], output_size: int = 512) -> Dict[str, torch.Tensor]:
    face_ratio = int(output_size / 512)
    eye_out_size = 80 * face_ratio
    mouth_out_size = 120 * face_ratio

    rois_left_eye = []
    rois_right_eye = []
    rois_mouths = []

    for batch_index, component_locations in enumerate(component_locations_batch):
        for part, loc in component_locations.items():
            x, y, half_len = loc
            x1, y1, x2, y2 = int(x - half_len), int(y - half_len), int(x + half_len), int(y + half_len)
            if part == 'left_eye':
                rois_left_eye.append([batch_index, x1, y1, x2, y2])  # [batch_index, x1, y1, x2, y2]
            elif part == 'right_eye':
                rois_right_eye.append([batch_index, x1, y1, x2, y2])  # [batch_index, x1, y1, x2, y2]
            elif part == 'mouth':
                rois_mouths.append([batch_index, x1, y1, x2, y2])  # [batch_index, x1, y1, x2, y2]

    rois_left_eye = torch.tensor(rois_left_eye, dtype=torch.float32).to(image_batch.device)
    rois_right_eye = torch.tensor(rois_right_eye, dtype=torch.float32).to(image_batch.device)
    rois_mouths = torch.tensor(rois_mouths, dtype=torch.float32).to(image_batch.device)

    left_eyes = roi_align(image_batch, rois_left_eye, output_size=(eye_out_size, eye_out_size))
    right_eyes = roi_align(image_batch, rois_right_eye, output_size=(eye_out_size, eye_out_size))
    mouths = roi_align(image_batch, rois_mouths, output_size=(mouth_out_size, mouth_out_size))

    return {'left_eyes': left_eyes, 'right_eyes': right_eyes, 'mouths': mouths}


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2 if downsample else 1, padding=padding, bias=bias)
        self.activate = nn.LeakyReLU(0.2, inplace=True) if activate else None

    def forward(self, x):
        x = self.conv(x)
        if self.activate is not None:
            x = self.activate(x)
        return x

class FacialComponentDiscriminator(nn.Module):
    def __init__(self):
        super(FacialComponentDiscriminator, self).__init__()
        self.conv1 = ConvLayer(3, 64, 3, downsample=False)
        self.conv2 = ConvLayer(64, 128, 3, downsample=True)
        self.conv3 = ConvLayer(128, 128, 3, downsample=False)
        self.conv4 = ConvLayer(128, 256, 3, downsample=True)
        self.conv5 = ConvLayer(256, 256, 3, downsample=False)
        self.final_conv = ConvLayer(256, 1, 3, bias=True, activate=False)

    def forward(self, x, return_feats=False):
        feat = self.conv1(x)
        feat = self.conv3(self.conv2(feat))
        rlt_feats = []
        if return_feats:
            rlt_feats.append(feat.clone())
        feat = self.conv5(self.conv4(feat))
        if return_feats:
            rlt_feats.append(feat.clone())
        out = self.final_conv(feat)
        if return_feats:
            return out, rlt_feats
        else:
            return out, None

class FacialComponentLoss(nn.Module):
    def __init__(self, left_eye_discriminator, right_eye_discriminator, mouth_discriminator):
        super(FacialComponentLoss, self).__init__()
        self.left_eye_discriminator = left_eye_discriminator
        self.right_eye_discriminator = right_eye_discriminator
        self.mouth_discriminator = mouth_discriminator
        self.cri_component = nn.MSELoss() 

    def forward(self, pred_img, real_img, component_locations):
        pred_components = get_roi_regions(pred_img, component_locations)
        real_components = get_roi_regions(real_img, component_locations)

        loss = 0
        loss_dict = {}
 
        # Left eye
        fake_left_eye, _ = self.left_eye_discriminator(pred_components['left_eyes'])
        real_left_eye, _ = self.left_eye_discriminator(real_components['left_eyes'])
        l_g_gan_left_eye = self.cri_component(fake_left_eye, torch.ones_like(fake_left_eye)) + self.cri_component(real_left_eye, torch.zeros_like(real_left_eye))
        loss += l_g_gan_left_eye
        loss_dict[f'l_g_gan_left_eye'] = l_g_gan_left_eye

        # Right eye
        fake_right_eye, _ = self.right_eye_discriminator(pred_components['right_eyes'])
        real_right_eye, _ = self.right_eye_discriminator(real_components['right_eyes'])
        l_g_gan_right_eye = self.cri_component(fake_right_eye, torch.ones_like(fake_right_eye)) + self.cri_component(real_right_eye, torch.zeros_like(real_right_eye))
        loss += l_g_gan_right_eye
        loss_dict[f'l_g_gan_right_eye'] = l_g_gan_right_eye

        # Mouth
        fake_mouth, _ = self.mouth_discriminator(pred_components['mouths'])
        real_mouth, _ = self.mouth_discriminator(real_components['mouths'])
        l_g_gan_mouth = self.cri_component(fake_mouth, torch.ones_like(fake_mouth)) + self.cri_component(real_mouth, torch.zeros_like(real_mouth))
        loss += l_g_gan_mouth
        loss_dict[f'l_g_gan_mouth'] = l_g_gan_mouth

        return loss, loss_dict

class GeneralLPIPSWithDiscriminatorComponent(nn.Module):
    def __init__(
        self,
        disc_start: int,
        logvar_init: float = 0.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_loss: str = "hinge",
        scale_input_to_tgt_size: bool = False,
        dims: int = 2,
        learn_logvar: bool = False,
        regularization_weights: Union[None, Dict[str, float]] = None,
        additional_log_keys: Optional[List[str]] = None,
        discriminator_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.dims = dims
        if self.dims > 2:
            print(
                f"running with dims={dims}. This means that for perceptual loss "
                f"calculation, the LPIPS loss will be applied to each frame "
                f"independently."
            )
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        assert disc_loss in ["hinge", "vanilla"]
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )
        self.learn_logvar = learn_logvar

        discriminator_config = default(
            discriminator_config,
            {
                "target": "talking_head.models.vae.modules.autoencoding.lpips.model.model.NLayerDiscriminator",
                "params": {
                    "input_nc": disc_in_channels,
                    "n_layers": disc_num_layers,
                    "use_actnorm": False,
                },
            },
        )

        self.discriminator = instantiate_from_config(discriminator_config).apply(
            weights_init
        )
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.regularization_weights = default(regularization_weights, {})

        self.forward_keys = [
            "optimizer_idx",
            "global_step",
            "last_layer",
            "split",
            "regularization_log",
        ]

        self.additional_log_keys = set(default(additional_log_keys, []))
        self.additional_log_keys.update(set(self.regularization_weights.keys()))


        self.left_eye_discriminator = FacialComponentDiscriminator()
        self.right_eye_discriminator = FacialComponentDiscriminator()
        self.mouth_discriminator = FacialComponentDiscriminator()
        self.facial_component_loss = FacialComponentLoss(
            self.left_eye_discriminator,
            self.right_eye_discriminator,
            self.mouth_discriminator
        )
 
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        params = list(self.discriminator.parameters())
        params += list(self.left_eye_discriminator.parameters())
        params += list(self.right_eye_discriminator.parameters())
        params += list(self.mouth_discriminator.parameters())
        return iter(params)


    def get_trainable_autoencoder_parameters(self) -> Iterator[nn.Parameter]:
        if self.learn_logvar:
            yield self.logvar
        yield from ()

    @torch.no_grad()
    def log_images(
        self, inputs: torch.Tensor, reconstructions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # calc logits of real/fake
        logits_real = self.discriminator(inputs.contiguous().detach())
        if len(logits_real.shape) < 4:
            # Non patch-discriminator
            return dict()
        logits_fake = self.discriminator(reconstructions.contiguous().detach())
        # -> (b, 1, h, w)

        # parameters for colormapping
        high = max(logits_fake.abs().max(), logits_real.abs().max()).item()
        cmap = colormaps["PiYG"]  # diverging colormap

        def to_colormap(logits: torch.Tensor) -> torch.Tensor:
            """(b, 1, ...) -> (b, 3, ...)"""
            logits = (logits + high) / (2 * high)
            logits_np = cmap(logits.cpu().numpy())[..., :3]  # truncate alpha channel
            # -> (b, 1, ..., 3)
            logits = torch.from_numpy(logits_np).to(logits.device)
            return rearrange(logits, "b 1 ... c -> b c ...")

        logits_real = torch.nn.functional.interpolate(
            logits_real,
            size=inputs.shape[-2:],
            mode="nearest",
            antialias=False,
        )
        logits_fake = torch.nn.functional.interpolate(
            logits_fake,
            size=reconstructions.shape[-2:],
            mode="nearest",
            antialias=False,
        )

        # alpha value of logits for overlay
        alpha_real = torch.abs(logits_real) / high
        alpha_fake = torch.abs(logits_fake) / high
        # -> (b, 1, h, w) in range [0, 0.5]
        # alpha value of lines don't really matter, since the values are the same
        # for both images and logits anyway
        grid_alpha_real = torchvision.utils.make_grid(alpha_real, nrow=4)
        grid_alpha_fake = torchvision.utils.make_grid(alpha_fake, nrow=4)
        grid_alpha = 0.8 * torch.cat((grid_alpha_real, grid_alpha_fake), dim=1)
        # -> (1, h, w)
        # blend logits and images together

        # prepare logits for plotting
        logits_real = to_colormap(logits_real)
        logits_fake = to_colormap(logits_fake)
        # resize logits
        # -> (b, 3, h, w)

        # make some grids
        # add all logits to one plot
        logits_real = torchvision.utils.make_grid(logits_real, nrow=4)
        logits_fake = torchvision.utils.make_grid(logits_fake, nrow=4)
        # I just love how torchvision calls the number of columns `nrow`
        grid_logits = torch.cat((logits_real, logits_fake), dim=1)
        # -> (3, h, w)

        grid_images_real = torchvision.utils.make_grid(0.5 * inputs + 0.5, nrow=4)
        grid_images_fake = torchvision.utils.make_grid(
            0.5 * reconstructions + 0.5, nrow=4
        )
        grid_images = torch.cat((grid_images_real, grid_images_fake), dim=1)
        # -> (3, h, w) in range [0, 1]

        grid_blend = grid_alpha * grid_logits + (1 - grid_alpha) * grid_images

        # Create labeled colorbar
        dpi = 100
        height = 128 / dpi
        width = grid_logits.shape[2] / dpi
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        img = ax.imshow(np.array([[-high, high]]), cmap=cmap)
        plt.colorbar(
            img,
            cax=ax,
            orientation="horizontal",
            fraction=0.9,
            aspect=width / height,
            pad=0.0,
        )
        img.set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        # manually convert figure to numpy
        cbar_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        cbar_np = cbar_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cbar = torch.from_numpy(cbar_np.copy()).to(grid_logits.dtype) / 255.0
        cbar = rearrange(cbar, "h w c -> c h w").to(grid_logits.device)

        # Add colorbar to plot
        annotated_grid = torch.cat((grid_logits, cbar), dim=1)
        blended_grid = torch.cat((grid_blend, cbar), dim=1)
        return {
            "vis_logits": 2 * annotated_grid[None, ...] - 1,
            "vis_logits_blended": 2 * blended_grid[None, ...] - 1,
        }

    def calculate_adaptive_weight(
        self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.Tensor
    ) -> torch.Tensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        *,  # added because I changed the order here
        regularization_log: Dict[str, torch.Tensor],
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor,
        split: str = "train",
        weights: Union[None, float, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if self.scale_input_to_tgt_size:
            inputs = torch.nn.functional.interpolate(
                inputs, reconstructions.shape[2:], mode="bicubic", antialias=True
            )

        if self.dims > 2:
            inputs, reconstructions = map(
                lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                (inputs, reconstructions),
            )

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss, weighted_nll_loss = self.get_nll_loss(rec_loss, weights)



        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if global_step >= self.discriminator_iter_start or not self.training:
                logits_fake = self.discriminator(reconstructions.contiguous())
                gan_loss = -torch.mean(logits_fake)
 
                # 提取面部组件位置
                component_locations_batch = [extract_facial_components(img.permute(1, 2, 0).cpu().numpy()) for img in inputs]        # 提取面部组件位置 
                valid_indices = [i for i, loc in enumerate(component_locations_batch) if loc is not None]
                if valid_indices:
                    valid_reconstructions = reconstructions[valid_indices]
                    valid_inputs = inputs[valid_indices]
                    valid_component_locations = [component_locations_batch[i] for i in valid_indices]
                    facial_comp_loss, _ = self.facial_component_loss(valid_reconstructions.contiguous(), valid_inputs.contiguous(), valid_component_locations)
                else:
                    facial_comp_loss = torch.tensor(0.0, requires_grad=True)

  
                g_loss = gan_loss  - facial_comp_loss

                if self.training:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)
                gan_loss = torch.tensor(0.0, requires_grad=True)
                facial_comp_loss = torch.tensor(0.0, requires_grad=True)

            loss = weighted_nll_loss + d_weight * self.disc_factor * g_loss
            log = dict()
            for k in regularization_log:
                if k in self.regularization_weights:
                    loss = loss + self.regularization_weights[k] * regularization_log[k]
                if k in self.additional_log_keys:
                    log[f"{split}/{k}"] = regularization_log[k].detach().float().mean()

            log.update(
                {
                    f"{split}/loss/total": loss.clone().detach().mean(),
                    f"{split}/loss/nll": nll_loss.detach().mean(),
                    f"{split}/loss/rec": rec_loss.detach().mean(),
                    f"{split}/loss/g": g_loss.detach().mean(),
                    f"{split}/loss/gan": gan_loss.detach().mean(),
                    f"{split}/loss/facial_component_gan_loss": -facial_comp_loss.detach().mean(),
                    f"{split}/loss/p_loss": p_loss.clone().detach().mean(),
                    f"{split}/scalars/logvar": self.logvar.detach(),
                    f"{split}/scalars/d_weight": d_weight.detach(),
                }
            )

            return loss, log
        elif optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
        
            # 提取面部组件位置
            component_locations_batch = [extract_facial_components(img.permute(1, 2, 0).cpu().numpy()) for img in inputs]    
            valid_indices = [i for i, loc in enumerate(component_locations_batch) if loc is not None]
            if valid_indices:
                valid_reconstructions = reconstructions[valid_indices]
                valid_inputs = inputs[valid_indices]
                valid_component_locations = [component_locations_batch[i] for i in valid_indices]
                facial_comp_loss, _ = self.facial_component_loss(valid_reconstructions.contiguous(), valid_inputs.contiguous(), valid_component_locations)
            else:
                facial_comp_loss = torch.tensor(0.0, requires_grad=True)

  

            if global_step >= self.discriminator_iter_start or not self.training:
                d_loss = self.disc_factor * (self.disc_loss(logits_real, logits_fake)  + facial_comp_loss)
            else:
                d_loss = torch.tensor(0.0, requires_grad=True)

            log = {
                f"{split}/loss/disc": d_loss.clone().detach().mean(),
                f"{split}/logits/real": logits_real.detach().mean(),
                f"{split}/logits/fake": logits_fake.detach().mean(),
            }
            return d_loss, log
        else:
            raise NotImplementedError(f"Unknown optimizer_idx {optimizer_idx}")

    def get_nll_loss(
        self,
        rec_loss: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        return nll_loss, weighted_nll_loss
