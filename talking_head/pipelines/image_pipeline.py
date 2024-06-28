# pylint: disable=R0801
"""
This module is responsible for handling the animation of faces using a combination of deep learning models and image processing techniques. 
It provides a pipeline to generate realistic face animations by incorporating user-provided conditions such as facial expressions and environments. 
The module utilizes various schedulers and utilities to optimize the animation process and ensure efficient performance.

Functions and Classes:
- ImagePipelineOutput: A class that represents the output of the animation pipeline, c
    ontaining properties and methods related to the generated images.
- prepare_latents: A function that prepares the initial noise for the animation process, 
    scaling it according to the scheduler's requirements.
- prepare_condition: A function that processes the user-provided conditions 
    (e.g., facial expressions) and prepares them for use in the animation pipeline.
- decode_latents: A function that decodes the latent representations of the face animations into 
    their corresponding image formats.
- prepare_extra_step_kwargs: A function that prepares additional parameters for each step of 
    the animation process, such as the generator and eta values.

Dependencies:
- numpy: A library for numerical computing.
- torch: A machine learning library based on PyTorch.
- diffusers: A library for image-to-image diffusion models.
- transformers: A library for pre-trained transformer models.

Usage:
- To create an instance of the animation pipeline, provide the necessary components such as 
    the VAE, reference UNET, denoising UNET, face locator, and image processor.
- Use the pipeline's methods to prepare the latents, conditions, and extra step arguments as 
    required for the animation process.
- Generate the face animations by decoding the latents and processing the conditions.

Note:
- The module is designed to work with the diffusers library, which is based on 
    the paper "Diffusion Models for Image-to-Image Translation" (https://arxiv.org/abs/2102.02765).
- The face animations generated by this module should be used for entertainment purposes 
    only and should respect the rights and privacy of the individuals involved.
"""
import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor

from talking_head.models.mutual_self_attention import ReferenceAttentionControl

if is_accelerate_available():
    from accelerate import cpu_offload
else:
    raise ImportError("Please install accelerate via `pip install accelerate`")


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    ImagePipelineOutput is a class that represents the output of the static pipeline.
    It contains the images generated by the pipeline as a union of torch.Tensor and np.ndarray.
    
    Attributes:
        images (Union[torch.Tensor, np.ndarray]): The generated images.
    """
    images: Union[torch.Tensor, np.ndarray]


class ImagePipeline(DiffusionPipeline):
    """
    ImagePipelineOutput is a class that represents the output of the static pipeline.
    It contains the images generated by the pipeline as a union of torch.Tensor and np.ndarray.
    
    Attributes:
        images (Union[torch.Tensor, np.ndarray]): The generated images.
    """
    _optional_components = []

    def __init__(
        self,
        vae,
        reference_unet,
        denoising_unet,
        face_locator,
        imageproj,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            face_locator=face_locator,
            scheduler=scheduler,
            imageproj=imageproj,
        )
        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def enable_vae_slicing(self):
        """
        Enable VAE slicing.

        This method enables slicing for the VAE model, which can help improve the performance of decoding latents when working with large images.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        """
        Disable vae slicing.

        This function disables the vae slicing for the ImagePipeline object. 
        It calls the `disable_slicing()` method of the vae model. 
        This is useful when you want to use the entire vae model for decoding latents 
        instead of slicing it for better performance.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        """
        Offloads selected models to the GPU for increased performance.

        Args:
            gpu_id (int, optional): The ID of the GPU to offload models to. Defaults to 0.
        """
        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        """
        Decode the given latents to video frames.

        Parameters:
        latents (torch.Tensor): The latents to be decoded. Shape: (batch_size, num_channels_latents, video_length, height, width).

        Returns:
        video (torch.Tensor): The decoded video frames. Shape: (batch_size, num_channels_latents, video_length, height, width).
        """
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(
                latents[frame_idx: frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        """
        Prepare extra keyword arguments for the scheduler step.

        Since not all schedulers have the same signature, this function helps to create a consistent interface for the scheduler.

        Args:
            generator (Optional[torch.Generator]): A random number generator for reproducibility.
            eta (float): The eta parameter used with the DDIMScheduler. It should be between 0 and 1.

        Returns:
            dict: A dictionary containing the extra keyword arguments for the scheduler step.
        """
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepares the initial latents for the diffusion pipeline.

        Args:
            batch_size (int): The number of images to generate in one forward pass.
            num_channels_latents (int): The number of channels in the latents tensor.
            width (int): The width of the latents tensor.
            height (int): The height of the latents tensor.
            dtype (torch.dtype): The data type of the latents tensor.
            device (torch.device): The device to place the latents tensor on.
            generator (Optional[torch.Generator], optional): A random number generator
                for reproducibility. Defaults to None.
            latents (Optional[torch.Tensor], optional): Pre-computed latents to use as
                initial conditions for the diffusion process. Defaults to None.

        Returns:
            torch.Tensor: The prepared latents tensor.
        """
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(
        self,
        cond_image,
        width,
        height,
        device,
        dtype,
        do_classififer_free_guidance=False,
    ):
        """
        Prepares the condition for the face animation pipeline.

        Args:
            cond_image (torch.Tensor): The conditional image tensor.
            width (int): The width of the output image.
            height (int): The height of the output image.
            device (torch.device): The device to run the pipeline on.
            dtype (torch.dtype): The data type of the tensor.
            do_classififer_free_guidance (bool, optional): Whether to use classifier-free guidance or not. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of processed condition and mask tensors.
        """
        image = self.cond_image_processor.preprocess(
            cond_image, height=height, width=width
        ).to(dtype=torch.float32)

        image = image.to(device=device, dtype=dtype)

        if do_classififer_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        face_mask,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        face_embedding,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        image_prompt_embeds = self.imageproj(face_embedding)
        uncond_image_prompt_embeds = self.imageproj(
            torch.zeros_like(face_embedding))

        if do_classifier_free_guidance:
            image_prompt_embeds = torch.cat(
                [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
            )

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            face_embedding.dtype,
            device,
            generator,
        )
        latents = latents.unsqueeze(2)  # (bs, c, 1, h', w')
        # latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        # Prepare face mask image
        face_mask_tensor = self.cond_image_processor.preprocess(
            face_mask, height=height, width=width
        )
        face_mask_tensor = face_mask_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
        face_mask_tensor = face_mask_tensor.to(
            device=device, dtype=self.face_locator.dtype
        )
        mask_fea = self.face_locator(face_mask_tensor)
        mask_fea = (
            torch.cat(
                [mask_fea] * 2) if do_classifier_free_guidance else mask_fea
        )

        # denoising loop
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )

                    # 2. Update reference unet feature into denosing net
                    reference_control_reader.update(reference_control_writer)

                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat(
                        [latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_prompt_embeds,
                    mask_cond_fea=mask_fea,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i +
                                                    1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
            reference_control_reader.clear()
            reference_control_writer.clear()

        # Post-processing
        image = self.decode_latents(latents)  # (b, c, 1, h, w)

        # Convert to tensor
        if output_type == "tensor":
            image = torch.from_numpy(image)

        if not return_dict:
            return image

        return ImagePipelineOutput(images=image)
