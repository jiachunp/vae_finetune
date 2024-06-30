from decord import VideoReader
import cv2
from insightface.app import FaceAnalysis
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange


def save_images_grid(images: torch.Tensor, path: str, rescale=False, n_rows=6):
    """
    Save a grid of images to a file.

    Args:
    - images: torch.Tensor of shape (b, c, h, w)
    - path: str, file path to save the image
    - rescale: bool, whether to rescale the image from [-1, 1] to [0, 1]
    - n_rows: int, number of images per row in the grid
    """
    images = torchvision.utils.make_grid(images, nrow=n_rows)
    images = images.transpose(0, 1).transpose(1, 2).squeeze(-1)
    if rescale:
        images = (images + 1.0) / 2.0
    images = (images * 255).numpy().astype(np.uint8)
    
    imageio.imsave(path, images)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    """
    Save a grid of videos to a file.
    
    Args:
    - videos: torch.Tensor of shape (b, f, c, h, w)
    - path: str, file path to save the video
    - rescale: bool, whether to rescale the video from [-1, 1] to [0, 1]
    - n_rows: int, number of videos per row in the grid
    - fps: int, frames per second of the video
    """
    videos = rearrange(videos, "b c f h w -> f b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    imageio.mimsave(path, outputs, fps=fps)


class VideoProcessor:
    def __init__(self, face_analysis_model_path):
        self.face_analysis = FaceAnalysis(
            name="",
            root=face_analysis_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    def get_face_embedding_from_video(self, input_path):
        try:
            video_reader = VideoReader(input_path.as_posix())
            assert (video_reader is not None and len(video_reader) > 0), "Fail to load video frames"
            video_length = len(video_reader)

            face_emb = None
            for i in range(video_length):
                frame = video_reader[i].asnumpy()
                # detect face
                faces = self.face_analysis.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                assert len(faces) == 1, "Only one face should be detected"
                # face embedding
                face_emb = faces[0]["embedding"]
                if face_emb is not None:
                    break
        except Exception as e:
            print(e)
            if "video_reader" in locals():
                del video_reader
            return None, None
        
        del video_reader
        return face_emb, video_length
