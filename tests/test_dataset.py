from einops import rearrange

from talking_head.datasets.video_dataset import VideoDataset
from talking_head.utils.video_utils import save_images_grid, save_videos_grid


dataset = VideoDataset(
    n_motion_frames=2,
    n_sample_frames=14,
    metadata_paths=["/mnt/data/public/dataset/talking_head/embeddings/HDTF/metadata.jsonl"],
)
print(f"dataset size: {len(dataset)}")
for i, d in enumerate(dataset):
    print(i)
    if i == 0:
        # print(d["pixel_values"].shape)  # (f c h w) otherwise
        # print(d["pixel_values_ref_img"].shape)  # (1+n_motion, c, h, w)
        # print(d["face_emb"].shape)
        # print(d["audio_emb"].shape)

        pixel_values = rearrange(d["pixel_values"].unsqueeze(0), "b f c h w -> b c f h w")
        for idx, pixel_value in enumerate(pixel_values):
            pixel_value = pixel_value.unsqueeze(0)
            save_videos_grid(pixel_value, f"sanity_check.gif", rescale=True)

        pixel_values_ref_img = d["pixel_values_ref_img"]
        save_images_grid(pixel_values_ref_img, f"sanity_check_ref.png", rescale=True)

    if i > 30:
        break
