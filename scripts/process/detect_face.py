from insightface.app import FaceAnalysis
from decord import VideoReader
import cv2


def detect_face(video_path):
    face_exists = True
    video_reader = VideoReader(video_path)
    video_len = len(video_reader)
    # sample 5 frames with equally paced
    batch_idx = [int(i * video_len / 5) for i in range(5)]
    frame_nps = video_reader.get_batch(batch_idx).asnumpy()
    for frame_np in frame_nps:
        faces = face_analysis.get(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        if len(faces) != 1:
            face_exists = False
            break
        # TODO
        # else:
        #     if faces[0].bbox[0] < 0 or faces[0].bbox[1] < 0 or faces[0].bbox[2] < 0 or faces[0].bbox[3] < 0:
        #         face_exists = False
        #         break

    return face_exists


if __name__ == "__main__":
    face_analysis_path = "/mnt/data/longtaozheng/talking-head/pretrained_models/face_analysis"
    face_analysis = FaceAnalysis(
        name="",
        root=face_analysis_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    video_paths = [
        "/mnt/data/longtaozheng/talking_head/check_samples/talkinghead_1kh/UjAxgnvxIJE_0072_S1227_E1257_L893_T135_R1181_B423.mp4",
        "/mnt/data/longtaozheng/talking_head/check_samples/talkinghead_1kh/mXNd6zUVwP0_0008_S1027_E1063_L650_T154_R1082_B586.mp4",
        # "/mnt/data/longtaozheng/talking_head/check_samples/talkinghead_1kh/eXeWNtyjgPk_0007_S764_E815_L306_T177_R562_B433.mp4",
        "/mnt/data/longtaozheng/talking_head/check_samples/vfhq/Clip+ZrcxU4cfzvQ+P0+C1+F317-470.mp4",
    ]

    for video_path in video_paths:
        face_exists = detect_face(video_path)
        print(f"face_exists: {face_exists}")
