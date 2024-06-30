from decord import VideoReader
import cv2
from insightface.app import FaceAnalysis


class VideoProcessor:
    def __init__(self, face_analysis_model_path):
        self.face_analysis = FaceAnalysis(
            name="",
            root=face_analysis_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    def get_face_embedding_from_video(self, input_path):
        video_reader = VideoReader(input_path.as_posix())
        face_emb = None
        for i in range(len(video_reader)):
            frame = video_reader[i].asnumpy()
            try:
                # detect face
                faces = self.face_analysis.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                assert len(faces) == 1, "Only one face should be detected"
                # face embedding
                face_emb = faces[0]["embedding"]
                if face_emb is not None:
                    break
            except Exception:
                continue
        
        del video_reader

        return face_emb
