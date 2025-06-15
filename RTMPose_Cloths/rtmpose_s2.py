# 파일명: rtmpose_s2.py
import cv2
from rtmlib import Wholebody

class RTMPoseStream:
    def __init__(self, device='cuda', backend='onnxruntime', inference_gap=4):
        self.pose_model = Wholebody(to_openpose=False, mode='lightweight', backend=backend, device=device)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 384)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
        self.inference_gap = inference_gap
        self.frame_count = 0
        self.keypoints = None
        self.scores = None

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        self.frame_count += 1
        frame = cv2.resize(frame, (768, 576))

        if self.frame_count % self.inference_gap == 0:
            self.keypoints, self.scores = self.pose_model(frame)

        return frame, self.keypoints, self.scores

    def release(self):
        self.cap.release()

