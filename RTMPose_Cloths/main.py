# main.py
import cv2
import rtmpose_s2
from rotate import apply_tshirt_overlay

# RTMPose 스트리밍 시작
pose_stream = rtmpose_s2.RTMPoseStream(inference_gap=4)

while True:
    frame, keypoints, scores = pose_stream.read()
    if frame is None:
        break

    if keypoints is not None and keypoints.shape[0] >= 1:
        person = keypoints[0]  # 첫 번째 사람만 사용
        frame = apply_tshirt_overlay(frame, person)

    cv2.imshow("3D T-Shirt Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose_stream.release()
cv2.destroyAllWindows()
