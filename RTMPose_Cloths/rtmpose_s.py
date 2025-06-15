import cv2
import time
from rtmlib import Wholebody, draw_skeleton

# 디바이스 및 백엔드 설정
device = 'cuda'  # 'cuda'를 사용하여 GPU 활용
backend = 'onnxruntime'  # 또는 'opencv', 'openvino'

# RTMPose-s 모델 초기화
pose_model = Wholebody(to_openpose=False, mode='lightweight', backend=backend, device=device)

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

# FPS 측정을 위한 초기 설정
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 포즈 추정
    keypoints, scores = pose_model(frame)

    # FPS 계산
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # 결과 시각화
    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

    # FPS 텍스트 추가
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 프레임 출력
    cv2.imshow('RTMPose-s Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
