import cv2
import time
from rtmlib import Wholebody, draw_skeleton
import matplotlib.pyplot as plt

device = 'cuda'
backend = 'onnxruntime'
pose_model = Wholebody(to_openpose=False, mode='lightweight', backend=backend, device=device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

results = []
test_duration = 15  # 각 설정당 15초간 테스트
frame_skip_values = list(range(1, 21))  # 1부터 20프레임 간격까지 테스트

for skip in frame_skip_values:
    print(f"Testing inference every {skip} frame(s)...")
    keypoints, scores = None, None
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (768, 576))
        frame_count += 1

        # 일정 프레임마다 추론 수행
        if frame_count % skip == 0:
            keypoints, scores = pose_model(frame)

        # 시각화
        if keypoints is not None:
            frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

        # 테스트 종료 조건
        if time.time() - start_time >= test_duration:
            break

    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    results.append((skip, fps))
    print(f"  ➤ Avg FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()

# 그래프 출력
x_vals, y_vals = zip(*results)
plt.plot(x_vals, y_vals, marker='o')
plt.xlabel("N Frames")
plt.ylabel("Average FPS over 15 seconds")
plt.title("Performance Trade-off: Inference Frequency vs FPS")
plt.grid(True)
plt.show()
