import cv2
import time

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 확인할 프레임 수
num_frames = 120

print("웹캠 FPS 측정 중...")

# 시작 시간
start = time.time()

# 프레임 수만큼 읽기
for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

# 종료 시간
end = time.time()

# 측정된 FPS
seconds = end - start
fps = num_frames / seconds
print(f"캡처된 프레임 수: {num_frames}")
print(f"걸린 시간: {seconds:.2f}초")
print(f"실제 웹캠 FPS: {fps:.2f}")

cap.release()
