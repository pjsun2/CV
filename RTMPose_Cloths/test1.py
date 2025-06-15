import cv2

cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다.

# 현재 해상도 확인
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"현재 해상도: {int(width)} x {int(height)}")

cap.release()
