import cv2
import numpy as np
import rtmpose_s2

# 티셔츠 PNG 이미지 (RGBA)
tshirt_original = cv2.imread("tshirt.png", cv2.IMREAD_UNCHANGED)
if tshirt_original is None:
    print("[ERROR] 티셔츠 이미지를 불러오지 못했습니다.")
    exit(1)

# 티셔츠 이미지 상에서 어깨 기준점 (픽셀 단위)
src_pts = np.float32([
    [241, 180],   # 오른쪽 어깨 (keypoints[6]에 대응)
    [760, 180],   # 왼쪽 어깨 (keypoints[5]에 대응)
    [491, 300]    # 티셔츠 중심점 (어깨 아래)
])

# RTMPose 스트리밍 시작
pose_stream = rtmpose_s2.RTMPoseStream(inference_gap=4)

while True:
    frame, keypoints, scores = pose_stream.read()
    if frame is None:
        break

    if keypoints is not None and keypoints.shape[0] >= 1:
        person = keypoints[0]

        if person.shape[0] > 6:
            try:
                # 어깨 좌표 추출
                l_shoulder = person[5]
                r_shoulder = person[6]

                # 어깨 중심 좌표
                mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
                mid_y = (l_shoulder[1] + r_shoulder[1]) / 2

                # 어깨 너비 계산
                shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)

                # 어깨 너비에 비례한 세로 길이 (최소 80, 최대 250)
                vertical_offset = np.clip(shoulder_width * 0.8, 0, 120)
                center = np.array([mid_x, mid_y + vertical_offset])

                # 목적지 좌표 설정
                dst_pts = np.float32([
                    r_shoulder,
                    l_shoulder,
                    center
                ])

                # 어파인 변환 행렬 계산
                M = cv2.getAffineTransform(src_pts, dst_pts)

                # 티셔츠 이미지 변환
                warped = cv2.warpAffine(
                    tshirt_original, M, (frame.shape[1], frame.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
                )

                # 알파 채널 기반 합성
                alpha = warped[:, :, 3] / 255.0
                for c in range(3):
                    frame[:, :, c] = (
                        frame[:, :, c] * (1 - alpha) + warped[:, :, c] * alpha
                    )

                # 디버그 로그 (원하면 삭제 가능)
                # print(f"[INFO] 어깨 너비: {shoulder_width:.1f}, 세로 오프셋: {vertical_offset:.1f}")

            except Exception as e:
                print(f"[ERROR] Affine 변환 또는 합성 중 오류: {e}")
        else:
            print("[WARN] 관절 수 부족 - 어깨 keypoints[5], [6] 없음")
    else:
        print("[WARN] keypoints 없음 또는 사람 미검출")

    cv2.imshow("3D T-Shirt Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose_stream.release()
cv2.destroyAllWindows()
