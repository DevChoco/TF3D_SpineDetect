"""
스켈레톤 파싱 및 자세 분석 모듈

이 모듈은 다음 기능을 제공합니다:
- AI 기반 랜드마크 검출 (MediaPipe)
- 포인트 클라우드에서 스켈레톤 생성
- 척추 각도 분석
- 스켈레톤 시각화
"""

import numpy as np
import math
import copy
import open3d as o3d
import mediapipe as mp
from PIL import Image


def detect_landmarks_with_ai(image_path):
    """
    MediaPipe를 사용하여 이미지에서 해부학적 랜드마크를 검출합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        
    Returns:
        dict: 랜드마크 좌표 딕셔너리 또는 None
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    try:
        # 이미지 로드 및 전처리
        with Image.open(image_path) as img:
            image = np.array(img)
            if len(image.shape) == 3:
                # RGB로 변환 (MediaPipe는 RGB 형식 요구)
                image_rgb = image
            else:
                # 그레이스케일을 RGB로 변환
                image_rgb = np.stack([image, image, image], axis=-1)
            
            # MediaPipe로 자세 검출
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # 주요 랜드마크 추출
                landmarks = {}
                h, w = image.shape[:2]
                
                # MediaPipe 랜드마크 인덱스
                landmark_indices = {
                    'nose': mp_pose.PoseLandmark.NOSE.value,
                    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    'left_hip': mp_pose.PoseLandmark.LEFT_HIP.value,
                    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
                }
                
                for name, idx in landmark_indices.items():
                    if idx < len(results.pose_landmarks.landmark):
                        landmark = results.pose_landmarks.landmark[idx]
                        if landmark.visibility > 0.5:  # 신뢰도가 높은 랜드마크만 사용
                            landmarks[name] = {
                                'x': landmark.x * w,
                                'y': landmark.y * h,
                                'visibility': landmark.visibility
                            }
                
                print(f"AI 랜드마크 검출 성공: {len(landmarks)}개 랜드마크")
                return landmarks
            else:
                print("AI 랜드마크 검출 실패")
                return None
                
    except Exception as e:
        print(f"AI 랜드마크 검출 중 오류: {e}")
        return None


def find_spine_center_at_height(points, height, search_radius=10):
    """
    특정 높이에서 실제 척추(등 중앙) 위치를 찾습니다.
    좌우 중앙선을 강제하고 노이즈를 제거합니다.
    
    Args:
        points: 포인트 클라우드 배열
        height: 검색할 높이
        search_radius: 높이 검색 반경
        
    Returns:
        tuple: (x, y, z) 척추 중심 좌표
    """
    # 해당 높이 근처의 포인트들 추출
    height_mask = np.abs(points[:, 1] - height) < search_radius
    height_points = points[height_mask]
    
    if len(height_points) == 0:
        return None
    
    # X축 중앙값 계산 (좌우 대칭의 정확한 중심)
    x_median = np.median(height_points[:, 0])
    
    # 척추는 항상 신체의 정중앙에 위치 - X 좌표 고정
    # 매우 좁은 중앙 영역만 선택 (전체 너비의 5%)
    x_range = np.ptp(height_points[:, 0]) * 0.05
    center_mask = np.abs(height_points[:, 0] - x_median) < x_range
    center_points = height_points[center_mask]
    
    if len(center_points) == 0:
        # 범위를 조금 넓혀서 재시도
        x_range = np.ptp(height_points[:, 0]) * 0.10
        center_mask = np.abs(height_points[:, 0] - x_median) < x_range
        center_points = height_points[center_mask]
        
        if len(center_points) == 0:
            return None
    
    # 등 쪽 (Z값이 가장 작은 쪽)의 포인트들을 찾음
    # 하위 10%의 Z값을 가진 포인트들만 선택 (더 엄격하게)
    z_values = center_points[:, 2]
    z_threshold = np.percentile(z_values, 10)  # 하위 10%
    back_points = center_points[center_points[:, 2] <= z_threshold]
    
    if len(back_points) == 0:
        z_threshold = np.percentile(z_values, 20)  # 재시도
        back_points = center_points[center_points[:, 2] <= z_threshold]
        
        if len(back_points) == 0:
            back_points = center_points
    
    # 척추 위치 계산
    # X는 정중앙으로 강제 (노이즈 제거)
    spine_x = x_median
    spine_y = height
    # Z는 등 표면의 median 사용 (mean보다 노이즈에 강함)
    spine_z = np.median(back_points[:, 2])
    
    return np.array([spine_x, spine_y, spine_z])


def create_skeleton_from_pointcloud(pcd, ai_landmarks=None):
    """
    포인트 클라우드에서 인체 스켈레톤을 생성합니다.
    실제 메시의 등 표면을 분석하여 척추 위치를 정확하게 예측합니다.
    
    Args:
        pcd (o3d.geometry.PointCloud): 포인트 클라우드
        ai_landmarks (dict): AI 검출된 랜드마크 딕셔너리
        
    Returns:
        dict: 스켈레톤 포인트 딕셔너리
    """
    points = np.asarray(pcd.points)
    
    # 포인트 클라우드의 바운딩 박스 계산
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    height = max_bound[1] - min_bound[1]
    width = max_bound[0] - min_bound[0]
    depth = max_bound[2] - min_bound[2]
    z_range = depth  # Z축 범위 (깊이)
    
    print(f"모델 크기 - Height: {height:.2f}, Width: {width:.2f}, Depth: {depth:.2f}")
    print("의학적으로 정확한 해부학적 위치 적용:")
    print("- 척추: 등 쪽(뒤쪽)에 위치")
    print("- 어깨: 척추보다 앞쪽에 위치")  
    print("- 골반: 몸의 중앙에 위치")
    
    # 주요 해부학적 랜드마크 정의
    skeleton_points = {}
    
    # AI 랜드마크를 사용한 정확한 위치 계산
    if ai_landmarks:
        print("AI 검출 랜드마크를 사용하여 정확한 골격 구조 생성")
        
        # 이미지 크기 (실제 depth map 크기로 조정)
        img_size = 512  # 일반적인 depth map 크기
        
        # 어깨 위치 (AI 검출 기반) - 높이 조정
        if 'left_shoulder' in ai_landmarks and 'right_shoulder' in ai_landmarks:
            left_shoulder_y = ai_landmarks['left_shoulder']['y']
            right_shoulder_y = ai_landmarks['right_shoulder']['y']
            left_shoulder_x = ai_landmarks['left_shoulder']['x']
            right_shoulder_x = ai_landmarks['right_shoulder']['x']
            
            # 이미지 좌표를 3D 좌표로 변환
            shoulder_height = max_bound[1] - (((left_shoulder_y + right_shoulder_y) / 2) / img_size) * height * 0.8
            shoulder_left_x = center[0] - width * ((img_size/2 - left_shoulder_x) / img_size) * 0.8
            shoulder_right_x = center[0] - width * ((img_size/2 - right_shoulder_x) / img_size) * 0.8
            
            # 어깨는 척추보다 앞쪽에 위치 (의학적으로 정확한 위치) - 신체 내부
            shoulder_z = center[2] - z_range * 0.05  # 척추보다 약간 앞쪽 (신체 내부)
            
            skeleton_points['left_shoulder'] = [shoulder_left_x, shoulder_height, shoulder_z]
            skeleton_points['right_shoulder'] = [shoulder_right_x, shoulder_height, shoulder_z]
            skeleton_points['shoulder_center'] = [(shoulder_left_x + shoulder_right_x) / 2, shoulder_height, shoulder_z]
        
        # 골반 위치 (AI 검출 기반) - 높이 조정
        if 'left_hip' in ai_landmarks and 'right_hip' in ai_landmarks:
            left_hip_y = ai_landmarks['left_hip']['y']
            right_hip_y = ai_landmarks['right_hip']['y']
            left_hip_x = ai_landmarks['left_hip']['x']
            right_hip_x = ai_landmarks['right_hip']['x']
            
            # 이미지 좌표를 3D 좌표로 변환
            hip_height = max_bound[1] - (((left_hip_y + right_hip_y) / 2) / img_size) * height * 0.8
            hip_left_x = center[0] - width * ((img_size/2 - left_hip_x) / img_size) * 0.8
            hip_right_x = center[0] - width * ((img_size/2 - right_hip_x) / img_size) * 0.8
            
            skeleton_points['left_hip'] = [hip_left_x, hip_height, center[2]]
            skeleton_points['right_hip'] = [hip_right_x, hip_height, center[2]]
            skeleton_points['pelvis_center'] = [(hip_left_x + hip_right_x) / 2, hip_height, center[2]]
        
        # 머리와 목 위치 (코 위치 기반) - 높이 조정
        if 'nose' in ai_landmarks:
            nose_y = ai_landmarks['nose']['y']
            nose_x = ai_landmarks['nose']['x']
            
            head_height = max_bound[1] - (nose_y / img_size) * height * 0.8
            head_x = center[0] - width * ((img_size/2 - nose_x) / img_size) * 0.8
            
            # 머리와 목은 척추 위에 위치 (의학적으로 정확한 위치)
            head_z = center[2] + depth * 0.25  # 척추 위치에 맞춤
            
            skeleton_points['head_top'] = [head_x, head_height + height * 0.05, head_z]
            skeleton_points['neck'] = [head_x, head_height - height * 0.05, head_z]
    
    # X축 중앙값 계산 (신체 정중선)
    x_median = np.median(points[:, 0])
    
    # AI 검출이 실패했거나 불완전한 경우 기본값 사용 (높이 조정)
    if 'shoulder_center' not in skeleton_points:
        print("AI 어깨 검출 실패, 조정된 기본 비율 사용")
        shoulder_height = max_bound[1] - height * 0.15  # 0.22에서 0.15로 변경 (더 위로)
        # 어깨는 척추보다 앞쪽에 위치 (의학적으로 정확한 위치) - 신체 내부
        shoulder_z = center[2] - z_range * 0.05  # 척추보다 약간 앞쪽 (신체 내부)
        # 어깨는 정중선을 기준으로 좌우 대칭
        skeleton_points['left_shoulder'] = [x_median - width * 0.25, shoulder_height, shoulder_z]
        skeleton_points['right_shoulder'] = [x_median + width * 0.25, shoulder_height, shoulder_z]
        skeleton_points['shoulder_center'] = [x_median, shoulder_height, shoulder_z]
    else:
        # AI 검출 성공시에도 X축은 정중선으로 보정
        shoulder_center_y = skeleton_points['shoulder_center'][1]
        shoulder_center_z = skeleton_points['shoulder_center'][2]
        shoulder_width = abs(skeleton_points['left_shoulder'][0] - skeleton_points['right_shoulder'][0]) / 2
        skeleton_points['left_shoulder'] = [x_median - shoulder_width, skeleton_points['left_shoulder'][1], skeleton_points['left_shoulder'][2]]
        skeleton_points['right_shoulder'] = [x_median + shoulder_width, skeleton_points['right_shoulder'][1], skeleton_points['right_shoulder'][2]]
        skeleton_points['shoulder_center'] = [x_median, shoulder_center_y, shoulder_center_z]
    
    if 'pelvis_center' not in skeleton_points:
        print("AI 골반 검출 실패, 조정된 기본 비율 사용")
        pelvis_height = max_bound[1] - height * 0.42  # 0.35에서 0.42로 변경 (조금 아래로)
        # 골반도 정중선을 기준으로 좌우 대칭
        skeleton_points['pelvis_center'] = [x_median, pelvis_height, center[2]]
        skeleton_points['left_hip'] = [x_median - width * 0.15, pelvis_height, center[2]]
        skeleton_points['right_hip'] = [x_median + width * 0.15, pelvis_height, center[2]]
    else:
        # AI 검출 성공시에도 X축은 정중선으로 보정
        pelvis_center_y = skeleton_points['pelvis_center'][1]
        pelvis_center_z = skeleton_points['pelvis_center'][2]
        hip_width = abs(skeleton_points['left_hip'][0] - skeleton_points['right_hip'][0]) / 2
        skeleton_points['left_hip'] = [x_median - hip_width, skeleton_points['left_hip'][1], skeleton_points['left_hip'][2]]
        skeleton_points['right_hip'] = [x_median + hip_width, skeleton_points['right_hip'][1], skeleton_points['right_hip'][2]]
        skeleton_points['pelvis_center'] = [x_median, pelvis_center_y, pelvis_center_z]
    
    if 'neck' not in skeleton_points:
        skeleton_points['head_top'] = [x_median, max_bound[1], center[2] - depth * 0.05]  # 머리는 척추 위
        skeleton_points['neck'] = [x_median, max_bound[1] - height * 0.08, center[2] - depth * 0.05]  # 목도 척추 위
    else:
        # AI 검출 성공시에도 X축은 정중선으로 보정
        head_y = skeleton_points['head_top'][1]
        head_z = skeleton_points['head_top'][2]
        neck_y = skeleton_points['neck'][1]
        neck_z = skeleton_points['neck'][2]
        skeleton_points['head_top'] = [x_median, head_y, head_z]
        skeleton_points['neck'] = [x_median, neck_y, neck_z]
    
    # 실제 검출된 위치를 기반으로 척추 구조 생성
    shoulder_height = skeleton_points['shoulder_center'][1]
    pelvis_height = skeleton_points['pelvis_center'][1]
    neck_height = skeleton_points['neck'][1]
    
    # 전체 척추 길이 설정 (목에서 꼬리뼈까지) - 자연스러운 비율로 조정
    spine_start_height = max_bound[1] - height * 0.08  # 목 시작
    spine_end_height = max_bound[1] - height * 0.55    # 미추 끝 (전체 키의 55%)
    total_spine_length = spine_start_height - spine_end_height
    
    # 척추 기본 Z 위치 - 신체 내부 중앙에서 약간 뒤쪽에 위치 (의학적으로 정확한 위치)
    # 포인트클라우드의 실제 Z 범위 내에서 척추 위치 결정
    z_range = max_bound[2] - min_bound[2]
    spine_z_base = center[2] - z_range * 0.1  # 신체 중심에서 뒤쪽으로 10% 이동 (내부에 위치)
    
    print(f"척추 전체 길이: {total_spine_length:.2f} (키의 47%)")
    
    # 머리와 목 - 척추 위에 위치하되 신체 내부에
    skeleton_points['head_top'] = [x_median, max_bound[1], spine_z_base + z_range * 0.05]  # 척추보다 약간 앞쪽 (신체 내부)
    skeleton_points['neck'] = [x_median, spine_start_height, spine_z_base]
    
    # 실제 메시 표면을 분석하여 척추 곡선 생성
    print("=== 실제 메시 표면 기반 척추 위치 추정 ===")
    
    # 경추 (C1-C7) - 전체 척추의 25% - 실제 등 표면 기반
    cervical_length = total_spine_length * 0.25
    cervical_start = skeleton_points['neck']
    cervical_end_height = spine_start_height - cervical_length
    
    print(f"경추 길이: {cervical_length:.2f} (전체 척추의 25%)")
    
    for i in range(7):
        ratio = i / 6
        # 각 경추의 높이 계산
        vertebra_height = cervical_start[1] + (cervical_end_height - cervical_start[1]) * ratio
        
        # 해당 높이에서 실제 등 중앙(척추) 위치 찾기
        spine_pos = find_spine_center_at_height(points, vertebra_height, search_radius=height * 0.05)
        
        if spine_pos is not None:
            skeleton_points[f'cervical_C{i+1}'] = spine_pos.tolist()
            print(f"  C{i+1}: 실제 표면에서 검출 - Z={spine_pos[2]:.2f}")
        else:
            # 검출 실패시 보간
            point = [
                cervical_start[0],
                vertebra_height,
                cervical_start[2]
            ]
            skeleton_points[f'cervical_C{i+1}'] = point
            print(f"  C{i+1}: 보간 사용")
    
    # 흉추 (T1-T12) - 전체 척추의 40% - 실제 등 표면 기반
    thoracic_length = total_spine_length * 0.40
    thoracic_start_height = cervical_end_height
    thoracic_end_height = thoracic_start_height - thoracic_length
    
    print(f"흉추 길이: {thoracic_length:.2f} (전체 척추의 40%)")
    
    for i in range(12):
        ratio = i / 11
        # 각 흉추의 높이 계산
        vertebra_height = thoracic_start_height + (thoracic_end_height - thoracic_start_height) * ratio
        
        # 해당 높이에서 실제 등 중앙(척추) 위치 찾기
        spine_pos = find_spine_center_at_height(points, vertebra_height, search_radius=height * 0.05)
        
        if spine_pos is not None:
            skeleton_points[f'thoracic_T{i+1}'] = spine_pos.tolist()
            print(f"  T{i+1}: 실제 표면에서 검출 - Z={spine_pos[2]:.2f}")
        else:
            # 검출 실패시 보간
            point = [
                center[0],
                vertebra_height,
                spine_z_base
            ]
            skeleton_points[f'thoracic_T{i+1}'] = point
            print(f"  T{i+1}: 보간 사용")
    
    # 요추 (L1-L5) - 전체 척추의 20% - 실제 등 표면 기반
    lumbar_length = total_spine_length * 0.20
    lumbar_start_height = thoracic_end_height
    lumbar_end_height = lumbar_start_height - lumbar_length
    
    print(f"요추 길이: {lumbar_length:.2f} (전체 척추의 20%)")
    
    for i in range(5):
        ratio = i / 4
        # 각 요추의 높이 계산
        vertebra_height = lumbar_start_height + (lumbar_end_height - lumbar_start_height) * ratio
        
        # 해당 높이에서 실제 등 중앙(척추) 위치 찾기
        spine_pos = find_spine_center_at_height(points, vertebra_height, search_radius=height * 0.05)
        
        if spine_pos is not None:
            skeleton_points[f'lumbar_L{i+1}'] = spine_pos.tolist()
            print(f"  L{i+1}: 실제 표면에서 검출 - Z={spine_pos[2]:.2f}")
        else:
            # 검출 실패시 보간
            point = [
                center[0],
                vertebra_height,
                spine_z_base
            ]
            skeleton_points[f'lumbar_L{i+1}'] = point
            print(f"  L{i+1}: 보간 사용")
    
    # 천추와 미추 - 전체 척추의 15% - 실제 등 표면 기반
    sacral_length = total_spine_length * 0.15
    sacral_start_height = lumbar_end_height
    sacral_end_height = sacral_start_height - sacral_length
    
    print(f"천추+미추 길이: {sacral_length:.2f} (전체 척추의 15%)")
    
    # 천추 (S1-S5) - 천추+미추의 80% (골반 후면의 중심)
    sacrum_length = sacral_length * 0.80
    
    for i in range(5):
        ratio = i / 4
        # 각 천추의 높이 계산 (천추 부분만)
        vertebra_height = sacral_start_height + (sacral_end_height - sacral_start_height) * ratio * 0.8
        
        # 해당 높이에서 실제 등 중앙(척추) 위치 찾기
        spine_pos = find_spine_center_at_height(points, vertebra_height, search_radius=height * 0.05)
        
        if spine_pos is not None:
            skeleton_points[f'sacral_S{i+1}'] = spine_pos.tolist()
            print(f"  S{i+1}: 실제 표면에서 검출 - Z={spine_pos[2]:.2f}")
        else:
            # 검출 실패시 보간
            point = [
                center[0],
                vertebra_height,
                spine_z_base
            ]
            skeleton_points[f'sacral_S{i+1}'] = point
            print(f"  S{i+1}: 보간 사용")
    
    # 천추 중심점
    sacrum_center = skeleton_points['sacral_S3']  # S3를 천추 중심으로 사용
    skeleton_points['sacrum'] = sacrum_center
    
    # 미추 (Co1-Co4) - 천추+미추의 20% (척추의 최종 끝)
    coccyx_length = sacral_length * 0.20
    
    for i in range(4):
        ratio = 0.8 + (i / 3) * 0.2  # 0.8에서 1.0까지
        # 각 미추의 높이 계산
        vertebra_height = sacral_start_height + (sacral_end_height - sacral_start_height) * ratio
        
        # 해당 높이에서 실제 등 중앙(척추) 위치 찾기
        spine_pos = find_spine_center_at_height(points, vertebra_height, search_radius=height * 0.05)
        
        if spine_pos is not None:
            skeleton_points[f'coccyx_Co{i+1}'] = spine_pos.tolist()
            print(f"  Co{i+1}: 실제 표면에서 검출 - Z={spine_pos[2]:.2f}")
        else:
            # 검출 실패시 보간
            point = [
                center[0],
                vertebra_height,
                spine_z_base
            ]
            skeleton_points[f'coccyx_Co{i+1}'] = point
            print(f"  Co{i+1}: 보간 사용")
    
    # 미추 끝점
    skeleton_points['coccyx'] = skeleton_points['coccyx_Co4']
    
    # 골반 위치 - 천추를 중심으로 해부학적으로 정확한 위치 설정
    # 골반(장골능)은 천추보다 약간 위에, 고관절은 천추와 같은 높이에 위치
    pelvis_height = sacrum_center[1] + sacrum_length * 0.3  # 천추 중심에서 약간 위
    hip_joint_height = sacrum_center[1]  # 천추 중심과 같은 높이
    
    # 골반은 정중선 기준으로 좌우 대칭 (center[2] - 배 쪽)
    skeleton_points['pelvis_center'] = [x_median, pelvis_height, center[2]]
    skeleton_points['left_hip'] = [x_median - width * 0.15, hip_joint_height, center[2]]
    skeleton_points['right_hip'] = [x_median + width * 0.15, hip_joint_height, center[2]]
    
    print(f"골반 중심 높이: {pelvis_height:.2f} (천추 기준)")
    print(f"고관절 높이: {hip_joint_height:.2f} (천추 중심과 동일)")
    
    # 어깨는 흉추 상부에 위치하되 척추보다 앞쪽에 (의학적으로 정확한 위치) - 신체 내부
    shoulder_height = cervical_end_height + thoracic_length * 0.2  # 흉추 시작에서 더 위로 올림
    shoulder_z = spine_z_base + z_range * 0.08  # 척추보다 앞쪽에 위치 (신체 내부)
    skeleton_points['left_shoulder'] = [x_median - width * 0.25, shoulder_height, shoulder_z]
    skeleton_points['right_shoulder'] = [x_median + width * 0.25, shoulder_height, shoulder_z]
    skeleton_points['shoulder_center'] = [x_median, shoulder_height, shoulder_z]
    
    return skeleton_points


def calculate_spine_angles(skeleton_points):
    """
    척추의 각종 각도를 의학적 Cobb angle 방식으로 계산합니다.
    
    Args:
        skeleton_points (dict): 스켈레톤 포인트 딕셔너리
        
    Returns:
        dict: 각도 분석 결과 딕셔너리
    """
    angles = {}
    
    def calculate_cobb_angle_from_vertebrae(upper_vertebra, lower_vertebra, middle_vertebrae):
        """
        실제 Cobb angle 계산: 상위 척추와 하위 척추의 수평선이 이루는 각도
        
        의학적 Cobb angle:
        1. 상위 척추의 상단 끝판(endplate)에 평행선 그리기
        2. 하위 척추의 하단 끝판에 평행선 그리기
        3. 두 평행선에 수직인 선을 그림
        4. 두 수직선이 이루는 각도 = Cobb angle
        
        단순화: 척추가 일직선이면 0°, 굽어있으면 양수
        """
        upper = np.array(upper_vertebra)
        lower = np.array(lower_vertebra)
        
        # 중간 척추들의 Z좌표 평균 (곡선의 apex 방향)
        if middle_vertebrae:
            middle_z = np.mean([np.array(v)[2] for v in middle_vertebrae])
        else:
            middle_z = (upper[2] + lower[2]) / 2
        
        # 상위 척추에서 중간으로의 벡터 (시상면 YZ 투영)
        upper_to_mid = np.array([0, (upper[1] + lower[1])/2 - upper[1], middle_z - upper[2]])
        upper_to_mid_norm = upper_to_mid / (np.linalg.norm(upper_to_mid) + 1e-6)
        
        # 중간에서 하위 척추로의 벡터 (시상면 YZ 투영)
        mid_to_lower = np.array([0, lower[1] - (upper[1] + lower[1])/2, lower[2] - middle_z])
        mid_to_lower_norm = mid_to_lower / (np.linalg.norm(mid_to_lower) + 1e-6)
        
        # 두 벡터의 각도
        cos_angle = np.clip(np.dot(upper_to_mid_norm, mid_to_lower_norm), -1, 1)
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        # 굽은 각도 = 180° - 내각
        cobb_angle = 180.0 - angle_deg
        
        return max(0, cobb_angle)  # 음수 방지
    
    def calculate_cobb_angle_proper(top_point, upper_mid, lower_mid, bottom_point):
        """
        의학적으로 정확한 Cobb angle: 4점을 사용한 계산
        상단 접선과 하단 접선이 이루는 각도
        
        Cobb angle = 두 접선이 이루는 각도 (0~180°)
        척추가 일직선이면 0°, 굽어있으면 양수값
        """
        top = np.array(top_point)
        upper = np.array(upper_mid)
        lower = np.array(lower_mid)
        bottom = np.array(bottom_point)
        
        # 상단 접선: top → upper 방향 (YZ 평면에 투영)
        upper_line = upper - top
        upper_line_yz = np.array([0, upper_line[1], upper_line[2]])
        upper_line_yz = upper_line_yz / (np.linalg.norm(upper_line_yz) + 1e-6)
        
        # 하단 접선: lower → bottom 방향 (YZ 평면에 투영)
        lower_line = bottom - lower
        lower_line_yz = np.array([0, lower_line[1], lower_line[2]])
        lower_line_yz = lower_line_yz / (np.linalg.norm(lower_line_yz) + 1e-6)
        
        # 두 접선의 각도
        cos_angle = np.clip(np.dot(upper_line_yz, lower_line_yz), -1, 1)
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        # Cobb angle = 두 접선이 이루는 각도 (그대로 사용)
        cobb = angle_deg
        
        return cobb
    
    # 경추 전만각 (Cervical Lordosis) - C2~C7
    # 의학 표준: 20-35° (정상), 10-20° or 35-45° (주의), <10° or >45° (비정상)
    try:
        c2 = skeleton_points['cervical_C2']
        c3 = skeleton_points['cervical_C3']
        c4 = skeleton_points['cervical_C4']
        c5 = skeleton_points['cervical_C5']
        c6 = skeleton_points['cervical_C6']
        c7 = skeleton_points['cervical_C7']
        
        # C2-C3을 상단 접선, C6-C7을 하단 접선으로 사용
        cervical_angle = calculate_cobb_angle_proper(c2, c3, c6, c7)
        angles['cervical_lordosis'] = cervical_angle
    except KeyError as e:
        print(f"  경추 각도 계산 실패: {e}")
        angles['cervical_lordosis'] = 0.0
    
    # 흉추 후만각 (Thoracic Kyphosis) - T1~T12
    # 의학 표준: 20-40° (정상), 15-20° or 40-55° (주의), <15° or >55° (비정상)
    try:
        t1 = skeleton_points['thoracic_T1']
        t2 = skeleton_points.get('thoracic_T2', t1)
        t11 = skeleton_points.get('thoracic_T11', None)
        t12 = skeleton_points['thoracic_T12']
        
        if t11 is None:
            t11 = t12
        
        # T1-T2를 상단 접선, T11-T12를 하단 접선으로 사용
        thoracic_angle = calculate_cobb_angle_proper(t1, t2, t11, t12)
        angles['thoracic_kyphosis'] = thoracic_angle
    except KeyError as e:
        print(f"  흉추 각도 계산 실패: {e}")
        angles['thoracic_kyphosis'] = 0.0
    
    # 요추 전만각 (Lumbar Lordosis) - L1~L5
    # 의학 표준: 40-60° (정상), 30-40° or 60-70° (주의), <30° or >70° (비정상)
    try:
        l1 = skeleton_points['lumbar_L1']
        l2 = skeleton_points.get('lumbar_L2', l1)
        l4 = skeleton_points.get('lumbar_L4', None)
        l5 = skeleton_points['lumbar_L5']
        
        if l4 is None:
            l4 = l5
        
        # L1-L2를 상단 접선, L4-L5를 하단 접선으로 사용
        lumbar_angle = calculate_cobb_angle_proper(l1, l2, l4, l5)
        angles['lumbar_lordosis'] = lumbar_angle
    except KeyError as e:
        print(f"  요추 각도 계산 실패: {e}")
        angles['lumbar_lordosis'] = 0.0
    
    # 어깨 수평도 (Shoulder Level)
    # 의학 표준: ≤2° (정상), 2-10° (주의), >10° (비정상)
    try:
        left_shoulder = np.array(skeleton_points['left_shoulder'])
        right_shoulder = np.array(skeleton_points['right_shoulder'])
        shoulder_vector = right_shoulder - left_shoulder
        
        # Y축(높이) 차이를 이용한 각도 계산
        height_diff = abs(shoulder_vector[1])
        horizontal_dist = math.sqrt(shoulder_vector[0]**2 + shoulder_vector[2]**2)
        shoulder_angle = math.degrees(math.atan2(height_diff, horizontal_dist))
        angles['shoulder_level'] = shoulder_angle
    except KeyError:
        angles['shoulder_level'] = 0.0
    
    # 골반 기울기 (Pelvic Tilt)
    # 의학 표준: ≤3° (정상), 3-10° (주의), >10° (비정상)
    try:
        left_hip = np.array(skeleton_points['left_hip'])
        right_hip = np.array(skeleton_points['right_hip'])
        pelvis_vector = right_hip - left_hip
        
        # Y축(높이) 차이를 이용한 각도 계산
        height_diff = abs(pelvis_vector[1])
        horizontal_dist = math.sqrt(pelvis_vector[0]**2 + pelvis_vector[2]**2)
        pelvis_angle = math.degrees(math.atan2(height_diff, horizontal_dist))
        angles['pelvis_tilt'] = pelvis_angle
    except KeyError:
        angles['pelvis_tilt'] = 0.0
    
    # 척추 정렬도 SVA (Sagittal Vertical Axis) - cm 단위
    # 의학 표준: <4cm (정상), 4-6cm (주의), >6cm (비정상)
    try:
        head_top = np.array(skeleton_points['head_top'])
        pelvis_center = np.array(skeleton_points['pelvis_center'])
        
        # X-Z 평면에서의 수평 거리 (앞뒤/좌우 방향)
        horizontal_offset = math.sqrt((head_top[0] - pelvis_center[0])**2 + 
                                     (head_top[2] - pelvis_center[2])**2)
        
        # mm를 cm로 변환 (스켈레톤 좌표가 mm 단위라고 가정)
        sva_cm = horizontal_offset / 10.0
        angles['spine_alignment'] = sva_cm
    except KeyError:
        angles['spine_alignment'] = 0.0
    
    return angles


def create_skeleton_visualization(skeleton_points):
    """
    스켈레톤 시각화를 위한 라인과 포인트를 생성합니다.
    
    Args:
        skeleton_points (dict): 스켈레톤 포인트 딕셔너리
        
    Returns:
        tuple: (skeleton point cloud, cylinder list)
    """
    # 스켈레톤 포인트들을 Open3D 포인트 클라우드로 변환
    skeleton_pcd = o3d.geometry.PointCloud()
    points = []
    colors = []
    
    # 각 부위별로 다른 색상 적용
    color_map = {
        'head': [1, 0, 1],      # 마젠타
        'cervical': [0, 1, 1],  # 시안 (경추)
        'thoracic': [1, 1, 0],  # 노랑 (흉추)
        'lumbar': [1, 0.5, 0],  # 주황 (요추)
        'shoulder': [0, 1, 0],  # 초록 (어깨)
        'pelvis': [1, 0, 0],    # 빨강 (골반)
        'sacrum': [0.5, 0, 0.5] # 보라 (천추)
    }
    
    for name, point in skeleton_points.items():
        points.append(point)
        
        # 부위별 색상 지정
        if 'head' in name or 'neck' in name:
            colors.append(color_map['head'])
        elif 'cervical' in name:
            colors.append(color_map['cervical'])
        elif 'thoracic' in name:
            colors.append(color_map['thoracic'])
        elif 'lumbar' in name:
            colors.append(color_map['lumbar'])
        elif 'shoulder' in name:
            colors.append(color_map['shoulder'])
        elif 'hip' in name or 'pelvis' in name:
            colors.append(color_map['pelvis'])
        else:
            colors.append(color_map['sacrum'])
    
    skeleton_pcd.points = o3d.utility.Vector3dVector(np.array(points))
    skeleton_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # 스켈레톤 연결선 생성 - 굵은 실린더로 대체
    cylinders = []
    
    def create_cylinder_between_points(p1, p2, radius=2.0, color=[1, 1, 1]):
        """두 점 사이에 실린더를 생성합니다."""
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        # 실린더의 높이와 방향 계산
        height = np.linalg.norm(p2 - p1)
        if height < 0.1:  # 너무 짧으면 None 반환
            return None
            
        # 실린더 생성
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        
        # 실린더를 두 점 사이에 정렬
        center = (p1 + p2) / 2
        
        # Z축이 두 점을 잇는 방향이 되도록 회전
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)
        
        # Z축과 방향 벡터 사이의 회전 계산
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            pass  # 회전 필요 없음
        elif np.allclose(direction, -z_axis):
            # 180도 회전
            cylinder = cylinder.rotate(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), center=(0, 0, 0))
        else:
            # 임의의 회전
            axis = np.cross(z_axis, direction)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(z_axis, direction))
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            cylinder = cylinder.rotate(rotation_matrix, center=(0, 0, 0))
        
        # 중심점으로 이동
        cylinder.translate(center)
        
        # 색상 적용
        cylinder.paint_uniform_color(color)
        
        return cylinder
    
    # 척추 연결 (경추) - 시안색
    for i in range(6):
        key1 = f'cervical_C{i+1}'
        key2 = f'cervical_C{i+2}'
        if key1 in skeleton_points and key2 in skeleton_points:
            cylinder = create_cylinder_between_points(
                skeleton_points[key1], skeleton_points[key2], radius=1.5, color=[0, 1, 1]
            )
            if cylinder:
                cylinders.append(cylinder)
    
    # 척추 연결 (흉추) - 노란색
    for i in range(11):
        key1 = f'thoracic_T{i+1}'
        key2 = f'thoracic_T{i+2}'
        if key1 in skeleton_points and key2 in skeleton_points:
            cylinder = create_cylinder_between_points(
                skeleton_points[key1], skeleton_points[key2], radius=1.5, color=[1, 1, 0]
            )
            if cylinder:
                cylinders.append(cylinder)
    
    # 척추 연결 (요추) - 주황색
    for i in range(4):
        key1 = f'lumbar_L{i+1}'
        key2 = f'lumbar_L{i+2}'
        if key1 in skeleton_points and key2 in skeleton_points:
            cylinder = create_cylinder_between_points(
                skeleton_points[key1], skeleton_points[key2], radius=1.5, color=[1, 0.5, 0]
            )
            if cylinder:
                cylinders.append(cylinder)
    
    # 경추-흉추 연결
    if 'cervical_C7' in skeleton_points and 'thoracic_T1' in skeleton_points:
        cylinder = create_cylinder_between_points(
            skeleton_points['cervical_C7'], 
            skeleton_points['thoracic_T1'], 
            radius=1.7, 
            color=[0.5, 1, 0.5]
        )
        if cylinder:
            cylinders.append(cylinder)
    
    # 흉추-요추 연결
    if 'thoracic_T12' in skeleton_points and 'lumbar_L1' in skeleton_points:
        cylinder = create_cylinder_between_points(
            skeleton_points['thoracic_T12'], 
            skeleton_points['lumbar_L1'], 
            radius=1.5, 
            color=[1, 0.75, 0]
        )
        if cylinder:
            cylinders.append(cylinder)
    
    # 어깨 연결 - 초록색
    if all(key in skeleton_points for key in ['left_shoulder', 'shoulder_center', 'right_shoulder']):
        cylinder1 = create_cylinder_between_points(
            skeleton_points['left_shoulder'], 
            skeleton_points['shoulder_center'], 
            radius=1.5,
            color=[0, 1, 0]
        )
        cylinder2 = create_cylinder_between_points(
            skeleton_points['shoulder_center'], 
            skeleton_points['right_shoulder'], 
            radius=1.5,
            color=[0, 1, 0]
        )
        if cylinder1:
            cylinders.append(cylinder1)
        if cylinder2:
            cylinders.append(cylinder2)
    
    # 골반 연결 - 빨간색
    if all(key in skeleton_points for key in ['left_hip', 'pelvis_center', 'right_hip']):
        cylinder1 = create_cylinder_between_points(
            skeleton_points['left_hip'], 
            skeleton_points['pelvis_center'], 
            radius=1.5,
            color=[1, 0, 0]
        )
        cylinder2 = create_cylinder_between_points(
            skeleton_points['pelvis_center'], 
            skeleton_points['right_hip'], 
            radius=1.5,
            color=[1, 0, 0]
        )
        if cylinder1:
            cylinders.append(cylinder1)
        if cylinder2:
            cylinders.append(cylinder2)
    
    # 척추 중심선 연결 (목-어깨, 요추-골반)
    if 'neck' in skeleton_points and 'shoulder_center' in skeleton_points:
        cylinder = create_cylinder_between_points(
            skeleton_points['neck'], 
            skeleton_points['shoulder_center'], 
            radius=1.5, 
            color=[1, 0, 1]
        )
        if cylinder:
            cylinders.append(cylinder)
    
    if 'lumbar_L5' in skeleton_points and 'pelvis_center' in skeleton_points:
        cylinder = create_cylinder_between_points(
            skeleton_points['lumbar_L5'], 
            skeleton_points['pelvis_center'], 
            radius=2.0, 
            color=[0.8, 0.2, 0.8]
        )
        if cylinder:
            cylinders.append(cylinder)
    
    return skeleton_pcd, cylinders


def print_angles(angles):
    """
    계산된 각도들을 의학 표준에 따라 출력합니다.
    Table 4 기준: 정상(Normal), 주의(Caution), 비정상(Abnormal)
    
    Args:
        angles (dict): 각도 분석 결과 딕셔너리
    """
    print("\n" + "="*60)
    print("           인체 자세 분석 결과 (의학 표준 기준)")
    print("="*60)
    
    def evaluate_metric(value, normal_range, caution_range, unit="°"):
        """지표를 평가하여 상태와 색상을 반환"""
        if unit == "cm":
            status = "✅ 정상"
            if value > caution_range[1]:
                status = "⚠️  주의"
            if value > normal_range[1]:
                status = "❌ 비정상"
        else:
            status = "✅ 정상"
            if value < caution_range[0] or value > caution_range[1]:
                status = "⚠️  주의"
            if value < normal_range[0] or value > normal_range[1]:
                status = "❌ 비정상"
        return status
    
    print(f"\n척추 각도 분석 (Cobb Angle):")
    
    # 경추 전만각 (Cervical Lordosis)
    cervical = angles['cervical_lordosis']
    cervical_status = evaluate_metric(cervical, (10, 45), (20, 35))
    print(f"   • 경추 전만각 (Cervical Lordosis): {cervical:.1f}° {cervical_status}")
    print(f"     - 정상(Normal): 20-35° | 주의(Caution): 10-20° or 35-45°")
    print(f"     - 비정상(Abnormal): <10° or >45°")
    
    # 흉추 후만각 (Thoracic Kyphosis)
    thoracic = angles['thoracic_kyphosis']
    thoracic_status = evaluate_metric(thoracic, (15, 55), (20, 40))
    print(f"\n   • 흉추 후만각 (Thoracic Kyphosis): {thoracic:.1f}° {thoracic_status}")
    print(f"     - 정상(Normal): 20-40° | 주의(Caution): 15-20° or 40-55°")
    print(f"     - 비정상(Abnormal): <15° or >55°")
    
    # 요추 전만각 (Lumbar Lordosis)
    lumbar = angles['lumbar_lordosis']
    lumbar_status = evaluate_metric(lumbar, (30, 70), (40, 60))
    print(f"\n   • 요추 전만각 (Lumbar Lordosis): {lumbar:.1f}° {lumbar_status}")
    print(f"     - 정상(Normal): 40-60° | 주의(Caution): 30-40° or 60-70°")
    print(f"     - 비정상(Abnormal): <30° or >70°")
    
    print(f"\n어깨 및 골반 분석:")
    
    # 어깨 수평도 (Shoulder Level)
    shoulder = angles['shoulder_level']
    shoulder_status = evaluate_metric(shoulder, (0, 10), (0, 2))
    print(f"   • 어깨 수평도 (Shoulder Level): {shoulder:.1f}° {shoulder_status}")
    print(f"     - 정상(Normal): ≤2° | 주의(Caution): 2-10° | 비정상(Abnormal): >10°")
    
    # 골반 기울기 (Pelvic Tilt)
    pelvis = angles['pelvis_tilt']
    pelvis_status = evaluate_metric(pelvis, (0, 10), (0, 3))
    print(f"\n   • 골반 기울기 (Pelvic Tilt): {pelvis:.1f}° {pelvis_status}")
    print(f"     - 정상(Normal): ≤3° | 주의(Caution): 3-10° | 비정상(Abnormal): >10°")
    
    print(f"\n전체 척추 정렬:")
    
    # 척추 정렬도 SVA (Sagittal Vertical Axis)
    sva = angles['spine_alignment']
    sva_status = evaluate_metric(sva, (0, 6), (0, 4), unit="cm")
    print(f"   • 척추 정렬도 SVA (Sagittal Vertical Axis): {sva:.1f}cm {sva_status}")
    print(f"     - 정상(Normal): <4cm | 주의(Caution): 4-6cm | 비정상(Abnormal): >6cm")
    
    # 자세 평가 요약
    print(f"\n자세 평가 요약:")
    issues = []
    
    # 경추 평가
    if cervical < 10 or cervical > 45:
        if cervical < 10:
            issues.append(f"❌ 경추 전만 부족 ({cervical:.1f}°) - 거북목 증후군 의심")
        else:
            issues.append(f"❌ 경추 과전만 ({cervical:.1f}°)")
    elif cervical < 20 or cervical > 35:
        if cervical < 20:
            issues.append(f"⚠️  경추 전만 약간 부족 ({cervical:.1f}°)")
        else:
            issues.append(f"⚠️  경추 전만 약간 과도 ({cervical:.1f}°)")
    
    # 흉추 평가
    if thoracic < 15 or thoracic > 55:
        if thoracic > 55:
            issues.append(f"❌ 흉추 과후만 ({thoracic:.1f}°) - 라운드 숄더 의심")
        else:
            issues.append(f"❌ 흉추 후만 부족 ({thoracic:.1f}°)")
    elif thoracic < 20 or thoracic > 40:
        if thoracic > 40:
            issues.append(f"⚠️  흉추 후만 약간 과도 ({thoracic:.1f}°)")
        else:
            issues.append(f"⚠️  흉추 후만 약간 부족 ({thoracic:.1f}°)")
    
    # 요추 평가
    if lumbar < 30 or lumbar > 70:
        if lumbar < 30:
            issues.append(f"❌ 요추 전만 부족 ({lumbar:.1f}°) - 평평한 허리")
        else:
            issues.append(f"❌ 요추 과전만 ({lumbar:.1f}°)")
    elif lumbar < 40 or lumbar > 60:
        if lumbar < 40:
            issues.append(f"⚠️  요추 전만 약간 부족 ({lumbar:.1f}°)")
        else:
            issues.append(f"⚠️  요추 전만 약간 과도 ({lumbar:.1f}°)")
    
    # 어깨 평가
    if shoulder > 10:
        issues.append(f"❌ 어깨 불균형 심각 ({shoulder:.1f}°)")
    elif shoulder > 2:
        issues.append(f"⚠️  어깨 약간 불균형 ({shoulder:.1f}°)")
    
    # 골반 평가
    if pelvis > 10:
        issues.append(f"❌ 골반 기울기 심각 ({pelvis:.1f}°)")
    elif pelvis > 3:
        issues.append(f"⚠️  골반 약간 기울어짐 ({pelvis:.1f}°)")
    
    # SVA 평가
    if sva > 6:
        issues.append(f"❌ 척추 정렬 심각 ({sva:.1f}cm) - 전방/후방 이동 과다")
    elif sva > 4:
        issues.append(f"⚠️  척추 약간 정렬 불량 ({sva:.1f}cm)")
    
    if issues:
        print(f"\n   발견된 문제점:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"   ✅ 모든 지표가 정상 범위입니다! 우수한 자세입니다.")
    
    print("="*60 + "\n")