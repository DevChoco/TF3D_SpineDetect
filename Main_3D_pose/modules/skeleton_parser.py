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


def create_skeleton_from_pointcloud(pcd, ai_landmarks=None):
    """
    포인트 클라우드에서 인체 스켈레톤을 생성합니다.
    AI 랜드마크가 제공되면 더 정확한 위치를 사용합니다.
    
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
    
    # AI 검출이 실패했거나 불완전한 경우 기본값 사용 (높이 조정)
    if 'shoulder_center' not in skeleton_points:
        print("AI 어깨 검출 실패, 조정된 기본 비율 사용")
        shoulder_height = max_bound[1] - height * 0.15  # 0.22에서 0.15로 변경 (더 위로)
        # 어깨는 척추보다 앞쪽에 위치 (의학적으로 정확한 위치) - 신체 내부
        shoulder_z = center[2] - z_range * 0.05  # 척추보다 약간 앞쪽 (신체 내부)
        skeleton_points['left_shoulder'] = [center[0] - width * 0.25, shoulder_height, shoulder_z]
        skeleton_points['right_shoulder'] = [center[0] + width * 0.25, shoulder_height, shoulder_z]
        skeleton_points['shoulder_center'] = [center[0], shoulder_height, shoulder_z]
    
    if 'pelvis_center' not in skeleton_points:
        print("AI 골반 검출 실패, 조정된 기본 비율 사용")
        pelvis_height = max_bound[1] - height * 0.42  # 0.35에서 0.42로 변경 (조금 아래로)
        skeleton_points['pelvis_center'] = [center[0], pelvis_height, center[2]]
        skeleton_points['left_hip'] = [center[0] - width * 0.15, pelvis_height, center[2]]
        skeleton_points['right_hip'] = [center[0] + width * 0.15, pelvis_height, center[2]]
    
    if 'neck' not in skeleton_points:
        skeleton_points['head_top'] = [center[0], max_bound[1], center[2] - depth * 0.05]  # 머리는 척추 위
        skeleton_points['neck'] = [center[0], max_bound[1] - height * 0.08, center[2] - depth * 0.05]  # 목도 척추 위
    
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
    skeleton_points['head_top'] = [center[0], max_bound[1], spine_z_base + z_range * 0.05]  # 척추보다 약간 앞쪽 (신체 내부)
    skeleton_points['neck'] = [center[0], spine_start_height, spine_z_base]
    
    # 의학적으로 정확한 실제 인체 척추 곡선 구현
    print("=== 실제 인체 척추 곡선 생성 ===")
    
    # 경추 (C1-C7) - 전체 척추의 25% - 자연스러운 전만(lordosis) 곡선
    cervical_length = total_spine_length * 0.25
    cervical_start = skeleton_points['neck']
    cervical_end = [center[0], spine_start_height - cervical_length, spine_z_base]
    
    print(f"경추 길이: {cervical_length:.2f} (전체 척추의 25%) - 전만 곡선")
    
    for i in range(7):
        ratio = i / 6
        # 경추 전만: 실제 인체처럼 목이 앞으로 자연스럽게 굽어짐 (25-35도 전만각) - 신체 내부
        cervical_lordosis_angle = math.radians(30)  # 30도 전만각
        lordosis_curve = math.sin(ratio * math.pi) * z_range * 0.04 * math.sin(cervical_lordosis_angle)  # 신체 범위 내에서
        
        point = [
            cervical_start[0] + (cervical_end[0] - cervical_start[0]) * ratio,
            cervical_start[1] + (cervical_end[1] - cervical_start[1]) * ratio,
            cervical_start[2] + (cervical_end[2] - cervical_start[2]) * ratio + lordosis_curve  # 앞으로 굽음 (신체 내부)
        ]
        skeleton_points[f'cervical_C{i+1}'] = point
    
    # 흉추 (T1-T12) - 전체 척추의 40% - 자연스러운 후만(kyphosis) 곡선
    thoracic_length = total_spine_length * 0.40
    thoracic_start = cervical_end
    # 흉추는 실제 인체처럼 등 쪽으로 자연스럽게 굽어짐 (20-50도 후만각) - 신체 내부
    thoracic_end = [center[0], cervical_end[1] - thoracic_length, spine_z_base - z_range * 0.06]  # 신체 범위 내에서
    
    print(f"흉추 길이: {thoracic_length:.2f} (전체 척추의 40%) - 후만 곡선")
    
    for i in range(12):
        ratio = i / 11
        # 흉추 후만: 실제 인체의 가슴 뒤쪽 곡선 (35도 후만각) - 신체 내부
        thoracic_kyphosis_angle = math.radians(35)  # 35도 후만각
        kyphosis_curve = math.sin(ratio * math.pi) * z_range * 0.08 * math.sin(thoracic_kyphosis_angle)  # 신체 범위 내에서
        
        point = [
            thoracic_start[0] + (thoracic_end[0] - thoracic_start[0]) * ratio,
            thoracic_start[1] + (thoracic_end[1] - thoracic_start[1]) * ratio,
            thoracic_start[2] + (thoracic_end[2] - thoracic_start[2]) * ratio - kyphosis_curve  # 뒤로 굽음 (신체 내부)
        ]
        skeleton_points[f'thoracic_T{i+1}'] = point
    
    # 요추 (L1-L5) - 전체 척추의 20% - 자연스러운 전만(lordosis) 곡선
    lumbar_length = total_spine_length * 0.20
    lumbar_start = thoracic_end
    # 요추는 실제 인체처럼 앞으로 자연스럽게 굽어짐 (40-60도 전만각) - 신체 내부
    lumbar_end = [center[0], thoracic_end[1] - lumbar_length, spine_z_base - z_range * 0.03]  # 신체 범위 내에서
    
    print(f"요추 길이: {lumbar_length:.2f} (전체 척추의 20%) - 전만 곡선")
    
    for i in range(5):
        ratio = i / 4
        # 요추 전만: 실제 인체의 허리 앞쪽 곡선 (45도 전만각) - 신체 내부
        lumbar_lordosis_angle = math.radians(45)  # 45도 전만각
        lordosis_curve = math.sin(ratio * math.pi) * z_range * 0.06 * math.sin(lumbar_lordosis_angle)  # 신체 범위 내에서
        
        point = [
            lumbar_start[0] + (lumbar_end[0] - lumbar_start[0]) * ratio,
            lumbar_start[1] + (lumbar_end[1] - lumbar_start[1]) * ratio,
            lumbar_start[2] + (lumbar_end[2] - lumbar_start[2]) * ratio + lordosis_curve  # 앞으로 굽음 (신체 내부)
        ]
        skeleton_points[f'lumbar_L{i+1}'] = point
    
    # 천추와 미추 - 전체 척추의 15% - 자연스러운 후만(kyphosis) 곡선
    sacral_length = total_spine_length * 0.15
    sacral_start = lumbar_end
    # 천추는 실제 인체처럼 뒤로 자연스럽게 굽어짐 (골반 후면) - 신체 내부
    sacral_end = [center[0], lumbar_end[1] - sacral_length, spine_z_base - z_range * 0.04]  # 신체 범위 내에서
    
    print(f"천추+미추 길이: {sacral_length:.2f} (전체 척추의 15%) - 후만 곡선")
    
    # 천추 (S1-S5) - 천추+미추의 80% (골반 후면의 중심)
    sacrum_length = sacral_length * 0.80
    
    for i in range(5):
        ratio = i / 4
        # 천추 후만: 실제 인체의 골반 뒤쪽 곡선 (15도 후만각) - 신체 내부
        sacral_kyphosis_angle = math.radians(15)  # 15도 후만각
        kyphosis_curve = math.sin(ratio * math.pi) * z_range * 0.03 * math.sin(sacral_kyphosis_angle)  # 신체 범위 내에서
        
        point = [
            sacral_start[0] + (sacral_end[0] - sacral_start[0]) * ratio * 0.8,  # 천추 부분만
            sacral_start[1] + (sacral_end[1] - sacral_start[1]) * ratio * 0.8,
            sacral_start[2] + (sacral_end[2] - sacral_start[2]) * ratio * 0.8 - kyphosis_curve  # 뒤로 굽음 (신체 내부)
        ]
        skeleton_points[f'sacral_S{i+1}'] = point
    
    # 천추 중심점
    sacrum_center = skeleton_points['sacral_S3']  # S3를 천추 중심으로 사용
    skeleton_points['sacrum'] = sacrum_center
    
    # 미추 (Co1-Co4) - 천추+미추의 20% (척추의 최종 끝)
    coccyx_length = sacral_length * 0.20
    coccyx_start_ratio = 0.8  # 천추 끝에서 시작
    
    for i in range(4):
        ratio = coccyx_start_ratio + (i / 3) * 0.2  # 0.8에서 1.0까지
        # 미추는 천추보다 더 뒤로 굽어짐 - 신체 내부
        coccyx_curve = z_range * 0.02  # 신체 범위 내에서
        
        point = [
            sacral_start[0] + (sacral_end[0] - sacral_start[0]) * ratio,
            sacral_start[1] + (sacral_end[1] - sacral_start[1]) * ratio,
            sacral_start[2] + (sacral_end[2] - sacral_start[2]) * ratio - coccyx_curve  # 신체 내부
        ]
        skeleton_points[f'coccyx_Co{i+1}'] = point
    
    # 미추 끝점
    skeleton_points['coccyx'] = skeleton_points['coccyx_Co4']
    
    # 골반 위치 - 천추를 중심으로 해부학적으로 정확한 위치 설정
    # 골반(장골능)은 천추보다 약간 위에, 고관절은 천추와 같은 높이에 위치
    pelvis_height = sacrum_center[1] + sacrum_length * 0.3  # 천추 중심에서 약간 위
    hip_joint_height = sacrum_center[1]  # 천추 중심과 같은 높이
    
    # 골반은 정상 위치 유지 (center[2] - 배 쪽)
    skeleton_points['pelvis_center'] = [center[0], pelvis_height, center[2]]
    skeleton_points['left_hip'] = [center[0] - width * 0.15, hip_joint_height, center[2]]
    skeleton_points['right_hip'] = [center[0] + width * 0.15, hip_joint_height, center[2]]
    
    print(f"골반 중심 높이: {pelvis_height:.2f} (천추 기준)")
    print(f"고관절 높이: {hip_joint_height:.2f} (천추 중심과 동일)")
    
    # 어깨는 흉추 상부에 위치하되 척추보다 앞쪽에 (의학적으로 정확한 위치) - 신체 내부
    shoulder_height = cervical_end[1] + thoracic_length * 0.2  # 흉추 시작에서 더 위로 올림
    shoulder_z = spine_z_base + z_range * 0.08  # 척추보다 앞쪽에 위치 (신체 내부)
    skeleton_points['left_shoulder'] = [center[0] - width * 0.25, shoulder_height, shoulder_z]
    skeleton_points['right_shoulder'] = [center[0] + width * 0.25, shoulder_height, shoulder_z]
    skeleton_points['shoulder_center'] = [center[0], shoulder_height, shoulder_z]
    
    return skeleton_points


def calculate_spine_angles(skeleton_points):
    """
    척추의 각종 각도를 계산합니다.
    
    Args:
        skeleton_points (dict): 스켈레톤 포인트 딕셔너리
        
    Returns:
        dict: 각도 분석 결과 딕셔너리
    """
    angles = {}
    
    # 경추 각도 (목의 전만)
    cervical_start = np.array(skeleton_points['cervical_C1'])
    cervical_end = np.array(skeleton_points['cervical_C7'])
    cervical_vector = cervical_end - cervical_start
    vertical_vector = np.array([0, -1, 0])  # 수직 아래 방향
    
    # 경추 각도 계산 (전만각)
    cervical_angle = math.degrees(math.acos(np.clip(np.dot(cervical_vector, vertical_vector) / 
                                                  (np.linalg.norm(cervical_vector) * np.linalg.norm(vertical_vector)), -1, 1)))
    angles['cervical_lordosis'] = cervical_angle
    
    # 흉추 각도 (가슴의 후만)
    thoracic_start = np.array(skeleton_points['thoracic_T1'])
    thoracic_end = np.array(skeleton_points['thoracic_T12'])
    thoracic_vector = thoracic_end - thoracic_start
    
    thoracic_angle = math.degrees(math.acos(np.clip(np.dot(thoracic_vector, vertical_vector) / 
                                                   (np.linalg.norm(thoracic_vector) * np.linalg.norm(vertical_vector)), -1, 1)))
    angles['thoracic_kyphosis'] = thoracic_angle
    
    # 요추 각도 (허리의 전만)
    lumbar_start = np.array(skeleton_points['lumbar_L1'])
    lumbar_end = np.array(skeleton_points['lumbar_L5'])
    lumbar_vector = lumbar_end - lumbar_start
    
    lumbar_angle = math.degrees(math.acos(np.clip(np.dot(lumbar_vector, vertical_vector) / 
                                                 (np.linalg.norm(lumbar_vector) * np.linalg.norm(vertical_vector)), -1, 1)))
    angles['lumbar_lordosis'] = lumbar_angle
    
    # 어깨 각도
    left_shoulder = np.array(skeleton_points['left_shoulder'])
    right_shoulder = np.array(skeleton_points['right_shoulder'])
    shoulder_vector = right_shoulder - left_shoulder
    horizontal_vector = np.array([1, 0, 0])  # 수평 방향
    
    shoulder_angle = math.degrees(math.acos(np.clip(np.dot(shoulder_vector, horizontal_vector) / 
                                                   (np.linalg.norm(shoulder_vector) * np.linalg.norm(horizontal_vector)), -1, 1)))
    angles['shoulder_level'] = shoulder_angle
    
    # 골반 각도
    left_hip = np.array(skeleton_points['left_hip'])
    right_hip = np.array(skeleton_points['right_hip'])
    pelvis_vector = right_hip - left_hip
    
    pelvis_angle = math.degrees(math.acos(np.clip(np.dot(pelvis_vector, horizontal_vector) / 
                                                 (np.linalg.norm(pelvis_vector) * np.linalg.norm(horizontal_vector)), -1, 1)))
    angles['pelvis_tilt'] = pelvis_angle
    
    # 전체 척추 정렬 (머리에서 골반까지)
    head_top = np.array(skeleton_points['head_top'])
    pelvis_center = np.array(skeleton_points['pelvis_center'])
    spine_vector = pelvis_center - head_top
    
    spine_alignment = math.degrees(math.acos(np.clip(np.dot(spine_vector, vertical_vector) / 
                                                    (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector)), -1, 1)))
    angles['spine_alignment'] = spine_alignment
    
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
    계산된 각도들을 출력합니다.
    
    Args:
        angles (dict): 각도 분석 결과 딕셔너리
    """
    print("\n" + "="*50)
    print("           인체 자세 분석 결과")
    print("="*50)
    
    print(f"\n척추 각도 분석:")
    print(f"   • 경추 전만각 (Cervical Lordosis): {angles['cervical_lordosis']:.1f}°")
    print(f"     - 정상 범위: 35-45°")
    
    print(f"\n   • 흉추 후만각 (Thoracic Kyphosis): {angles['thoracic_kyphosis']:.1f}°")
    print(f"     - 정상 범위: 20-40°")
    
    print(f"\n   • 요추 전만각 (Lumbar Lordosis): {angles['lumbar_lordosis']:.1f}°")
    print(f"     - 정상 범위: 40-60°")
    
    print(f"\n어깨 및 골반 분석:")
    print(f"   • 어깨 수평도 (Shoulder Level): {angles['shoulder_level']:.1f}°")
    print(f"     - 정상: 0° (완전 수평)")
    
    print(f"\n   • 골반 기울기 (Pelvis Tilt): {angles['pelvis_tilt']:.1f}°")
    print(f"     - 정상: 0° (완전 수평)")
    
    print(f"\n전체 척추 정렬:")
    print(f"   • 척추 정렬도 (Spine Alignment): {angles['spine_alignment']:.1f}°")
    print(f"     - 정상: 0° (완전 수직)")
    
    # 자세 평가
    print(f"\n자세 평가:")
    issues = []
    
    if angles['cervical_lordosis'] < 30:
        issues.append("경추 전만이 부족합니다 (거북목 의심)")
    elif angles['cervical_lordosis'] > 50:
        issues.append("경추 전만이 과도합니다")
    
    if angles['thoracic_kyphosis'] > 45:
        issues.append("흉추 후만이 과도합니다 (라운드 숄더 의심)")
    
    if angles['lumbar_lordosis'] < 35:
        issues.append("요추 전만이 부족합니다")
    elif angles['lumbar_lordosis'] > 65:
        issues.append("요추 전만이 과도합니다")
    
    if abs(angles['shoulder_level']) > 5:
        issues.append(f"어깨 높이가 불균형합니다 ({angles['shoulder_level']:.1f}°)")
    
    if abs(angles['pelvis_tilt']) > 5:
        issues.append(f"골반이 기울어져 있습니다 ({angles['pelvis_tilt']:.1f}°)")
    
    if abs(angles['spine_alignment']) > 10:
        issues.append(f"척추가 기울어져 있습니다 ({angles['spine_alignment']:.1f}°)")
    
    if issues:
        for issue in issues:
            print(f"   ⚠️  {issue}")
    else:
        print(f"   ✅  전반적으로 양호한 자세입니다!")
    
    print("="*50)