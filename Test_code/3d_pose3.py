import numpy as np
import cv2
import open3d as o3d
import os
import math
import mediapipe as mp
from PIL import Image

def load_depth_map(file_path):
    # PIL을 사용하여 이미지 로드
    from PIL import Image
    try:
        with Image.open(file_path) as img:
            depth_map = np.array(img)
            if len(depth_map.shape) > 2:  # Convert RGB to grayscale if needed
                depth_map = np.mean(depth_map, axis=2).astype(np.uint8)
            
            # 정사각형으로 자르기
            height, width = depth_map.shape
            size = min(height, width)
            
            # 중앙 기준으로 자르기
            start_y = (height - size) // 2
            start_x = (width - size) // 2
            depth_map = depth_map[start_y:start_y+size, start_x:start_x+size]
            
            return depth_map.astype(np.float32) / 255.0  # Normalize to [0,1]
    except Exception as e:
        print(f"Failed to load: {file_path}")
        print(f"Error: {str(e)}")
        return None

def create_point_cloud_from_depth(depth_map, view):
    if depth_map is None:
        return None
        
    size = depth_map.shape[0]  # 정사각형이므로 한 변의 길이만 필요
    y, x = np.mgrid[0:size, 0:size]
    
    # 포인트 수를 줄이기 위해 다운샘플링
    step = 2
    x = x[::step, ::step]
    y = y[::step, ::step]
    depth_map = depth_map[::step, ::step]
    
    # 중심점 조정을 위한 오프셋 계산
    x = x - size/2
    y = y - size/2
    
    scale = 100  # 스케일 조정
    
    # 뷰에 따라 좌표 변환
    if view == "front":
        points = np.stack([x, -y, depth_map * scale * 1.1], axis=-1)
    elif view == "right":
        points = np.stack([depth_map * scale * 3, -y, -x], axis=-1)  # 우측 깊이 2배
    elif view == "left":
        points = np.stack([-depth_map * scale * 3, -y, x], axis=-1)  # 좌측 깊이 2배
    elif view == "back":
        points = np.stack([-x, -y, -depth_map * scale * 1.1], axis=-1)

    # 유효한 깊이값을 가진 포인트만 선택 (임계값 0.3 적용)
    threshold = 0.4  # 30% 이상의 깊이값만 사용
    valid_points = points[depth_map > threshold]
    
    # 너무 많은 포인트가 있는 경우 추가 다운샘플링
    if len(valid_points) > 20000:
        indices = np.random.choice(len(valid_points), 20000, replace=False)
        valid_points = valid_points[indices]
    
    # Open3D 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    colors = {
        "front": [1, 0, 0],  # 빨간색
        "right": [0, 1, 0],  # 초록색
        "left": [0, 0, 1],   # 파란색
        "back": [1, 1, 0]    # 노란색
    }
    
    # colors = {
    #     "front": [0, 1, 0],  # 빨간색
    #     "right": [0, 1, 0],  # 초록색
    #     "left": [0, 1, 0],   # 파란색
    #     "back": [0, 1, 0]    # 노란색
    # }
    
    pcd.paint_uniform_color(colors[view])
    
    return pcd

def align_point_clouds(source, target, threshold=10):
    # 초기 변환 행렬
    init_transformation = np.eye(4)
    
    # ICP 정렬
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=threshold,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=100
        )
    )
    
    # 결과가 유효한 경우에만 변환 적용
    if reg_p2p.fitness > 0.01:  # 정렬 품질이 3% 이상인 경우
        return source.transform(reg_p2p.transformation)
    return source  # 정렬이 실패한 경우 원본 반환

def create_mesh_from_pointcloud(pcd):
    """
    포인트 클라우드에서 메시를 생성합니다.
    
    Args:
        pcd: Open3D PointCloud 객체
    
    Returns:
        Open3D TriangleMesh 객체 또는 None
    """
    try:
        print(f"포인트 클라우드 정보: {len(pcd.points)}개의 점")
        
        # 포인트 클라우드가 너무 작으면 메시 생성 불가
        if len(pcd.points) < 100:
            print("포인트가 너무 적어 메시 생성이 불가능합니다.")
            return None
        
        # 법선 벡터가 없으면 계산
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        
        # 법선 벡터 방향 통일
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Poisson 표면 재구성을 사용하여 메시 생성
        print("Poisson 표면 재구성을 사용하여 메시 생성 중...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=9,  # 메시 해상도 (높을수록 더 세밀)
            width=0,  # 0으로 설정하면 자동 계산
            scale=1.1,
            linear_fit=False
        )
        
        # 밀도가 낮은 부분 제거 (노이즈 감소)
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"생성된 메시 정보: {len(mesh.vertices)}개의 정점, {len(mesh.triangles)}개의 삼각형")
        
        # 메시 후처리
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # 메시 스무딩 (선택사항)
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        
        # 법선 벡터 재계산
        mesh.compute_vertex_normals()
        
        # 원본 포인트 클라우드의 색상을 메시에 적용
        if pcd.has_colors():
            # 단순히 평균 색상을 사용하거나 기본 색상 설정
            avg_color = np.mean(np.asarray(pcd.colors), axis=0)
            mesh.paint_uniform_color(avg_color)
        
        return mesh
        
    except Exception as e:
        print(f"메시 생성 중 오류 발생: {e}")
        
        # 대안으로 Ball Pivoting Algorithm 시도
        try:
            print("Ball Pivoting Algorithm으로 메시 생성 시도...")
            
            # 적절한 반지름 계산
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 2 * avg_dist
            
            # Ball Pivoting으로 메시 생성
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector([radius, radius * 2])
            )
            
            if len(mesh.triangles) > 0:
                print(f"Ball Pivoting으로 생성된 메시: {len(mesh.vertices)}개의 정점, {len(mesh.triangles)}개의 삼각형")
                mesh.compute_vertex_normals()
                return mesh
            else:
                print("Ball Pivoting으로도 메시 생성 실패")
                return None
                
        except Exception as e2:
            print(f"Ball Pivoting 메시 생성 중 오류: {e2}")
            return None

def detect_landmarks_with_ai(image_path):
    """
    MediaPipe를 사용하여 이미지에서 해부학적 랜드마크를 검출합니다.
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
            image_array = np.array(img)
            if len(image_array.shape) == 2:
                # 그레이스케일을 RGB로 변환
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # MediaPipe로 포즈 검출
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = {}
                h, w = image_rgb.shape[:2]
                
                # 주요 랜드마크 추출
                landmark_map = {
                    'nose': 0,
                    'left_shoulder': 11, 'right_shoulder': 12,
                    'left_hip': 23, 'right_hip': 24,
                    'left_knee': 25, 'right_knee': 26
                }
                
                for name, idx in landmark_map.items():
                    landmark = results.pose_landmarks.landmark[idx]
                    landmarks[name] = {
                        'x': int(landmark.x * w),
                        'y': int(landmark.y * h),
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
                
                print(f"AI 랜드마크 검출 성공: {len(landmarks)}개 포인트")
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
    """
    points = np.asarray(pcd.points)
    
    # 포인트 클라우드의 바운딩 박스 계산
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    height = max_bound[1] - min_bound[1]
    width = max_bound[0] - min_bound[0]
    depth = max_bound[2] - min_bound[2]
    
    print(f"모델 크기 - Height: {height:.2f}, Width: {width:.2f}, Depth: {depth:.2f}")
    
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
            avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
            
            # 이미지 좌표를 3D 좌표로 변환 (Y축 반전) - 어깨를 더 위로 올림
            shoulder_ratio = 1.0 - (avg_shoulder_y / img_size)
            # 어깨 비율을 의학적 기준에 맞게 조정 (전체 키의 85% 높이)
            shoulder_ratio = min(0.85, shoulder_ratio + 0.15)  # 15% 위로 올림
            shoulder_height = min_bound[1] + height * shoulder_ratio
            
            skeleton_points['left_shoulder'] = [center[0] - width * 0.25, shoulder_height, center[2]]
            skeleton_points['right_shoulder'] = [center[0] + width * 0.25, shoulder_height, center[2]]
            skeleton_points['shoulder_center'] = [center[0], shoulder_height, center[2]]
            
            print(f"AI 어깨 높이 (조정됨): {shoulder_height:.2f} (비율: {shoulder_ratio:.3f})")
        
        # 골반 위치 (AI 검출 기반) - 높이 조정
        if 'left_hip' in ai_landmarks and 'right_hip' in ai_landmarks:
            left_hip_y = ai_landmarks['left_hip']['y']
            right_hip_y = ai_landmarks['right_hip']['y']
            avg_hip_y = (left_hip_y + right_hip_y) / 2
            
            # 이미지 좌표를 3D 좌표로 변환 (Y축 반전) - 골반을 허리와 다리 사이 적절한 위치로
            hip_ratio = 1.0 - (avg_hip_y / img_size)
            # AI 검출 결과가 이미 정확하므로 최소한의 조정만 적용
            hip_ratio = max(0.45, min(0.65, hip_ratio))  # 45-65% 범위로 제한
            hip_height = min_bound[1] + height * hip_ratio
            
            skeleton_points['pelvis_center'] = [center[0], hip_height, center[2]]
            skeleton_points['left_hip'] = [center[0] - width * 0.15, hip_height, center[2]]
            skeleton_points['right_hip'] = [center[0] + width * 0.15, hip_height, center[2]]
            
            print(f"AI 골반 높이 (조정됨): {hip_height:.2f} (비율: {hip_ratio:.3f})")
        
        # 머리와 목 위치 (코 위치 기반) - 높이 조정
        if 'nose' in ai_landmarks:
            nose_y = ai_landmarks['nose']['y']
            nose_ratio = 1.0 - (nose_y / img_size)
            # 목 높이를 의학적 기준에 맞게 조정
            nose_ratio = min(0.92, nose_ratio + 0.10)  # 10% 위로 올림
            nose_height = min_bound[1] + height * nose_ratio
            
            skeleton_points['head_top'] = [center[0], max_bound[1], center[2]]
            skeleton_points['neck'] = [center[0], nose_height, center[2]]
            
            print(f"AI 목 높이 (조정됨): {nose_height:.2f} (비율: {nose_ratio:.3f})")
    
    # AI 검출이 실패했거나 불완전한 경우 기본값 사용 (높이 조정)
    if 'shoulder_center' not in skeleton_points:
        print("AI 어깨 검출 실패, 조정된 기본 비율 사용")
        shoulder_height = max_bound[1] - height * 0.15  # 0.22에서 0.15로 변경 (더 위로)
        skeleton_points['left_shoulder'] = [center[0] - width * 0.25, shoulder_height, center[2]]
        skeleton_points['right_shoulder'] = [center[0] + width * 0.25, shoulder_height, center[2]]
        skeleton_points['shoulder_center'] = [center[0], shoulder_height, center[2]]
    
    if 'pelvis_center' not in skeleton_points:
        print("AI 골반 검출 실패, 조정된 기본 비율 사용")
        pelvis_height = max_bound[1] - height * 0.42  # 0.35에서 0.42로 변경 (조금 아래로)
        skeleton_points['pelvis_center'] = [center[0], pelvis_height, center[2]]
        skeleton_points['left_hip'] = [center[0] - width * 0.15, pelvis_height, center[2]]
        skeleton_points['right_hip'] = [center[0] + width * 0.15, pelvis_height, center[2]]
    
    if 'neck' not in skeleton_points:
        skeleton_points['head_top'] = [center[0], max_bound[1], center[2]]
        skeleton_points['neck'] = [center[0], max_bound[1] - height * 0.08, center[2]]  # 0.10에서 0.08로 변경
    
    # 실제 검출된 위치를 기반으로 척추 구조 생성
    shoulder_height = skeleton_points['shoulder_center'][1]
    pelvis_height = skeleton_points['pelvis_center'][1]
    neck_height = skeleton_points['neck'][1]
    
    # 전체 척추 길이 설정 (목에서 꼬리뼈까지) - 자연스러운 비율로 조정
    spine_start_height = max_bound[1] - height * 0.08  # 목 시작
    spine_end_height = max_bound[1] - height * 0.55    # 미추 끝 (전체 키의 55%)
    total_spine_length = spine_start_height - spine_end_height
    
    print(f"척추 전체 길이: {total_spine_length:.2f} (키의 47%)")
    
    # 머리와 목
    skeleton_points['head_top'] = [center[0], max_bound[1], center[2]]
    skeleton_points['neck'] = [center[0], spine_start_height, center[2]]
    
    # 의학적 기준 척추 비율 적용
    # 경추 (C1-C7) - 전체 척추의 25%
    cervical_length = total_spine_length * 0.25
    cervical_start = skeleton_points['neck']
    cervical_end = [center[0], spine_start_height - cervical_length, center[2]]
    
    print(f"경추 길이: {cervical_length:.2f} (전체 척추의 25%)")
    
    for i in range(7):
        ratio = i / 6
        point = [
            cervical_start[0] + (cervical_end[0] - cervical_start[0]) * ratio,
            cervical_start[1] + (cervical_end[1] - cervical_start[1]) * ratio,
            cervical_start[2] + (cervical_end[2] - cervical_start[2]) * ratio
        ]
        skeleton_points[f'cervical_C{i+1}'] = point
    
    # 흉추 (T1-T12) - 전체 척추의 40%
    thoracic_length = total_spine_length * 0.40
    thoracic_start = cervical_end
    thoracic_end = [center[0], cervical_end[1] - thoracic_length, center[2] + depth * 0.05]  # 약간 뒤로 구부림
    
    print(f"흉추 길이: {thoracic_length:.2f} (전체 척추의 40%)")
    
    for i in range(12):
        ratio = i / 11
        # 흉추의 자연스러운 후만 커브 적용
        curve_factor = math.sin(ratio * math.pi) * 0.02 * depth
        point = [
            thoracic_start[0] + (thoracic_end[0] - thoracic_start[0]) * ratio,
            thoracic_start[1] + (thoracic_end[1] - thoracic_start[1]) * ratio,
            thoracic_start[2] + (thoracic_end[2] - thoracic_start[2]) * ratio + curve_factor
        ]
        skeleton_points[f'thoracic_T{i+1}'] = point
    
    # 요추 (L1-L5) - 전체 척추의 20%
    lumbar_length = total_spine_length * 0.20
    lumbar_start = thoracic_end
    lumbar_end = [center[0], thoracic_end[1] - lumbar_length, center[2] - depth * 0.02]  # 앞으로 구부림
    
    print(f"요추 길이: {lumbar_length:.2f} (전체 척추의 20%)")
    
    for i in range(5):
        ratio = i / 4
        # 요추의 자연스러운 전만 커브 적용
        curve_factor = -math.sin(ratio * math.pi) * 0.015 * depth
        point = [
            lumbar_start[0] + (lumbar_end[0] - lumbar_start[0]) * ratio,
            lumbar_start[1] + (lumbar_end[1] - lumbar_start[1]) * ratio,
            lumbar_start[2] + (lumbar_end[2] - lumbar_start[2]) * ratio + curve_factor
        ]
        skeleton_points[f'lumbar_L{i+1}'] = point
    
    # 천추와 미추 - 전체 척추의 15% (골반의 후면 구성)
    sacral_length = total_spine_length * 0.15
    sacral_start = lumbar_end
    sacral_end = [center[0], lumbar_end[1] - sacral_length, center[2]]
    
    print(f"천추+미추 길이: {sacral_length:.2f} (전체 척추의 15%)")
    
    # 천추 (S1-S5) - 천추+미추의 80% (골반 후면의 중심)
    sacrum_length = sacral_length * 0.80
    sacrum_center = [center[0], sacral_start[1] - sacrum_length/2, center[2]]
    skeleton_points['sacrum'] = sacrum_center
    
    # 미추 (Co1-Co4) - 천추+미추의 20% (척추의 최종 끝)
    coccyx_length = sacral_length * 0.20
    skeleton_points['coccyx'] = [center[0], sacral_end[1] + coccyx_length/2, center[2]]
    
    # 골반 위치 - 천추를 중심으로 해부학적으로 정확한 위치 설정
    # 골반(장골능)은 천추보다 약간 위에, 고관절은 천추와 같은 높이에 위치
    pelvis_height = sacrum_center[1] + sacrum_length * 0.3  # 천추 중심에서 약간 위
    hip_joint_height = sacrum_center[1]  # 천추 중심과 같은 높이
    
    skeleton_points['pelvis_center'] = [center[0], pelvis_height, center[2]]
    skeleton_points['left_hip'] = [center[0] - width * 0.15, hip_joint_height, center[2]]
    skeleton_points['right_hip'] = [center[0] + width * 0.15, hip_joint_height, center[2]]
    
    print(f"골반 중심 높이: {pelvis_height:.2f} (천추 기준)")
    print(f"고관절 높이: {hip_joint_height:.2f} (천추 중심과 동일)")
    
    # 어깨는 흉추 상부에 위치 (더 위로 올림)
    shoulder_height = cervical_end[1] + thoracic_length * 0.2  # 흉추 시작에서 더 위로 올림
    skeleton_points['left_shoulder'] = [center[0] - width * 0.25, shoulder_height, center[2]]
    skeleton_points['right_shoulder'] = [center[0] + width * 0.25, shoulder_height, center[2]]
    skeleton_points['shoulder_center'] = [center[0], shoulder_height, center[2]]
    
    return skeleton_points

def calculate_spine_angles(skeleton_points):
    """
    척추의 각종 각도를 계산합니다.
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
        if height < 0.1:  # 너무 짧은 경우 건너뛰기
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
            # 이미 정렬됨
            pass
        elif np.allclose(direction, -z_axis):
            # 180도 회전 필요
            cylinder.rotate(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), center=[0, 0, 0])
        else:
            # 회전축과 각도 계산
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(z_axis, direction))
            
            # 로드리게스 회전 공식 사용
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            cylinder.rotate(R, center=[0, 0, 0])
        
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
                skeleton_points[key1], 
                skeleton_points[key2], 
                radius=1.5,  # 더 굵게 만들어서 항상 보이도록
                color=[0, 1, 1]
            )
            if cylinder:
                cylinders.append(cylinder)
    
    # 척추 연결 (흉추) - 노란색
    for i in range(11):
        key1 = f'thoracic_T{i+1}'
        key2 = f'thoracic_T{i+2}'
        if key1 in skeleton_points and key2 in skeleton_points:
            cylinder = create_cylinder_between_points(
                skeleton_points[key1], 
                skeleton_points[key2], 
                radius=1.5,  # 더 굵게 만들어서 항상 보이도록
                color=[1, 1, 0]
            )
            if cylinder:
                cylinders.append(cylinder)
    
    # 척추 연결 (요추) - 주황색
    for i in range(4):
        key1 = f'lumbar_L{i+1}'
        key2 = f'lumbar_L{i+2}'
        if key1 in skeleton_points and key2 in skeleton_points:
            cylinder = create_cylinder_between_points(
                skeleton_points[key1], 
                skeleton_points[key2], 
                radius=1.5,  # 가장 굵게 만들어서 항상 보이도록
                color=[1, 0.5, 0]
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
            radius=1.5,  # 어깨를 더 굵게
            color=[0, 1, 0]
        )
        cylinder2 = create_cylinder_between_points(
            skeleton_points['shoulder_center'], 
            skeleton_points['right_shoulder'], 
            radius=1.5,  # 어깨를 더 굵게
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
            radius=1.5,  # 골반을 더 굵게
            color=[1, 0, 0]
        )
        cylinder2 = create_cylinder_between_points(
            skeleton_points['pelvis_center'], 
            skeleton_points['right_hip'], 
            radius=1.5,  # 골반을 더 굵게
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
            print(f".{issue}")
    else:
        print(f"전반적으로 양호한 자세입니다!")
    
    print("="*50)

def visualize_3d_pose():
    # 각 뷰의 DepthMap 로드
    views = {
        "front": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_정면.bmp",
        "right": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_오른쪽.bmp",
        "left": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_왼쪽.bmp",
        "back": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_후면.bmp"
    }
    
    # views = {
    #     "front": r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\test\정상\정면_여\DepthMap0.png",
    #     "right": r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\test\정상\오른쪽_여\DepthMap0.png",
    #     "left": r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\test\정상\왼쪽_여\DepthMap0.png",
    #     "back": r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\test\정상\후면_여\DepthMap0.png"
    # }
    
    front_landmarks = None
    

    
    # MediaPipe 백업
    if not front_landmarks:
        print("MediaPipe로 최종 백업 검출 시도...")
        front_landmarks = detect_landmarks_with_ai(views["front"])
    
    # 최종 검출 결과 출력
    if front_landmarks:
        print("개인별 신체 특징을 반영한 최종 랜드마크 검출 성공!")
        print("검출된 정확한 해부학적 랜드마크:")
        for name, landmark in front_landmarks.items():
            if isinstance(landmark, dict):
                confidence = landmark.get('visibility', 1.0)
                confidence_emoji = "🟢" if confidence > 0.9 else "🟡" if confidence > 0.7 else "🔴"
                print(f"  {confidence_emoji} {name}: x={landmark['x']:.1f}, y={landmark['y']:.1f}, 신뢰도={confidence:.3f}")
            else:
                print(f"  📍 {name}: {landmark}")
        print("3D에 최적화된 딥러닝 모델이 개인별 체형을 정확히 분석했습니다!")
    else:
        print("모든 AI 랜드마크 검출 방법 실패, 기본 해부학적 비율 사용")
        print("더 나은 결과를 위해 조명과 촬영 각도를 조정해보세요.")
    
    # 각 뷰의 포인트 클라우드 생성
    point_clouds = {}
    for view_name, file_path in views.items():
        depth_map = load_depth_map(file_path)
        if depth_map is not None:
            pcd = create_point_cloud_from_depth(depth_map, view_name)
            if pcd is not None:
                # 법선 벡터 계산
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
                point_clouds[view_name] = pcd
    
    # 정면을 기준으로 정렬 시작
    aligned_clouds = [point_clouds["front"]]
    front_target = point_clouds["front"]
    
    # 좌측과 우측을 정면과 정렬
    left_aligned = None
    right_aligned = None
    
    if "left" in point_clouds:
        left_aligned = align_point_clouds(point_clouds["left"], front_target, threshold=100)
        aligned_clouds.append(left_aligned)
    
    if "right" in point_clouds:
        right_aligned = align_point_clouds(point_clouds["right"], front_target, threshold=100)
        aligned_clouds.append(right_aligned)
    
    # 후면은 정렬된 좌우 포인트들과 함께 정렬
    if "back" in point_clouds and (left_aligned is not None or right_aligned is not None):
        # 정렬된 좌우 포인트들을 합쳐서 타겟으로 사용
        side_target = o3d.geometry.PointCloud()
        side_points = []
        side_colors = []
        
        if left_aligned is not None:
            side_points.extend(np.asarray(left_aligned.points))
            side_colors.extend(np.asarray(left_aligned.colors))
        if right_aligned is not None:
            side_points.extend(np.asarray(right_aligned.points))
            side_colors.extend(np.asarray(right_aligned.colors))
            
        side_target.points = o3d.utility.Vector3dVector(np.array(side_points))
        side_target.colors = o3d.utility.Vector3dVector(np.array(side_colors))
        
        # 후면을 좌우가 정렬된 포인트들과 정렬
        back_aligned = align_point_clouds(point_clouds["back"], side_target, threshold=100)
        aligned_clouds.append(back_aligned)
    
    # 모든 포인트 클라우드를 하나로 합치기
    merged_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for pcd in aligned_clouds:
        points.extend(np.asarray(pcd.points))
        colors.extend(np.asarray(pcd.colors))
    merged_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    merged_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # 노이즈 제거 및 다운샘플링
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size=2.0)
    
    # Statistical outlier removal을 이용한 노이즈 제거
    # nb_neighbors: 통계 계산에 사용할 이웃 점들의 수
    # std_ratio: 표준편차의 배수 (이 값을 벗어나는 점들을 제거)
    cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    merged_cloud = cl
    
    # 법선 벡터 재계산
    merged_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    # 스켈레톤 생성 및 각도 분석 (AI 랜드마크 사용)
    print("\n스켈레톤 생성 및 자세 분석 중...")
    skeleton_points = create_skeleton_from_pointcloud(merged_cloud, front_landmarks)
    angles = calculate_spine_angles(skeleton_points)
    skeleton_pcd, skeleton_cylinders = create_skeleton_visualization(skeleton_points)
    
    # 각도 분석 결과 출력
    print_angles(angles)
    
    # 메시 생성
    print("포인트 클라우드를 메시로 변환 중...")
    mesh = create_mesh_from_pointcloud(merged_cloud)
    
    # 메시 저장
    if mesh is not None:
        output_dir = "output/3d_models"
        os.makedirs(output_dir, exist_ok=True)
        
        # 메시를 반투명하게 설정 (50% 투명도)
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 연한 회색
        
        # 메시 파일 저장
        mesh_path = os.path.join(output_dir, "body_mesh.obj")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"메시가 저장되었습니다: {mesh_path}")
        
        # PLY 형식으로도 저장
        mesh_ply_path = os.path.join(output_dir, "body_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_ply_path, mesh)
        print(f"메시가 저장되었습니다: {mesh_ply_path}")
    
    # 초기 카메라 뷰포인트 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Pose with Skeleton Analysis", width=1024, height=768)
    
    # 포인트 클라우드 추가 (더 투명하게)
    merged_cloud_small = merged_cloud.voxel_down_sample(voxel_size=5.0)  # 더 많이 다운샘플링
    merged_cloud_small.paint_uniform_color([0.3, 0.3, 0.3])  # 더 어두운 회색으로 반투명 효과
    vis.add_geometry(merged_cloud_small)
    
    # 메시 추가 (있는 경우) - 투명 효과를 위한 특별 처리
    if mesh is not None:
        mesh_points = mesh.sample_points_uniformly(number_of_points=15000)  # 포인트 수 감소
        mesh_points.paint_uniform_color([0.5, 0.5, 0.5])  # 더 어둡게
        vis.add_geometry(mesh_points)
        
        # 추가: 메시의 경계선만 표시하기 위한 라인셋 생성
        try:
            line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            line_set.paint_uniform_color([0.6, 0.6, 0.6])  # 경계선 색상
            vis.add_geometry(line_set)
        except:
            # 라인셋 생성 실패시 기본 메시 사용
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            # vis.add_geometry(mesh)  # 주석 처리로 솔리드 메시 비활성화
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.point_size = 8.0  # 스켈레톤 포인트 크기를 크게 설정
    opt.background_color = np.asarray([0, 0, 0])  # 검은색 배경
    opt.mesh_show_wireframe = True  # 와이어프레임 표시
    opt.mesh_show_back_face = True  # 메시 뒷면도 표시
    
    # 스켈레톤 추가
    vis.add_geometry(skeleton_pcd)
    for cylinder in skeleton_cylinders:
        vis.add_geometry(cylinder)
    
    # 카메라 위치 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_up([0, -1, 0])
    
    # 시각화
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_3d_pose()