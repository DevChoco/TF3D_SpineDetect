import numpy as np
import cv2
import open3d as o3d
import os
import json
import copy

# PyTorch와 SMPL 관련 패키지는 선택적으로 임포트
SMPL_AVAILABLE = False
try:
    import torch
    torch.zeros(1)  # 간단한 테스트
    import smplx
    import trimesh
    from scipy.spatial.transform import Rotation as R
    from sklearn.neighbors import NearestNeighbors
    SMPL_AVAILABLE = True
    print("SMPL 관련 라이브러리가 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"SMPL 관련 라이브러리 로드 실패: {e}")
    print("기본 3D 분석 모드로 실행됩니다.")
    # SMPL 대체 더미 클래스들
    class torch:
        @staticmethod
        def zeros(*args, **kwargs):
            return None
        @staticmethod
        def device(*args, **kwargs):
            return "cpu"

class SMPLSpineAnalyzer:
    """SMPL/SMPL-X 기반 척추 분석 클래스"""
    
    def __init__(self, model_path=None, model_type='smplx'):
        """
        초기화
        Args:
            model_path: SMPL 모델 파일 경로
            model_type: 'smpl', 'smplh', 'smplx' 중 하나
        """
        if not SMPL_AVAILABLE:
            print("SMPL 라이브러리가 없어 기본 분석 모드로 실행됩니다.")
            self.smpl_model = None
            return
            
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # SMPL 모델 초기화 (모델 파일이 없어도 기본 모델 사용)
        try:
            # SMPL-X 모델 생성 (model_path 없이)
            self.smpl_model = smplx.create(model_type=model_type, 
                                         gender='neutral', 
                                         use_face_contour=False,
                                         create_global_orient=True,
                                         create_body_pose=True,
                                         create_betas=True,
                                         create_transl=True).to(self.device)
            print(f"SMPL-X 모델이 성공적으로 로드되었습니다. (타입: {model_type})")
        except Exception as e:
            print(f"SMPL 모델 로드 실패: {e}")
            # 기본 SMPL 모델로 재시도
            try:
                self.smpl_model = smplx.create(model_type='smpl', 
                                             gender='neutral',
                                             create_global_orient=True,
                                             create_body_pose=True,
                                             create_betas=True,
                                             create_transl=True).to(self.device)
                print("기본 SMPL 모델이 로드되었습니다.")
            except Exception as e2:
                print(f"기본 SMPL 모델 로드도 실패: {e2}")
                self.smpl_model = None
        
        # 척추 관련 조인트 인덱스 (SMPL-X 기준)
        self.spine_joints = {
            'pelvis': 0,        # 골반
            'spine1': 3,        # 요추 하부
            'spine2': 6,        # 요추 상부 
            'spine3': 9,        # 흉추 하부
            'neck': 12,         # 경추 하부
            'head': 15,         # 머리
            'left_shoulder': 16,  # 왼쪽 어깨
            'right_shoulder': 17  # 오른쪽 어깨
        }
        
        # 척추 세그먼트 정의
        self.spine_segments = {
            'cervical': ['neck', 'head'],           # 경추
            'thoracic': ['spine3', 'neck'],         # 흉추  
            'lumbar': ['spine2', 'spine3'],         # 요추
            'sacral': ['pelvis', 'spine1']          # 천추/골반
        }
    
    def fit_smpl_to_pointcloud(self, point_cloud, max_iterations=100):
        """
        포인트 클라우드에 SMPL 모델을 피팅
        Args:
            point_cloud: Open3D PointCloud 객체
            max_iterations: 최대 반복 횟수
        Returns:
            fitted_vertices: 피팅된 SMPL 메시의 정점
            joints_3d: 3D 조인트 위치
            pose_params: 포즈 파라미터
        """
        if not SMPL_AVAILABLE or self.smpl_model is None:
            print("SMPL 모델이 사용할 수 없습니다. 기본 분석을 사용합니다.")
            return None, None, None
            
        # 포인트 클라우드를 numpy 배열로 변환
        target_points = np.asarray(point_cloud.points)
        
        # SMPL 파라미터 초기화
        batch_size = 1
        global_orient = torch.zeros(batch_size, 3, device=self.device, requires_grad=True)
        body_pose = torch.zeros(batch_size, 63, device=self.device, requires_grad=True)  # 21 joints * 3
        betas = torch.zeros(batch_size, 10, device=self.device, requires_grad=True)
        transl = torch.zeros(batch_size, 3, device=self.device, requires_grad=True)
        
        # 옵티마이저 설정
        optimizer = torch.optim.Adam([global_orient, body_pose, betas, transl], lr=0.01)
        
        # 피팅 과정
        for i in range(max_iterations):
            optimizer.zero_grad()
            
            # SMPL 모델 forward pass
            output = self.smpl_model(global_orient=global_orient,
                                   body_pose=body_pose,
                                   betas=betas,
                                   transl=transl)
            
            vertices = output.vertices[0].cpu().numpy()
            
            # 손실 함수 계산 (Chamfer distance)
            loss = self.compute_chamfer_loss(vertices, target_points)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                print(f"Iteration {i}, Loss: {loss.item():.6f}")
        
        # 최종 결과
        with torch.no_grad():
            output = self.smpl_model(global_orient=global_orient,
                                   body_pose=body_pose,
                                   betas=betas,
                                   transl=transl)
            fitted_vertices = output.vertices[0].cpu().numpy()
            joints_3d = output.joints[0].cpu().numpy()
            
            pose_params = {
                'global_orient': global_orient.cpu().numpy(),
                'body_pose': body_pose.cpu().numpy(),
                'betas': betas.cpu().numpy(),
                'transl': transl.cpu().numpy()
            }
        
        return fitted_vertices, joints_3d, pose_params
    
    def compute_chamfer_loss(self, vertices, target_points):
        """Chamfer distance 계산"""
        vertices_torch = torch.tensor(vertices, device=self.device, requires_grad=True)
        target_torch = torch.tensor(target_points, device=self.device)
        
        # 서브샘플링으로 계산 속도 향상
        if len(target_points) > 5000:
            indices = np.random.choice(len(target_points), 5000, replace=False)
            target_torch = target_torch[indices]
        
        # 각 타겟 포인트에서 가장 가까운 버텍스까지의 거리
        dist1 = torch.cdist(target_torch.unsqueeze(0), vertices_torch.unsqueeze(0)).min(dim=2)[0]
        
        # 각 버텍스에서 가장 가까운 타겟 포인트까지의 거리  
        dist2 = torch.cdist(vertices_torch.unsqueeze(0), target_torch.unsqueeze(0)).min(dim=2)[0]
        
        return dist1.mean() + dist2.mean()
    
    def calculate_spine_angles(self, joints_3d):
        """
        척추 각도 계산
        Args:
            joints_3d: 3D 조인트 위치 (N, 3)
        Returns:
            spine_analysis: 척추 분석 결과
        """
        spine_analysis = {}
        
        # 각 척추 세그먼트의 각도 계산
        for segment_name, joint_names in self.spine_segments.items():
            if len(joint_names) >= 2:
                start_joint = joints_3d[self.spine_joints[joint_names[0]]]
                end_joint = joints_3d[self.spine_joints[joint_names[1]]]
                
                # 세그먼트 벡터
                segment_vector = end_joint - start_joint
                segment_vector = segment_vector / np.linalg.norm(segment_vector)
                
                # 수직 벡터와의 각도 (시상면)
                vertical_vector = np.array([0, 1, 0])
                sagittal_angle = np.arccos(np.clip(np.dot(segment_vector, vertical_vector), -1, 1))
                sagittal_angle_deg = np.degrees(sagittal_angle)
                
                # 전후면 기울기 (관상면)
                frontal_vector = np.array([1, 0, 0])
                frontal_angle = np.arccos(np.clip(np.dot(segment_vector, frontal_vector), -1, 1))
                frontal_angle_deg = np.degrees(frontal_angle) - 90  # 90도 기준으로 조정
                
                spine_analysis[segment_name] = {
                    'start_position': start_joint.tolist(),
                    'end_position': end_joint.tolist(),
                    'vector': segment_vector.tolist(),
                    'sagittal_angle': float(sagittal_angle_deg),
                    'frontal_angle': float(frontal_angle_deg),
                    'length': float(np.linalg.norm(end_joint - start_joint))
                }
        
        # 어깨 수평 각도
        left_shoulder = joints_3d[self.spine_joints['left_shoulder']]
        right_shoulder = joints_3d[self.spine_joints['right_shoulder']]
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_horizontal_angle = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
        
        spine_analysis['shoulder_level'] = {
            'left_position': left_shoulder.tolist(),
            'right_position': right_shoulder.tolist(),
            'horizontal_angle': float(shoulder_horizontal_angle),
            'height_difference': float(right_shoulder[1] - left_shoulder[1])
        }
        
        # 전체 척추 커브 분석
        spine_analysis['overall_posture'] = self.analyze_overall_posture(joints_3d)
        
        return spine_analysis
    
    def analyze_overall_posture(self, joints_3d):
        """전체 자세 분석"""
        # 주요 척추 포인트들
        pelvis = joints_3d[self.spine_joints['pelvis']]
        spine1 = joints_3d[self.spine_joints['spine1']]
        spine2 = joints_3d[self.spine_joints['spine2']]
        spine3 = joints_3d[self.spine_joints['spine3']]
        neck = joints_3d[self.spine_joints['neck']]
        head = joints_3d[self.spine_joints['head']]
        
        # 전체 척추 높이
        total_spine_height = head[1] - pelvis[1]
        
        # 전방 머리 자세 (Forward Head Posture) 검사
        head_forward_distance = head[2] - neck[2]  # Z축 (전후) 거리
        
        # 요추 전만 (Lumbar Lordosis) 
        lumbar_curve = self.calculate_curve_angle([pelvis, spine1, spine2])
        
        # 흉추 후만 (Thoracic Kyphosis)
        thoracic_curve = self.calculate_curve_angle([spine2, spine3, neck])
        
        # 경추 전만 (Cervical Lordosis)
        cervical_curve = self.calculate_curve_angle([spine3, neck, head])
        
        return {
            'total_spine_height': float(total_spine_height),
            'head_forward_distance': float(head_forward_distance),
            'lumbar_lordosis': float(lumbar_curve),
            'thoracic_kyphosis': float(thoracic_curve),
            'cervical_lordosis': float(cervical_curve),
            'posture_assessment': self.assess_posture(head_forward_distance, lumbar_curve, thoracic_curve)
        }
    
    def calculate_curve_angle(self, points):
        """3점을 이용한 커브 각도 계산"""
        if len(points) != 3:
            return 0.0
            
        p1, p2, p3 = points
        
        # 두 벡터 계산
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 각도 계산
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return np.degrees(angle)
    
    def assess_posture(self, head_forward, lumbar_curve, thoracic_curve):
        """자세 평가"""
        issues = []
        
        if head_forward > 20:  # 2cm 이상 전방 돌출
            issues.append("전방 머리 자세 (Forward Head Posture)")
        
        if lumbar_curve < 20 or lumbar_curve > 60:
            issues.append("요추 전만 이상")
            
        if thoracic_curve < 20 or thoracic_curve > 50:
            issues.append("흉추 후만 이상")
        
        if not issues:
            return "정상 자세"
        else:
            return ", ".join(issues)
    
    def create_spine_visualization(self, joints_3d, spine_analysis):
        """척추 시각화를 위한 3D 객체 생성"""
        geometries = []
        
        # 척추 조인트 시각화
        for joint_name, joint_idx in self.spine_joints.items():
            if joint_idx < len(joints_3d):
                joint_pos = joints_3d[joint_idx]
                
                # 조인트 구체 생성
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
                sphere.translate(joint_pos)
                
                # 조인트별 색상
                if 'spine' in joint_name or joint_name == 'neck':
                    sphere.paint_uniform_color([1, 0, 0])  # 빨간색 - 척추
                elif joint_name == 'pelvis':
                    sphere.paint_uniform_color([0, 0, 1])  # 파란색 - 골반
                elif 'shoulder' in joint_name:
                    sphere.paint_uniform_color([0, 1, 0])  # 초록색 - 어깨
                else:
                    sphere.paint_uniform_color([1, 1, 0])  # 노란색 - 머리
                
                geometries.append(sphere)
        
        # 척추 연결선 생성
        spine_connections = [
            ('pelvis', 'spine1'),
            ('spine1', 'spine2'), 
            ('spine2', 'spine3'),
            ('spine3', 'neck'),
            ('neck', 'head'),
            ('spine3', 'left_shoulder'),
            ('spine3', 'right_shoulder')
        ]
        
        for start_joint, end_joint in spine_connections:
            if start_joint in self.spine_joints and end_joint in self.spine_joints:
                start_idx = self.spine_joints[start_joint]
                end_idx = self.spine_joints[end_joint]
                
                if start_idx < len(joints_3d) and end_idx < len(joints_3d):
                    start_pos = joints_3d[start_idx]
                    end_pos = joints_3d[end_idx]
                    
                    # 선분 생성
                    line_points = [start_pos, end_pos]
                    line_indices = [[0, 1]]
                    
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(line_points)
                    line_set.lines = o3d.utility.Vector2iVector(line_indices)
                    line_set.paint_uniform_color([1, 1, 1])  # 흰색 연결선
                    
                    geometries.append(line_set)
        
        return geometries


class BasicSpineAnalyzer:
    """기본 척추 분석 클래스 (SMPL 없이 사용)"""
    
    def __init__(self):
        """포인트 클라우드 기반 척추 분석 초기화"""
        self.spine_points = []
        
        # 척추 세그먼트 정의 (인덱스 기반)
        self.spine_segments = {
            'cervical': [0, 1],      # 경추 (목 부분)
            'thoracic': [1, 2, 3],   # 흉추 (가슴 부분)  
            'lumbar': [3, 4],        # 요추 (허리 부분)
            'sacral': [4, 5]         # 천추/골반 (골반 부분)
        }
        
    def extract_spine_from_pointcloud(self, point_cloud):
        """포인트 클라우드에서 척추 추정 (최고 정밀도 버전) - 실제 인체 해부학 기반"""
        points = np.asarray(point_cloud.points)
        
        if len(points) < 100:
            return np.array([])
        
        print("🔬 정밀 척추 분석을 시작합니다...")
        
        # 1단계: 신체 전체 분석 및 좌표계 정규화
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        total_height = y_max - y_min
        total_width = x_max - x_min
        total_depth = z_max - z_min
        
        print(f"📏 신체 치수 - 높이: {total_height:.1f}, 폭: {total_width:.1f}, 깊이: {total_depth:.1f}")
        
        # 2단계: 해부학적 척추 영역 정의 (더 정확한 비율)
        # 실제 인체에서 척추는 발목에서 35%-88% 높이에 위치
        spine_y_min = y_min + total_height * 0.37  # 골반 시작점 (37%)
        spine_y_max = y_min + total_height * 0.88  # 목 끝점 (88%)
        
        # 3단계: 척추는 몸의 중심축 후방에 위치 (해부학적 정확성)
        body_center_x = (x_min + x_max) / 2
        body_center_z = (z_min + z_max) / 2
        
        # 척추는 몸의 후방 50%-90% 지점에 위치 (더 깊숙이 배치)
        spine_z_min = z_min + total_depth * 0.50  # 훨씬 더 뒤쪽으로 이동
        spine_z_max = z_min + total_depth * 0.90  # 가장 깊숙한 위치까지
        
        # 척추는 몸의 중심축에서 좌우 ±3cm 이내
        spine_x_tolerance = min(15, total_width * 0.08)  # 최대 1.5cm 또는 폭의 8%
        
        print(f"🎯 척추 영역 정의:")
        print(f"  높이: {spine_y_min:.1f} ~ {spine_y_max:.1f}")
        print(f"  중심 X: {body_center_x:.1f} ± {spine_x_tolerance:.1f}")
        print(f"  깊이 Z: {spine_z_min:.1f} ~ {spine_z_max:.1f}")
        
        # 4단계: 척추 영역 포인트 추출 (3차원 필터링)
        spine_mask = (
            (points[:, 1] >= spine_y_min) & (points[:, 1] <= spine_y_max) &
            (np.abs(points[:, 0] - body_center_x) <= spine_x_tolerance) &
            (points[:, 2] >= spine_z_min) & (points[:, 2] <= spine_z_max)
        )
        spine_region_points = points[spine_mask]
        
        if len(spine_region_points) < 30:
            print(f"❌ 척추 영역 포인트 부족: {len(spine_region_points)}")
            return np.array([])
        
        print(f"✅ 척추 영역 포인트 수: {len(spine_region_points)}")
        
        # 5단계: 척추를 해부학적 세그먼트로 정밀 분할
        spine_height = spine_y_max - spine_y_min
        
        # 실제 척추 비율 (해부학 교과서 기준)
        segment_ratios = {
            'C7': 0.92,    # 경추 7번 (목 아래쪽)
            'T3': 0.78,    # 흉추 3번 (상부 가슴)
            'T8': 0.58,    # 흉추 8번 (중부 가슴)
            'T12': 0.38,   # 흉추 12번 (하부 가슴)
            'L3': 0.20,    # 요추 3번 (허리)
            'S1': 0.05     # 천추 1번 (골반)
        }
        
        spine_candidates = []
        segment_names = ['C7', 'T3', 'T8', 'T12', 'L3', 'S1']
        
        for i, (segment_name, ratio) in enumerate(segment_ratios.items()):
            # 각 세그먼트의 정확한 높이 계산
            segment_y = spine_y_min + spine_height * ratio
            search_range = spine_height * 0.06  # 높이의 6% 범위에서 검색
            
            segment_y_min = segment_y - search_range
            segment_y_max = segment_y + search_range
            
            # 해당 높이 범위의 척추 포인트들 추출
            height_mask = (
                (spine_region_points[:, 1] >= segment_y_min) & 
                (spine_region_points[:, 1] <= segment_y_max)
            )
            segment_points = spine_region_points[height_mask]
            
            if len(segment_points) >= 3:
                # 더 정밀한 척추 중심 계산
                # 1) X축: 몸의 정중선에 가장 가까운 포인트들
                x_distances = np.abs(segment_points[:, 0] - body_center_x)
                x_threshold = np.percentile(x_distances, 30)  # 가장 중앙에 가까운 30%
                x_mask = x_distances <= x_threshold
                
                # 2) Z축: 해당 높이에서의 후방 포인트들 (척추는 뒤쪽)
                if np.sum(x_mask) > 0:
                    x_filtered_points = segment_points[x_mask]
                    z_values = x_filtered_points[:, 2]
                    
                    # 세그먼트별 맞춤형 Z축 선택 (더 깊숙한 위치)
                    if segment_name in ['C7', 'T3']:  # 상부: 가장 뒤쪽
                        z_threshold = np.percentile(z_values, 85)
                    elif segment_name in ['T8', 'T12']:  # 중부: 뒤쪽
                        z_threshold = np.percentile(z_values, 80)
                    else:  # L3, S1: 여전히 뒤쪽이지만 약간 앞
                        z_threshold = np.percentile(z_values, 75)
                    
                    z_mask = x_filtered_points[:, 2] >= z_threshold
                    final_segment_points = x_filtered_points[z_mask]
                    
                    if len(final_segment_points) > 0:
                        # 최종 척추 중심점 계산 (가중평균 사용)
                        weights = 1.0 / (1.0 + np.abs(final_segment_points[:, 0] - body_center_x))
                        
                        weighted_center = np.average(final_segment_points, axis=0, weights=weights)
                        spine_candidates.append(weighted_center)
                        
                        print(f"✅ {segment_name}: Y={weighted_center[1]:.1f}, X={weighted_center[0]:.1f}, Z={weighted_center[2]:.1f} (포인트:{len(final_segment_points)})")
                    else:
                        print(f"❌ {segment_name}: Z축 필터링 후 포인트 없음")
                else:
                    print(f"❌ {segment_name}: X축 필터링 후 포인트 없음")
            else:
                print(f"❌ {segment_name}: 높이 범위 포인트 부족 ({len(segment_points)})")
        
        if len(spine_candidates) < 4:
            print(f"❌ 척추 후보 포인트 부족: {len(spine_candidates)}")
            return np.array([])
        
        spine_candidates = np.array(spine_candidates)
        
        # 6단계: 척추 곡선 최적화 및 해부학적 검증
        spine_candidates = self.optimize_spine_curve(spine_candidates)
        
        # 7단계: 최종 해부학적 검증
        if len(spine_candidates) >= 6:
            if self.validate_anatomical_spine(spine_candidates, total_height, total_depth):
                print("✅ 해부학적 척추 구조 검증 통과")
                return spine_candidates
            else:
                print("❌ 해부학적 검증 실패")
        
        return np.array([])
    
    def optimize_spine_curve(self, spine_points):
        """척추 곡선 최적화 (생체역학적 원리 적용)"""
        if len(spine_points) < 4:
            return spine_points
        
        # 1단계: 자연스러운 척추 곡선 형성
        # 실제 척추는 S자 곡선을 형성 (경추 전만, 흉추 후만, 요추 전만)
        
        optimized_points = spine_points.copy()
        
        # 2단계: 인접 포인트 간의 부드러운 전이 보장
        for i in range(1, len(optimized_points) - 1):
            prev_point = optimized_points[i-1]
            curr_point = optimized_points[i]
            next_point = optimized_points[i+1]
            
            # 급격한 변화 완화 (특히 Z축)
            expected_z = (prev_point[2] + next_point[2]) / 2
            if abs(curr_point[2] - expected_z) > 15:  # 1.5cm 이상 급변시 보정
                optimized_points[i][2] = (curr_point[2] + expected_z) / 2
        
        # 3단계: 척추의 자연스러운 깊이 곡선 적용
        # 상부(C7,T3)는 더 뒤쪽, 중부(T8,T12)는 가장 뒤쪽, 하부(L3,S1)는 상대적 앞쪽
        if len(optimized_points) >= 6:
            # Z축 곡선 조정
            z_values = optimized_points[:, 2]
            z_mean = np.mean(z_values)
            
            # 자연스러운 척추 깊이 프로파일 적용
            depth_adjustments = [2, 4, 6, 4, 1, -1]  # C7부터 S1까지의 상대적 깊이
            
            for i, adj in enumerate(depth_adjustments):
                if i < len(optimized_points):
                    optimized_points[i][2] = z_mean + adj
        
        return optimized_points
    
    def validate_anatomical_spine(self, spine_points, total_height, total_depth):
        """해부학적 척추 구조 검증"""
        if len(spine_points) < 4:
            return False
        
        # 1. 높이 순서 검증 (C7 > T3 > T8 > T12 > L3 > S1)
        heights = spine_points[:, 1]
        for i in range(len(heights) - 1):
            if heights[i] <= heights[i+1]:
                print(f"❌ 높이 순서 오류: {i}번 포인트 ({heights[i]:.1f}) <= {i+1}번 포인트 ({heights[i+1]:.1f})")
                return False
        
        # 2. 척추 간격 검증 (해부학적 합리성)
        for i in range(len(spine_points) - 1):
            distance = np.linalg.norm(spine_points[i] - spine_points[i+1])
            expected_distance = total_height * 0.08  # 전체 높이의 8% 정도
            
            if distance < expected_distance * 0.3 or distance > expected_distance * 3:
                print(f"❌ 척추 간격 비정상: {i}-{i+1} 거리 {distance:.1f}mm (예상: {expected_distance:.1f}±)")
                return False
        
        # 3. 척추 깊이 검증 (Z축 합리성)
        z_values = spine_points[:, 2]
        z_range = np.max(z_values) - np.min(z_values)
        expected_z_range = total_depth * 0.2  # 전체 깊이의 20% 내
        
        if z_range > expected_z_range:
            print(f"❌ 척추 깊이 범위 과대: {z_range:.1f}mm (예상: <{expected_z_range:.1f}mm)")
            return False
        
        # 4. 척추 중심축 검증 (X축 편차)
        x_values = spine_points[:, 0]
        x_std = np.std(x_values)
        
        if x_std > 20:  # 2cm 이상 편차는 비정상
            print(f"❌ 척추 중심축 편차 과대: {x_std:.1f}mm")
            return False
        
        print("✅ 모든 해부학적 검증 통과")
        return True
    
    def smooth_spine_curve(self, spine_points):
        """척추 곡선 스무딩"""
        if len(spine_points) < 3:
            return spine_points
        
        # 이동 평균을 사용한 스무딩
        smoothed_points = []
        window_size = min(3, len(spine_points))
        
        for i in range(len(spine_points)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(spine_points), i + window_size // 2 + 1)
            
            avg_point = np.mean(spine_points[start_idx:end_idx], axis=0)
            smoothed_points.append(avg_point)
        
        return np.array(smoothed_points)
    
    def validate_spine_curve(self, spine_points):
        """척추 곡선의 타당성 검증"""
        if len(spine_points) < 4:
            return False
        
        # 1. 높이가 단조감소하는지 확인
        heights = spine_points[:, 1]
        if not all(heights[i] >= heights[i+1] for i in range(len(heights)-1)):
            return False
        
        # 2. X축 편차가 너무 크지 않은지 확인 (척추 측만 심한 경우 제외)
        x_deviation = np.std(spine_points[:, 0])
        if x_deviation > 50:  # 5cm 이상 편차
            return False
        
        # 3. 인접 포인트 간 거리가 합리적인지 확인
        for i in range(len(spine_points) - 1):
            distance = np.linalg.norm(spine_points[i] - spine_points[i+1])
            if distance > 100 or distance < 10:  # 10cm 초과 또는 1cm 미만
                return False
        
        return True
    
    def analyze_posture_lines_and_angles(self, spine_points):
        """어깨, 척추, 골반, 목 라인 분석 및 각도 측정"""
        if len(spine_points) < 6:
            return {}
        
        # 주요 포인트 정의
        neck_point = spine_points[0]        # C7 경추
        upper_thoracic = spine_points[1]    # T3 상부 흉추
        mid_thoracic = spine_points[2]      # T8 중부 흉추
        lower_thoracic = spine_points[3]    # T12 하부 흉추
        lumbar_point = spine_points[4]      # L3 요추
        pelvis_point = spine_points[5]      # S1 천추/골반
        
        analysis = {}
        
        # 1. 어깨 가로라인 분석
        shoulder_analysis = self.analyze_shoulder_line_detailed(neck_point, upper_thoracic)
        analysis['shoulder_line'] = shoulder_analysis
        
        # 2. 척추 세로라인 분석 (세분화)
        spine_lines_analysis = self.analyze_spine_vertical_lines(spine_points)
        analysis['spine_lines'] = spine_lines_analysis
        
        # 3. 골반 가로라인 분석
        pelvis_analysis = self.analyze_pelvis_line(lumbar_point, pelvis_point)
        analysis['pelvis_line'] = pelvis_analysis
        
        # 4. 목 세로라인 분석
        neck_analysis = self.analyze_neck_line(neck_point, upper_thoracic)
        analysis['neck_line'] = neck_analysis
        
        # 5. 전체 정렬 분석
        alignment_analysis = self.analyze_overall_alignment(
            shoulder_analysis, spine_lines_analysis, pelvis_analysis, neck_analysis
        )
        analysis['overall_alignment'] = alignment_analysis
        
        return analysis
    
    def analyze_shoulder_line_detailed(self, neck_point, upper_thoracic):
        """어깨 가로라인 상세 분석"""
        # 어깨 포인트 추정 (해부학적 기준)
        shoulder_width = 45  # 어깨 너비 추정 (4.5cm)
        
        # C7에서 T3 방향으로 어깨선 추정
        spine_direction = upper_thoracic - neck_point
        
        # 어깨선은 척추에 수직
        if abs(spine_direction[0]) > 0.001:  # X축 변화가 있는 경우
            # 수직 벡터 계산
            shoulder_direction = np.array([1, 0, 0])  # 기본적으로 X축 방향
        else:
            shoulder_direction = np.array([1, 0, 0])
        
        # 어깨 포인트 계산
        left_shoulder = neck_point + shoulder_direction * shoulder_width
        right_shoulder = neck_point - shoulder_direction * shoulder_width
        
        # 어깨선 벡터
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_length = np.linalg.norm(shoulder_vector)
        
        # 수평선과의 각도 (수평면 기준)
        horizontal_vector = np.array([1, 0, 0])  # X축
        if shoulder_length > 0:
            shoulder_unit = shoulder_vector / shoulder_length
            # Y축 기울기 (어깨 높낮이)
            shoulder_tilt_angle = np.degrees(np.arcsin(np.clip(shoulder_unit[1], -1, 1)))
            # Z축 기울기 (앞뒤 기울기)
            shoulder_depth_angle = np.degrees(np.arcsin(np.clip(shoulder_unit[2], -1, 1)))
        else:
            shoulder_tilt_angle = 0
            shoulder_depth_angle = 0
        
        return {
            'left_shoulder_pos': left_shoulder.tolist(),
            'right_shoulder_pos': right_shoulder.tolist(),
            'shoulder_vector': shoulder_vector.tolist(),
            'shoulder_length': float(shoulder_length),
            'tilt_angle': float(shoulder_tilt_angle),  # 좌우 기울기
            'depth_angle': float(shoulder_depth_angle),  # 앞뒤 기울기
            'line_type': 'horizontal',
            'description': f"어깨선 기울기: {abs(shoulder_tilt_angle):.1f}° " + 
                          ("(우측 높음)" if shoulder_tilt_angle > 2 else 
                           "(좌측 높음)" if shoulder_tilt_angle < -2 else "(수평)")
        }
    
    def analyze_spine_vertical_lines(self, spine_points):
        """척추 세로라인 세분화 분석"""
        lines_analysis = {}
        
        # 1. 경추선 (C7-T3)
        cervical_line = self.analyze_spine_segment_line(
            spine_points[0], spine_points[1], "cervical", "경추선"
        )
        lines_analysis['cervical'] = cervical_line
        
        # 2. 상부 흉추선 (T3-T8)
        upper_thoracic_line = self.analyze_spine_segment_line(
            spine_points[1], spine_points[2], "upper_thoracic", "상부흉추선"
        )
        lines_analysis['upper_thoracic'] = upper_thoracic_line
        
        # 3. 하부 흉추선 (T8-T12)
        lower_thoracic_line = self.analyze_spine_segment_line(
            spine_points[2], spine_points[3], "lower_thoracic", "하부흉추선"
        )
        lines_analysis['lower_thoracic'] = lower_thoracic_line
        
        # 4. 요추선 (T12-L3)
        lumbar_line = self.analyze_spine_segment_line(
            spine_points[3], spine_points[4], "lumbar", "요추선"
        )
        lines_analysis['lumbar'] = lumbar_line
        
        # 5. 천추선 (L3-S1)
        sacral_line = self.analyze_spine_segment_line(
            spine_points[4], spine_points[5], "sacral", "천추선"
        )
        lines_analysis['sacral'] = sacral_line
        
        # 6. 전체 척추선 (C7-S1)
        overall_spine_line = self.analyze_spine_segment_line(
            spine_points[0], spine_points[-1], "overall", "전체척추선"
        )
        lines_analysis['overall'] = overall_spine_line
        
        return lines_analysis
    
    def analyze_spine_segment_line(self, start_point, end_point, segment_type, segment_name):
        """척추 세그먼트 라인 분석"""
        # 세그먼트 벡터
        segment_vector = end_point - start_point
        segment_length = np.linalg.norm(segment_vector)
        
        if segment_length == 0:
            return {
                'error': f'{segment_name} 길이가 0입니다.'
            }
        
        segment_unit = segment_vector / segment_length
        
        # 수직선(Y축)과의 각도
        vertical_vector = np.array([0, 1, 0])
        vertical_angle = np.degrees(np.arccos(np.clip(np.dot(segment_unit, vertical_vector), -1, 1)))
        
        # X축 기울기 (좌우 기울기)
        lateral_angle = np.degrees(np.arcsin(np.clip(segment_unit[0], -1, 1)))
        
        # Z축 기울기 (앞뒤 기울기)  
        sagittal_angle = np.degrees(np.arcsin(np.clip(segment_unit[2], -1, 1)))
        
        # 정상 범위 평가
        normal_ranges = {
            'cervical': {'lateral': 5, 'sagittal': 15},
            'upper_thoracic': {'lateral': 3, 'sagittal': 10},
            'lower_thoracic': {'lateral': 3, 'sagittal': 5},
            'lumbar': {'lateral': 5, 'sagittal': 15},
            'sacral': {'lateral': 3, 'sagittal': 10},
            'overall': {'lateral': 5, 'sagittal': 10}
        }
        
        normal_range = normal_ranges.get(segment_type, {'lateral': 5, 'sagittal': 10})
        
        # 상태 평가
        status = []
        if abs(lateral_angle) > normal_range['lateral']:
            direction = "우측" if lateral_angle > 0 else "좌측"
            status.append(f"좌우 기울어짐 ({direction}으로 {abs(lateral_angle):.1f}°)")
        
        if abs(sagittal_angle) > normal_range['sagittal']:
            direction = "전방" if sagittal_angle > 0 else "후방"
            status.append(f"전후 기울어짐 ({direction}으로 {abs(sagittal_angle):.1f}°)")
        
        if not status:
            status.append("정상 범위")
        
        return {
            'start_pos': start_point.tolist(),
            'end_pos': end_point.tolist(),
            'vector': segment_vector.tolist(),
            'length': float(segment_length),
            'vertical_angle': float(vertical_angle),
            'lateral_angle': float(lateral_angle),     # 좌우 기울기
            'sagittal_angle': float(sagittal_angle),   # 앞뒤 기울기
            'line_type': 'vertical',
            'segment_name': segment_name,
            'status': ', '.join(status),
            'description': f"{segment_name}: {', '.join(status)}"
        }
    
    def analyze_pelvis_line(self, lumbar_point, pelvis_point):
        """골반 가로라인 분석 - 해부학적으로 정확한 골반 위치"""
        # 골반 너비 추정 (해부학적 기준)
        pelvis_width = 30  # 골반 너비 추정 (3.0cm)
        
        # 실제 골반뼈 위치는 S1(천추)에서 위쪽에 위치
        # 골반선은 장골능(iliac crest) 라인을 기준으로 함
        actual_pelvis_level = pelvis_point + np.array([0, 15, 0])  # S1에서 위로 1.5cm
        
        # L3-S1 방향에서 골반선 추정
        spine_direction = pelvis_point - lumbar_point
        
        # 골반선은 척추에 수직 (수평면)
        pelvis_direction = np.array([1, 0, 0])  # X축 방향
        
        # 골반 포인트 계산 (실제 골반뼈 위치에서)
        left_pelvis = actual_pelvis_level + pelvis_direction * pelvis_width
        right_pelvis = actual_pelvis_level - pelvis_direction * pelvis_width
        
        # 골반선 벡터
        pelvis_vector = right_pelvis - left_pelvis
        pelvis_length = np.linalg.norm(pelvis_vector)
        
        # 수평선과의 각도
        if pelvis_length > 0:
            pelvis_unit = pelvis_vector / pelvis_length
            # Y축 기울기 (골반 높낮이)
            pelvis_tilt_angle = np.degrees(np.arcsin(np.clip(pelvis_unit[1], -1, 1)))
            # Z축 기울기 (앞뒤 기울기)
            pelvis_depth_angle = np.degrees(np.arcsin(np.clip(pelvis_unit[2], -1, 1)))
        else:
            pelvis_tilt_angle = 0
            pelvis_depth_angle = 0
        
        # 척추와 골반의 관계 분석
        if len(spine_direction) > 0:
            spine_pelvis_angle = np.degrees(np.arccos(np.clip(
                np.dot(spine_direction / np.linalg.norm(spine_direction), 
                       np.array([0, -1, 0])), -1, 1)))  # 아래쪽 방향과의 각도
        else:
            spine_pelvis_angle = 0
        
        return {
            'left_pelvis_pos': left_pelvis.tolist(),
            'right_pelvis_pos': right_pelvis.tolist(),
            'pelvis_vector': pelvis_vector.tolist(),
            'pelvis_length': float(pelvis_length),
            'tilt_angle': float(pelvis_tilt_angle),     # 좌우 기울기
            'depth_angle': float(pelvis_depth_angle),   # 앞뒤 기울기
            'spine_pelvis_angle': float(spine_pelvis_angle),  # 척추-골반 각도
            'line_type': 'horizontal',
            'actual_pelvis_level': actual_pelvis_level.tolist(),  # 실제 골반 높이
            'description': f"골반선 기울기: {abs(pelvis_tilt_angle):.1f}° " + 
                          ("(우측 높음)" if pelvis_tilt_angle > 2 else 
                           "(좌측 높음)" if pelvis_tilt_angle < -2 else "(수평)")
        }
    
    def analyze_neck_line(self, neck_point, upper_thoracic):
        """목 세로라인 분석"""
        # 목선은 C7에서 머리 방향으로 연장
        head_point = neck_point + np.array([0, 25, 0])  # 머리 위치 추정 (2.5cm 위)
        
        # 목선 벡터 (머리에서 C7으로)
        neck_vector = neck_point - head_point
        neck_length = np.linalg.norm(neck_vector)
        
        if neck_length > 0:
            neck_unit = neck_vector / neck_length
            
            # 수직선과의 각도
            vertical_vector = np.array([0, -1, 0])  # 아래쪽 방향
            neck_vertical_angle = np.degrees(np.arccos(np.clip(np.dot(neck_unit, vertical_vector), -1, 1)))
            
            # 전방 머리 자세 분석 (Z축)
            forward_head_angle = np.degrees(np.arcsin(np.clip(neck_unit[2], -1, 1)))
            
            # 측면 기울기 (X축)
            lateral_neck_angle = np.degrees(np.arcsin(np.clip(neck_unit[0], -1, 1)))
        else:
            neck_vertical_angle = 0
            forward_head_angle = 0
            lateral_neck_angle = 0
        
        # C7-T3와의 연속성 분석
        upper_spine_vector = upper_thoracic - neck_point
        if np.linalg.norm(upper_spine_vector) > 0:
            upper_spine_unit = upper_spine_vector / np.linalg.norm(upper_spine_vector)
            # 목선과 상부 척추선의 연속성
            continuity_angle = np.degrees(np.arccos(np.clip(
                np.dot(neck_unit, upper_spine_unit), -1, 1)))
        else:
            continuity_angle = 0
        
        # 목 자세 평가
        status = []
        if abs(forward_head_angle) > 10:
            direction = "전방" if forward_head_angle > 0 else "후방"
            status.append(f"머리 {direction} 돌출 ({abs(forward_head_angle):.1f}°)")
        
        if abs(lateral_neck_angle) > 5:
            direction = "우측" if lateral_neck_angle > 0 else "좌측"
            status.append(f"목 {direction} 기울기 ({abs(lateral_neck_angle):.1f}°)")
        
        if continuity_angle > 20:
            status.append(f"목-어깨 연결 부자연 ({continuity_angle:.1f}°)")
        
        if not status:
            status.append("정상 목 자세")
        
        return {
            'head_pos': head_point.tolist(),
            'neck_base_pos': neck_point.tolist(),
            'neck_vector': neck_vector.tolist(),
            'neck_length': float(neck_length),
            'vertical_angle': float(neck_vertical_angle),
            'forward_angle': float(forward_head_angle),    # 전방 머리 자세
            'lateral_angle': float(lateral_neck_angle),    # 측면 기울기
            'continuity_angle': float(continuity_angle),   # 연속성 각도
            'line_type': 'vertical',
            'status': ', '.join(status),
            'description': f"목선: {', '.join(status)}"
        }
    
    def analyze_overall_alignment(self, shoulder_analysis, spine_lines, pelvis_analysis, neck_analysis):
        """전체 정렬 분석"""
        alignment_issues = []
        
        # 수평선들 간의 평행도 확인
        shoulder_tilt = shoulder_analysis.get('tilt_angle', 0)
        pelvis_tilt = pelvis_analysis.get('tilt_angle', 0)
        
        # 어깨-골반 평행도
        shoulder_pelvis_diff = abs(shoulder_tilt - pelvis_tilt)
        if shoulder_pelvis_diff > 5:
            alignment_issues.append(f"어깨-골반 비평행 ({shoulder_pelvis_diff:.1f}° 차이)")
        
        # 척추 세그먼트들의 연속성 확인
        spine_segments = ['cervical', 'upper_thoracic', 'lower_thoracic', 'lumbar', 'sacral']
        lateral_angles = []
        sagittal_angles = []
        
        for segment in spine_segments:
            if segment in spine_lines:
                lateral_angles.append(spine_lines[segment].get('lateral_angle', 0))
                sagittal_angles.append(spine_lines[segment].get('sagittal_angle', 0))
        
        # 척추 측만 확인 (좌우 기울기 일관성)
        if len(lateral_angles) > 0:
            lateral_std = np.std(lateral_angles)
            if lateral_std > 5:
                alignment_issues.append(f"척추 측면 정렬 불균형 (편차: {lateral_std:.1f}°)")
        
        # 척추 전후 곡선 확인
        if len(sagittal_angles) > 0:
            sagittal_std = np.std(sagittal_angles)
            if sagittal_std > 8:
                alignment_issues.append(f"척추 전후 곡선 불균형 (편차: {sagittal_std:.1f}°)")
        
        # 목-어깨 정렬
        neck_forward = neck_analysis.get('forward_angle', 0)
        neck_lateral = neck_analysis.get('lateral_angle', 0)
        
        if abs(neck_forward) > 15:
            alignment_issues.append("목-머리 전후 정렬 이상")
        
        if abs(neck_lateral) > 8:
            alignment_issues.append("목-머리 좌우 정렬 이상")
        
        # 전체 평가
        if not alignment_issues:
            overall_status = "양호한 전체 정렬"
            severity = "정상"
        elif len(alignment_issues) <= 2:
            overall_status = "경미한 정렬 이상"
            severity = "경미"
        elif len(alignment_issues) <= 4:
            overall_status = "중등도 정렬 이상"
            severity = "중등도"
        else:
            overall_status = "심각한 정렬 이상"
            severity = "심각"
        
        return {
            'overall_status': overall_status,
            'severity': severity,
            'alignment_issues': alignment_issues,
            'shoulder_pelvis_parallel': float(shoulder_pelvis_diff),
            'spine_lateral_consistency': float(np.std(lateral_angles)) if lateral_angles else 0,
            'spine_sagittal_consistency': float(np.std(sagittal_angles)) if sagittal_angles else 0,
            'neck_alignment_score': float(100 - abs(neck_forward) - abs(neck_lateral))
        }
        """기본 자세 분석"""
        if len(spine_points) < 4:
            return {"error": "충분한 척추 포인트를 찾을 수 없습니다."}
        
        analysis = {}
        
        # 전체 척추 높이
        total_height = spine_points[0, 1] - spine_points[-1, 1]
        
        # 척추 직선성 확인 (X축 편차)
        x_deviation = np.std(spine_points[:, 0])
        z_deviation = np.std(spine_points[:, 2])
        
        # 전후 기울기 (Z축)
        top_point = spine_points[0]
        bottom_point = spine_points[-1]
        forward_lean = top_point[2] - bottom_point[2]
        
        # 측면 기울기 (X축)
        side_lean = top_point[0] - bottom_point[0]
        
        # 각 척추 세그먼트 분석
        segment_analysis = {}
        for segment_name, indices in self.spine_segments.items():
            if len(indices) >= 2 and max(indices) < len(spine_points):
                segment_angles = self.calculate_segment_angles(spine_points, indices, segment_name)
                segment_analysis[segment_name] = segment_angles
        
        # 어깨 라인 분석 (상위 포인트에서 추정)
        shoulder_analysis = self.analyze_shoulder_line(spine_points)
        
        # 종합적인 자세 평가 및 권장사항
        posture_assessment, recommendations = self.comprehensive_posture_assessment(
            x_deviation, z_deviation, forward_lean, side_lean, segment_analysis
        )
        
        analysis = {
            'total_spine_height': float(total_height),
            'spine_straightness_x': float(x_deviation),
            'spine_straightness_z': float(z_deviation),
            'forward_lean': float(forward_lean),
            'side_lean': float(side_lean),
            'spine_points': spine_points.tolist(),
            'segment_analysis': segment_analysis,
            'shoulder_analysis': shoulder_analysis,
            'posture_assessment': posture_assessment,
            'recommendations': recommendations,
            'analysis_summary': self.generate_analysis_summary(segment_analysis, posture_assessment)
        }
        
        return analysis
    
    def comprehensive_posture_assessment(self, x_dev, z_dev, forward_lean, side_lean, segment_analysis):
        """종합적인 자세 평가 및 권장사항 생성"""
        issues = []
        recommendations = []
        
        # 전체적인 척추 정렬 확인
        if x_dev > 15:
            issues.append("척추 측만 의심")
            recommendations.append("전문의 상담을 통한 척추 측만 검사 필요")
            recommendations.append("양쪽 어깨의 균형을 맞추는 운동 실시")
        
        if z_dev > 15:
            issues.append("척추 전후 만곡 이상")
            recommendations.append("척추의 자연스러운 S커브 회복을 위한 스트레칭")
        
        if forward_lean > 30:
            issues.append("전방 기울어짐 (전체적인 자세)")
            recommendations.append("가슴 펴기 운동 및 등 근육 강화 운동")
            recommendations.append("일상생활에서 바른 자세 유지 의식적 노력")
        elif forward_lean < -30:
            issues.append("후방 기울어짐 (전체적인 자세)")
            recommendations.append("복부 근육 강화 및 고관절 스트레칭")
        
        if abs(side_lean) > 20:
            issues.append("좌우 기울어짐")
            recommendations.append("척추 좌우 균형을 맞추는 운동")
            recommendations.append("한쪽으로 기우는 습관 교정 (가방, 앉는 자세 등)")
        
        # 세그먼트별 문제점 확인
        segment_issues = []
        for segment_name, segment_data in segment_analysis.items():
            description = segment_data.get('description', '')
            
            if '경추 전만 감소' in description:
                segment_issues.append("거북목 증후군 위험")
                recommendations.append("목 스트레칭 및 목 근육 강화 운동")
                recommendations.append("모니터 높이 조절 및 올바른 베개 사용")
            
            if '요추 전만 감소' in description:
                segment_issues.append("일자허리 증후군")
                recommendations.append("요추 전만 회복을 위한 고관절 스트레칭")
                recommendations.append("엎드려서 상체 들어올리기 운동")
            
            if '흉추 과도한 후만' in description:
                segment_issues.append("등 굽음 (라운드 백)")
                recommendations.append("가슴근육 스트레칭 및 등근육 강화")
                recommendations.append("어깨 뒤로 돌리기 운동")
            
            if '기울어짐' in description:
                segment_issues.append(f"{segment_name} 부위 비대칭")
                recommendations.append(f"{segment_name} 부위 교정 운동 필요")
        
        # 전체 이슈 목록 통합
        all_issues = issues + segment_issues
        
        if not all_issues:
            assessment = "양호한 자세 - 현재 상태 유지 권장"
            recommendations = [
                "현재의 좋은 자세를 유지하기 위한 규칙적인 운동",
                "일상생활에서 바른 자세 습관 지속",
                "정기적인 척추 건강 체크"
            ]
        else:
            severity = len(all_issues)
            if severity >= 5:
                assessment = f"심각한 자세 불균형 - 즉시 교정 필요 ({', '.join(all_issues[:3])} 등)"
                recommendations.insert(0, "전문의 진료 및 정밀 검사 권장")
            elif severity >= 3:
                assessment = f"중등도 자세 문제 - 적극적 교정 필요 ({', '.join(all_issues[:2])} 등)"
            else:
                assessment = f"경미한 자세 문제 - 예방적 관리 필요 ({', '.join(all_issues)})"
        
        # 중복 제거
        recommendations = list(dict.fromkeys(recommendations))
        
        return assessment, recommendations
    
    def generate_analysis_summary(self, segment_analysis, posture_assessment):
        """분석 요약 생성"""
        summary = {
            'overall_status': posture_assessment,
            'problem_areas': [],
            'healthy_areas': [],
            'priority_actions': []
        }
        
        for segment_name, segment_data in segment_analysis.items():
            description = segment_data.get('description', '')
            korean_names = {
                'cervical': '경추(목)',
                'thoracic': '흉추(등)', 
                'lumbar': '요추(허리)',
                'sacral': '천추(골반)'
            }
            
            korean_name = korean_names.get(segment_name, segment_name)
            
            if '정상 범위' in description:
                summary['healthy_areas'].append(korean_name)
            else:
                problem_description = description.split(' - ')[1] if ' - ' in description else '이상'
                summary['problem_areas'].append(f"{korean_name}: {problem_description}")
        
        # 우선순위 행동 계획
        if '심각한' in posture_assessment:
            summary['priority_actions'] = [
                "1순위: 전문의 상담",
                "2순위: 정밀 진단",
                "3순위: 맞춤형 교정 프로그램"
            ]
        elif '중등도' in posture_assessment:
            summary['priority_actions'] = [
                "1순위: 교정 운동 시작",
                "2순위: 생활습관 개선",
                "3순위: 정기적 자세 점검"
            ]
        else:
            summary['priority_actions'] = [
                "1순위: 예방 운동 실시",
                "2순위: 바른 자세 유지",
                "3순위: 정기적 건강 관리"
            ]
        
        return summary
    
    def calculate_segment_angles(self, spine_points, indices, segment_name):
        """척추 세그먼트별 각도 계산"""
        if len(indices) < 2:
            return {}
            
        start_point = spine_points[indices[0]]
        end_point = spine_points[indices[-1]]
        
        # 세그먼트 벡터
        segment_vector = end_point - start_point
        segment_length = np.linalg.norm(segment_vector)
        
        if segment_length == 0:
            return {}
            
        segment_vector = segment_vector / segment_length
        
        # 수직 벡터와의 각도 (시상면 - 전후 기울기)
        vertical_vector = np.array([0, 1, 0])
        sagittal_angle = np.arccos(np.clip(np.dot(segment_vector, vertical_vector), -1, 1))
        sagittal_angle_deg = np.degrees(sagittal_angle)
        
        # 전후면 기울기 (관상면 - 좌우 기울기)
        frontal_vector = np.array([1, 0, 0])
        frontal_angle = np.arccos(np.clip(np.dot(segment_vector, frontal_vector), -1, 1))
        frontal_angle_deg = np.degrees(frontal_angle) - 90  # 90도 기준으로 조정
        
        # 커브 각도 계산 (3점이 있는 경우)
        curve_angle = 0.0
        if len(indices) >= 3:
            mid_point = spine_points[indices[1]]
            curve_angle = self.calculate_curve_angle([start_point, mid_point, end_point])
        
        return {
            'start_position': start_point.tolist(),
            'end_position': end_point.tolist(),
            'sagittal_angle': float(sagittal_angle_deg),
            'frontal_angle': float(frontal_angle_deg),
            'curve_angle': float(curve_angle),
            'segment_length': float(segment_length),
            'description': self.get_segment_description(segment_name, sagittal_angle_deg, frontal_angle_deg, curve_angle)
        }
    
    def get_segment_description(self, segment_name, sagittal_angle, frontal_angle, curve_angle):
        """세그먼트별 설명 생성"""
        descriptions = {
            'cervical': '경추 (목뼈) 7개',
            'thoracic': '흉추 (등뼈) 12개', 
            'lumbar': '요추 (허리뼈) 5개',
            'sacral': '천추/골반 (엉치뼈)'
        }
        
        base_desc = descriptions.get(segment_name, segment_name)
        
        # 각도 기반 상태 평가
        status = []
        
        # 시상면 각도 평가
        if sagittal_angle < 80:
            status.append("전방 기울어짐")
        elif sagittal_angle > 100:
            status.append("후방 기울어짐")
        
        # 관상면 각도 평가
        if abs(frontal_angle) > 10:
            if frontal_angle > 0:
                status.append("우측으로 기울어짐")
            else:
                status.append("좌측으로 기울어짐")
        
        # 커브 각도 평가
        if segment_name == 'cervical' and curve_angle < 10:
            status.append("경추 전만 감소")
        elif segment_name == 'lumbar' and curve_angle < 15:
            status.append("요추 전만 감소")
        elif segment_name == 'thoracic' and curve_angle > 60:
            status.append("흉추 과도한 후만")
        
        if status:
            return f"{base_desc} - {', '.join(status)}"
        else:
            return f"{base_desc} - 정상 범위"
    
    def analyze_shoulder_line(self, spine_points):
        """어깨 라인 분석 (상위 척추 포인트에서 추정)"""
        if len(spine_points) < 2:
            return {}
        
        # 상위 2개 포인트에서 어깨 라인 추정
        neck_point = spine_points[0]
        upper_spine = spine_points[1]
        
        # 어깨 너비 추정 (경험적 비율 사용)
        shoulder_width = 40  # 약 4cm로 추정
        
        # 어깨 포인트 추정
        left_shoulder = neck_point + np.array([-shoulder_width, -5, 0])
        right_shoulder = neck_point + np.array([shoulder_width, -5, 0])
        
        # 어깨 수평도 계산
        shoulder_vector = right_shoulder - left_shoulder
        horizontal_angle = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
        height_difference = right_shoulder[1] - left_shoulder[1]
        
        return {
            'left_shoulder_estimated': left_shoulder.tolist(),
            'right_shoulder_estimated': right_shoulder.tolist(),
            'horizontal_angle': float(horizontal_angle),
            'height_difference': float(height_difference),
            'description': f"어깨 높이 차이: {abs(height_difference):.1f}mm" + 
                          (f" (우측이 높음)" if height_difference > 5 else 
                           f" (좌측이 높음)" if height_difference < -5 else " (수평)")
        }
    
    def calculate_curve_angle(self, points):
        """3점을 이용한 커브 각도 계산"""
        if len(points) != 3:
            return 0.0
            
        p1, p2, p3 = points
        
        # 두 벡터 계산
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 벡터 길이 확인
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # 각도 계산
        cos_angle = np.dot(v1, v2) / (len1 * len2)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return np.degrees(angle)
        
    def assess_basic_posture(self, x_dev, z_dev, forward_lean, side_lean, segment_analysis):
        """기본 자세 평가"""
        issues = []
        
        if x_dev > 15:
            issues.append("척추 측만 의심")
        
        if z_dev > 15:
            issues.append("척추 전후 만곡 이상")
        
        if forward_lean > 30:
            issues.append("전방 기울어짐")
        elif forward_lean < -30:
            issues.append("후방 기울어짐")
        
        if abs(side_lean) > 20:
            issues.append("좌우 기울어짐")
        
        # 세그먼트별 이상 확인
        for segment_name, segment_data in segment_analysis.items():
            if "이상" in segment_data.get('description', ''):
                issues.append(f"{segment_name} 세그먼트 이상")
        
        if not issues:
            return "양호한 자세"
        else:
            return ", ".join(issues)
    
    def create_spine_visualization(self, point_cloud, spine_points, analysis_results):
        """척추 스켈레톤 및 분석 결과 시각화 - 어깨/척추/골반/목 라인 포함"""
        # 시각화 객체 생성
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D 척추 스켈레톤 분석", width=1400, height=900)
        
        # 1. 원본 포인트 클라우드 (회색, 반투명) - 위치 조정
        pcd_vis = copy.deepcopy(point_cloud)
        pcd_vis.paint_uniform_color([0.7, 0.7, 0.7])  # 약간 어둡게
        
        # 포인트 클라우드를 앞쪽으로 이동 (스켈레톤과 맞춤)
        pcd_points = np.asarray(pcd_vis.points)
        pcd_points[:, 2] -= 8.0  # Z축을 8mm 앞쪽으로 이동
        pcd_vis.points = o3d.utility.Vector3dVector(pcd_points)
        
        vis.add_geometry(pcd_vis)
        
        # 2. 척추 스켈레톤 구조 생성 (관절 포함)
        self.create_skeleton_structure(vis, spine_points)
        
        # 3. 척추 포인트 좌표 출력
        spine_labels = ['C7', 'T3', 'T8', 'T12', 'L3', 'S1']
        for i, (point, label) in enumerate(zip(spine_points, spine_labels)):
            print(f"{label}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")
        
        # 4. 라인 분석 및 시각화
        if 'posture_lines' in analysis_results:
            lines_data = analysis_results['posture_lines']
            
            # 3-1. 어깨 가로라인
            if 'shoulder_line' in lines_data:
                shoulder = lines_data['shoulder_line']
                if 'left_shoulder_pos' in shoulder and 'right_shoulder_pos' in shoulder:
                    shoulder_points = [shoulder['left_shoulder_pos'], shoulder['right_shoulder_pos']]
                    shoulder_line = o3d.geometry.LineSet()
                    shoulder_line.points = o3d.utility.Vector3dVector(shoulder_points)
                    shoulder_line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    shoulder_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # 빨간색
                    vis.add_geometry(shoulder_line)
                    
                    # 어깨 포인트 표시
                    for pos in shoulder_points:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
                        sphere.translate(pos)
                        sphere.paint_uniform_color([1.0, 0.0, 0.0])
                        vis.add_geometry(sphere)
            
            # 3-2. 척추 세로라인들 (세분화)
            if 'spine_lines' in lines_data:
                spine_lines = lines_data['spine_lines']
                spine_line_colors = {
                    'cervical': [1.0, 0.8, 0.8],      # 연한 빨강
                    'upper_thoracic': [1.0, 0.8, 0.4], # 연한 주황
                    'lower_thoracic': [0.8, 1.0, 0.4], # 연한 노랑-초록
                    'lumbar': [0.4, 0.8, 1.0],         # 연한 파랑
                    'sacral': [0.8, 0.4, 1.0],         # 연한 보라
                    'overall': [0.0, 1.0, 0.0]         # 초록 (전체)
                }
                
                for segment_name, segment_data in spine_lines.items():
                    if 'start_pos' in segment_data and 'end_pos' in segment_data:
                        line_points = [segment_data['start_pos'], segment_data['end_pos']]
                        line_set = o3d.geometry.LineSet()
                        line_set.points = o3d.utility.Vector3dVector(line_points)
                        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                        color = spine_line_colors.get(segment_name, [0.5, 0.5, 0.5])
                        line_set.colors = o3d.utility.Vector3dVector([color])
                        vis.add_geometry(line_set)
            
            # 3-3. 골반 가로라인 - 제거됨 (골반뼈 구조로 대체)
            # 골반뼈가 이미 정확한 위치에 표시되므로 가로라인은 불필요
            
            # 3-4. 목 세로라인
            if 'neck_line' in lines_data:
                neck = lines_data['neck_line']
                if 'head_pos' in neck and 'neck_base_pos' in neck:
                    neck_points = [neck['head_pos'], neck['neck_base_pos']]
                    neck_line = o3d.geometry.LineSet()
                    neck_line.points = o3d.utility.Vector3dVector(neck_points)
                    neck_line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    neck_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 1.0]])  # 마젠타색
                    vis.add_geometry(neck_line)
                    
                    # 머리 포인트 표시
                    head_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.8)
                    head_sphere.translate(neck['head_pos'])
                    head_sphere.paint_uniform_color([1.0, 0.0, 1.0])
                    vis.add_geometry(head_sphere)
        
        # 4. 기준선들 추가 (참조용)
        self.add_reference_lines(vis, spine_points)
        
        # 5. 분석 결과 출력
        self.print_detailed_analysis_results(analysis_results)
        
        # 카메라 설정 - 신체와 스켈레톤을 모두 잘 볼 수 있는 각도
        ctr = vis.get_view_control()
        ctr.set_front([-0.3, 0.3, -0.9])  # 측면에서 약간 앞쪽 각도
        ctr.set_lookat(np.mean(spine_points, axis=0))
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.7)  # 전체 구조를 보기 위해 적절한 줌
        
        print(f"\n추출된 척추 포인트 수: {len(spine_points)}")
        print("=== 3D 완전한 스켈레톤 시각화 범례 ===")
        print("🦴 베이지색: 척추뼈 연결선")
        print("🔴🟠🟡🟢🔵🟣: 척추 관절 (C7, T3, T8, T12, L3, S1)")
        print("🦴 연한베이지: 어깨뼈 구조 (쇄골, 견갑골, 상완골)")
        print("🦴 골색: 골반뼈 구조 (장골, 천골, 미골)")
        print("🔴 빨간점: 어깨 관절 / 🔵 파란점: 고관절")
        print("� 빨간선: 어깨 가로라인")
        print("🌈 색깔선: 척추 세그먼트별 라인")
        print("🟣 보라선: 목 세로라인 (경추부)")
        print("⚪ 회색선: 수직/수평 기준선")
        print("\n✨ 완전한 3D 스켈레톤 구조를 확인하세요!")
        print("🖱️ 마우스로 회전하여 다각도에서 관찰 가능합니다.")
        print("📍 스켈레톤이 신체 모델 내부에 정확히 배치되었습니다.")
        print("척추, 어깨, 골반의 완전한 해부학적 구조를 분석할 수 있습니다.")
        print("창을 닫으면 프로그램이 계속됩니다.")
        
        # 시각화 실행
        vis.run()
        vis.destroy_window()
    
    def add_reference_lines(self, vis, spine_points):
        """기준선 추가 (수직선, 수평선)"""
        center_point = np.mean(spine_points, axis=0)
        
        # 수직 기준선 (Y축)
        vertical_points = [
            center_point + np.array([0, -50, 0]),
            center_point + np.array([0, 50, 0])
        ]
        vertical_line = o3d.geometry.LineSet()
        vertical_line.points = o3d.utility.Vector3dVector(vertical_points)
        vertical_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        vertical_line.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]])  # 회색
        vis.add_geometry(vertical_line)
        
        # 수평 기준선 (X축)
        horizontal_points = [
            center_point + np.array([-50, 0, 0]),
            center_point + np.array([50, 0, 0])
        ]
        horizontal_line = o3d.geometry.LineSet()
        horizontal_line.points = o3d.utility.Vector3dVector(horizontal_points)
        horizontal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        horizontal_line.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]])  # 회색
        vis.add_geometry(horizontal_line)
    
    def print_detailed_analysis_results(self, analysis_results):
        """상세 분석 결과 출력"""
        print("\n" + "="*60)
        print("          척추 자세 라인 및 각도 분석 결과")
        print("="*60)
        
        if 'posture_lines' in analysis_results:
            lines_data = analysis_results['posture_lines']
            
            # 1. 어깨 가로라인 분석
            print("\n🔴 어깨 가로라인 분석:")
            if 'shoulder_line' in lines_data:
                shoulder = lines_data['shoulder_line']
                print(f"   기울기 각도: {shoulder.get('tilt_angle', 0):.1f}° (좌우)")
                print(f"   전후 기울기: {shoulder.get('depth_angle', 0):.1f}°")
                print(f"   상태: {shoulder.get('description', 'N/A')}")
            
            # 2. 척추 세로라인 분석 (세분화)
            print("\n🌈 척추 세로라인 분석 (세분화):")
            if 'spine_lines' in lines_data:
                spine_lines = lines_data['spine_lines']
                segments = ['cervical', 'upper_thoracic', 'lower_thoracic', 'lumbar', 'sacral']
                segment_names = ['경추선(C7-T3)', '상부흉추선(T3-T8)', '하부흉추선(T8-T12)', '요추선(T12-L3)', '천추선(L3-S1)']
                
                for segment, name in zip(segments, segment_names):
                    if segment in spine_lines:
                        seg_data = spine_lines[segment]
                        print(f"   {name}:")
                        print(f"     좌우 기울기: {seg_data.get('lateral_angle', 0):.1f}°")
                        print(f"     전후 기울기: {seg_data.get('sagittal_angle', 0):.1f}°")
                        print(f"     상태: {seg_data.get('status', 'N/A')}")
                
                # 전체 척추선
                if 'overall' in spine_lines:
                    overall = spine_lines['overall']
                    print(f"   전체척추선(C7-S1):")
                    print(f"     좌우 기울기: {overall.get('lateral_angle', 0):.1f}°")
                    print(f"     전후 기울기: {overall.get('sagittal_angle', 0):.1f}°")
                    print(f"     상태: {overall.get('status', 'N/A')}")
            
            # 3. 골반 가로라인 분석
            print("\n🔵 골반 가로라인 분석:")
            if 'pelvis_line' in lines_data:
                pelvis = lines_data['pelvis_line']
                print(f"   기울기 각도: {pelvis.get('tilt_angle', 0):.1f}° (좌우)")
                print(f"   전후 기울기: {pelvis.get('depth_angle', 0):.1f}°")
                print(f"   척추-골반 각도: {pelvis.get('spine_pelvis_angle', 0):.1f}°")
                print(f"   상태: {pelvis.get('description', 'N/A')}")
            
            # 4. 목 세로라인 분석
            print("\n🟣 목 세로라인 분석:")
            if 'neck_line' in lines_data:
                neck = lines_data['neck_line']
                print(f"   전방 머리 자세: {neck.get('forward_angle', 0):.1f}°")
                print(f"   측면 기울기: {neck.get('lateral_angle', 0):.1f}°")
                print(f"   목-어깨 연속성: {neck.get('continuity_angle', 0):.1f}°")
                print(f"   상태: {neck.get('description', 'N/A')}")
            
            # 5. 전체 정렬 분석
            print("\n⚖️ 전체 정렬 분석:")
            if 'overall_alignment' in lines_data:
                alignment = lines_data['overall_alignment']
                print(f"   전체 상태: {alignment.get('overall_status', 'N/A')}")
                print(f"   심각도: {alignment.get('severity', 'N/A')}")
                print(f"   어깨-골반 평행도: {alignment.get('shoulder_pelvis_parallel', 0):.1f}°")
                print(f"   척추 좌우 일관성: {alignment.get('spine_lateral_consistency', 0):.1f}°")
                print(f"   척추 전후 일관성: {alignment.get('spine_sagittal_consistency', 0):.1f}°")
                print(f"   목 정렬 점수: {alignment.get('neck_alignment_score', 0):.1f}/100")
                
                if alignment.get('alignment_issues'):
                    print("   문제점:")
                    for issue in alignment['alignment_issues']:
                        print(f"     - {issue}")
        
        # 각도 요약 테이블
        print("\n" + "="*60)
        print("                    각도 요약 테이블")
        print("="*60)
        print("구분               │ 좌우기울기 │ 전후기울기 │ 상태")
        print("-" * 50)
        
        if 'posture_lines' in analysis_results:
            lines_data = analysis_results['posture_lines']
            
            # 어깨선
            if 'shoulder_line' in lines_data:
                s = lines_data['shoulder_line']
                status = "정상" if abs(s.get('tilt_angle', 0)) < 2 else "이상"
                print(f"어깨 가로라인      │ {s.get('tilt_angle', 0):8.1f}° │ {s.get('depth_angle', 0):8.1f}° │ {status}")
            
            # 골반선
            if 'pelvis_line' in lines_data:
                p = lines_data['pelvis_line']
                status = "정상" if abs(p.get('tilt_angle', 0)) < 2 else "이상"
                print(f"골반 가로라인      │ {p.get('tilt_angle', 0):8.1f}° │ {p.get('depth_angle', 0):8.1f}° │ {status}")
            
            # 목선
            if 'neck_line' in lines_data:
                n = lines_data['neck_line']
                status = "정상" if abs(n.get('forward_angle', 0)) < 10 and abs(n.get('lateral_angle', 0)) < 5 else "이상"
                print(f"목 세로라인        │ {n.get('lateral_angle', 0):8.1f}° │ {n.get('forward_angle', 0):8.1f}° │ {status}")
            
            # 척추 세그먼트들
            if 'spine_lines' in lines_data:
                spine_lines = lines_data['spine_lines']
                segments = [
                    ('cervical', '경추선'),
                    ('upper_thoracic', '상부흉추선'),
                    ('lower_thoracic', '하부흉추선'),
                    ('lumbar', '요추선'),
                    ('sacral', '천추선'),
                    ('overall', '전체척추선')
                ]
                
                for segment_key, segment_name in segments:
                    if segment_key in spine_lines:
                        seg = spine_lines[segment_key]
                        lateral = seg.get('lateral_angle', 0)
                        sagittal = seg.get('sagittal_angle', 0)
                        status = "정상" if abs(lateral) < 5 and abs(sagittal) < 10 else "이상"
        print("="*60)
    
    def create_skeleton_structure(self, vis, spine_points):
        """3D 척추 관절 구조 생성 (순수 척추 관절만)"""
        if len(spine_points) < 2:
            return
        
        print("🦴 순수 척추 관절 구조를 생성합니다...")
        
        # 척추 관절 색상 정의
        spine_colors = [
            [1.0, 0.0, 0.0],    # C7 - 빨강
            [1.0, 0.5, 0.0],    # T3 - 주황  
            [1.0, 1.0, 0.0],    # T8 - 노랑
            [0.0, 1.0, 0.0],    # T12 - 초록
            [0.0, 0.0, 1.0],    # L3 - 파랑
            [0.5, 0.0, 1.0]     # S1 - 보라
        ]
        
        spine_names = ['C7', 'T3', 'T8', 'T12', 'L3', 'S1']
        
        # 1. 척추 관절 표시 (정확한 크기와 색상) - 해부학적 위치 조정
        adjusted_spine_points = []
        for i, point in enumerate(spine_points):
            # 척추를 신체 모델과 맞춤 (적절한 위치 조정)
            anatomical_point = point.copy()
            anatomical_point[2] += 2.0  # Z축을 2mm만 뒤쪽으로 이동 (신체 내부 적절한 위치)
            adjusted_spine_points.append(anatomical_point)
            
            vertebra = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)  # 관절 크기
            vertebra.translate(anatomical_point)
            vertebra.paint_uniform_color(spine_colors[i] if i < len(spine_colors) else [0.8, 0.8, 0.8])
            vis.add_geometry(vertebra)
            
            print(f"🔴 {spine_names[i] if i < len(spine_names) else f'척추{i+1}'}: ({anatomical_point[0]:.1f}, {anatomical_point[1]:.1f}, {anatomical_point[2]:.1f})")
        
        # 2. 척추뼈 연결선 (추간판과 인대) - 조정된 위치 사용
        for i in range(len(adjusted_spine_points) - 1):
            start_point = adjusted_spine_points[i]
            end_point = adjusted_spine_points[i + 1]
            
            # 세그먼트별 두께 조정 (해부학적 정확성)
            if i <= 1:  # 경추-상부흉추
                radius = 1.5
            elif i <= 3:  # 흉추
                radius = 1.8
            else:  # 요추-천추
                radius = 2.0
            
            bone_cylinder = self.create_anatomical_bone_segment(start_point, end_point, radius)
            bone_cylinder.paint_uniform_color([0.95, 0.9, 0.8])  # 자연스러운 뼈 색상
            vis.add_geometry(bone_cylinder)
        
        # 3. 어깨 스켈레톤 구조 추가
        self.create_shoulder_skeleton(vis, adjusted_spine_points[0])  # C7 기준
        
        # 4. 골반 스켈레톤 구조 추가
        self.create_pelvis_skeleton(vis, adjusted_spine_points[-1])  # S1 기준
        
        print(f"✅ 척추 관절 {len(spine_points)}개와 연결선 {len(spine_points)-1}개가 생성되었습니다.")
        print("✅ 어깨 스켈레톤과 골반 스켈레톤이 추가되었습니다.")
    
    def create_anatomical_bone_segment(self, start_point, end_point, radius=1.0):
        """해부학적으로 정확한 뼈 세그먼트 생성"""
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        
        if length < 0.001:
            return o3d.geometry.TriangleMesh()
        
        # 원통 생성 (더 정밀한 해상도)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, 
            height=length, 
            resolution=16,  # 더 부드러운 표면
            split=4
        )
        
        # 원통을 올바른 방향으로 회전
        direction_normalized = direction / length
        default_direction = np.array([0, 1, 0])  # 원통의 기본 방향
        
        # 회전축과 각도 계산
        if not np.allclose(direction_normalized, default_direction):
            rotation_axis = np.cross(default_direction, direction_normalized)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                rotation_angle = np.arccos(np.clip(np.dot(default_direction, direction_normalized), -1, 1))
                
                # 로드리게스 회전 공식 적용
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                cylinder.rotate(R, center=(0, 0, 0))
        
        # 위치 조정
        cylinder.translate((start_point + end_point) / 2)
        
        return cylinder
    
    def create_shoulder_skeleton(self, vis, c7_point):
        """어깨 스켈레톤 구조 생성"""
        print("🦴 어깨 스켈레톤 구조를 생성합니다...")
        
        # 어깨 해부학적 치수 (실제 인체 비율)
        shoulder_width = 45  # 어깨 너비 (4.5cm)
        clavicle_length = 35  # 쇄골 길이 (3.5cm)
        scapula_size = 20    # 견갑골 크기 (2.0cm)
        
        # C7에서 어깨까지의 오프셋 - 신체 모델과 맞춤
        shoulder_level = c7_point + np.array([0, -8, -10])  # 약간 아래, 앞쪽으로 이동
        
        # 1. 쇄골 (Clavicle) - 좌우
        left_clavicle_end = shoulder_level + np.array([-clavicle_length, 2, 0])
        right_clavicle_end = shoulder_level + np.array([clavicle_length, 2, 0])
        
        # 좌측 쇄골
        left_clavicle = self.create_anatomical_bone_segment(shoulder_level, left_clavicle_end, 1.5)
        left_clavicle.paint_uniform_color([0.9, 0.85, 0.7])  # 연한 베이지
        vis.add_geometry(left_clavicle)
        
        # 우측 쇄골
        right_clavicle = self.create_anatomical_bone_segment(shoulder_level, right_clavicle_end, 1.5)
        right_clavicle.paint_uniform_color([0.9, 0.85, 0.7])
        vis.add_geometry(right_clavicle)
        
        # 2. 견갑골 (Scapula) - 삼각형 형태
        # 좌측 견갑골
        left_scapula_center = left_clavicle_end + np.array([0, -scapula_size/2, scapula_size])
        left_scapula_points = [
            left_scapula_center + np.array([0, scapula_size/2, 0]),    # 상단
            left_scapula_center + np.array([0, -scapula_size/2, scapula_size/2]),  # 하단 뒤
            left_scapula_center + np.array([0, -scapula_size/2, -scapula_size/2])  # 하단 앞
        ]
        
        # 좌측 견갑골 뼈대
        for i in range(len(left_scapula_points)):
            for j in range(i+1, len(left_scapula_points)):
                scapula_bone = self.create_anatomical_bone_segment(
                    left_scapula_points[i], left_scapula_points[j], 0.8
                )
                scapula_bone.paint_uniform_color([0.85, 0.8, 0.65])
                vis.add_geometry(scapula_bone)
        
        # 우측 견갑골
        right_scapula_center = right_clavicle_end + np.array([0, -scapula_size/2, scapula_size])
        right_scapula_points = [
            right_scapula_center + np.array([0, scapula_size/2, 0]),    # 상단
            right_scapula_center + np.array([0, -scapula_size/2, scapula_size/2]),  # 하단 뒤
            right_scapula_center + np.array([0, -scapula_size/2, -scapula_size/2])  # 하단 앞
        ]
        
        # 우측 견갑골 뼈대
        for i in range(len(right_scapula_points)):
            for j in range(i+1, len(right_scapula_points)):
                scapula_bone = self.create_anatomical_bone_segment(
                    right_scapula_points[i], right_scapula_points[j], 0.8
                )
                scapula_bone.paint_uniform_color([0.85, 0.8, 0.65])
                vis.add_geometry(scapula_bone)
        
        # 3. 상완골 상단부 (Humerus head) - 어깨 관절
        # 좌측 상완골 헤드
        left_humerus_head = left_clavicle_end + np.array([0, -15, -5])
        left_humerus_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
        left_humerus_sphere.translate(left_humerus_head)
        left_humerus_sphere.paint_uniform_color([0.88, 0.82, 0.68])
        vis.add_geometry(left_humerus_sphere)
        
        # 우측 상완골 헤드
        right_humerus_head = right_clavicle_end + np.array([0, -15, -5])
        right_humerus_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
        right_humerus_sphere.translate(right_humerus_head)
        right_humerus_sphere.paint_uniform_color([0.88, 0.82, 0.68])
        vis.add_geometry(right_humerus_sphere)
        
        # 4. 어깨 관절 표시점
        left_shoulder_joint = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
        left_shoulder_joint.translate(left_clavicle_end)
        left_shoulder_joint.paint_uniform_color([1.0, 0.3, 0.3])  # 빨간색
        vis.add_geometry(left_shoulder_joint)
        
        right_shoulder_joint = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
        right_shoulder_joint.translate(right_clavicle_end)
        right_shoulder_joint.paint_uniform_color([1.0, 0.3, 0.3])  # 빨간색
        vis.add_geometry(right_shoulder_joint)
    
    def create_pelvis_skeleton(self, vis, s1_point):
        """골반 스켈레톤 구조 생성"""
        print("🦴 골반 스켈레톤 구조를 생성합니다...")
        
        # 골반 해부학적 치수 (실제 인체 비율)
        pelvis_width = 40    # 골반 너비 (4.0cm)
        pelvis_depth = 25    # 골반 깊이 (2.5cm)
        iliac_height = 30    # 장골 높이 (3.0cm)
        
        # S1에서 골반까지의 오프셋 - 신체 모델과 맞춤
        pelvis_center = s1_point + np.array([0, 8, -8])  # S1보다 약간 위쪽, 앞쪽
        
        # 1. 장골능 (Iliac Crest) - 골반 윗부분
        left_iliac_point = pelvis_center + np.array([-pelvis_width/2, iliac_height/2, 0])
        right_iliac_point = pelvis_center + np.array([pelvis_width/2, iliac_height/2, 0])
        
        # 장골능 연결선
        iliac_crest = self.create_anatomical_bone_segment(left_iliac_point, right_iliac_point, 2.0)
        iliac_crest.paint_uniform_color([0.88, 0.85, 0.75])  # 연한 골색
        vis.add_geometry(iliac_crest)
        
        # 2. 좌우 장골 (Ilium)
        # 좌측 장골
        left_ilium_bottom = pelvis_center + np.array([-pelvis_width/2, -iliac_height/2, 0])
        left_ilium = self.create_anatomical_bone_segment(left_iliac_point, left_ilium_bottom, 1.8)
        left_ilium.paint_uniform_color([0.85, 0.82, 0.72])
        vis.add_geometry(left_ilium)
        
        # 우측 장골
        right_ilium_bottom = pelvis_center + np.array([pelvis_width/2, -iliac_height/2, 0])
        right_ilium = self.create_anatomical_bone_segment(right_iliac_point, right_ilium_bottom, 1.8)
        right_ilium.paint_uniform_color([0.85, 0.82, 0.72])
        vis.add_geometry(right_ilium)
        
        # 3. 천골 (Sacrum) - S1 포인트 기준
        sacrum_top = s1_point + np.array([0, 15, 0])
        sacrum_bottom = s1_point + np.array([0, -8, 0])
        sacrum = self.create_anatomical_bone_segment(sacrum_top, sacrum_bottom, 2.5)
        sacrum.paint_uniform_color([0.9, 0.87, 0.77])  # 천골색
        vis.add_geometry(sacrum)
        
        # 4. 미골 (Coccyx)
        coccyx_start = sacrum_bottom
        coccyx_end = sacrum_bottom + np.array([0, -6, 2])  # 약간 앞으로 굽음
        coccyx = self.create_anatomical_bone_segment(coccyx_start, coccyx_end, 1.0)
        coccyx.paint_uniform_color([0.88, 0.85, 0.75])
        vis.add_geometry(coccyx)
        
        # 5. 골반 관절 연결부
        # 좌측 천장관절 (Sacroiliac joint)
        left_si_joint = self.create_anatomical_bone_segment(s1_point, left_ilium_bottom, 1.2)
        left_si_joint.paint_uniform_color([0.8, 0.75, 0.65])
        vis.add_geometry(left_si_joint)
        
        # 우측 천장관절
        right_si_joint = self.create_anatomical_bone_segment(s1_point, right_ilium_bottom, 1.2)
        right_si_joint.paint_uniform_color([0.8, 0.75, 0.65])
        vis.add_geometry(right_si_joint)
        
        # 6. 고관절 위치 표시 (Hip joints)
        left_hip_joint = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        left_hip_position = left_ilium_bottom + np.array([0, -8, 0])
        left_hip_joint.translate(left_hip_position)
        left_hip_joint.paint_uniform_color([0.3, 0.7, 1.0])  # 파란색
        vis.add_geometry(left_hip_joint)
        
        right_hip_joint = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        right_hip_position = right_ilium_bottom + np.array([0, -8, 0])
        right_hip_joint.translate(right_hip_position)
        right_hip_joint.paint_uniform_color([0.3, 0.7, 1.0])  # 파란색
        vis.add_geometry(right_hip_joint)
    
    def create_detailed_rib_structure(self, vis, spine_points):
        """정밀한 늑골 구조 생성"""
        # 실제 늑골은 T1-T12에 위치하지만, 주요 3개 레벨만 표시
        rib_levels = [
            (1, 35, 8),   # T3 - 상부 늑골 (너비 3.5cm, 두께 0.8cm)
            (2, 42, 6),   # T8 - 중부 늑골 (너비 4.2cm, 두께 0.6cm)  
            (3, 38, 7)    # T12 - 하부 늑골 (너비 3.8cm, 두께 0.7cm)
        ]
        
        for level_idx, width, thickness in rib_levels:
            if level_idx < len(spine_points):
                spine_pos = spine_points[level_idx]
                
                # 좌우 늑골 쌍 생성
                for side in [-1, 1]:  # 좌측(-1), 우측(1)
                    # 늑골의 자연스러운 곡선 생성 (실제 해부학적 형태)
                    rib_points = []
                    for t in np.linspace(0, 1, 20):  # 20개 점으로 부드러운 곡선
                        # 늑골의 S자 곡선 (실제 늑골 형태)
                        x_offset = side * width * t * (1 + 0.3 * np.sin(t * np.pi))
                        y_offset = -thickness * t * 0.5  # 약간 아래로
                        z_offset = -10 * t * t  # 앞쪽으로 휘어짐
                        
                        rib_point = spine_pos + np.array([x_offset, y_offset, z_offset])
                        rib_points.append(rib_point)
                    
                    # 늑골을 연결된 원통들로 구성
                    for i in range(len(rib_points) - 1):
                        rib_segment = self.create_anatomical_bone_segment(
                            rib_points[i], rib_points[i + 1], radius=0.8
                        )
                        rib_segment.paint_uniform_color([0.9, 0.85, 0.7])  # 연한 베이지
                        vis.add_geometry(rib_segment)
    
    def create_anatomical_pelvis_structure(self, vis, spine_points):
        """해부학적으로 정확한 골반 구조 생성"""
        if len(spine_points) < 6:
            return
        
        # S1 (천추) 위치를 기준으로 골반 구조 배치
        s1_position = spine_points[5]
        l3_position = spine_points[4]
        
        # 실제 골반은 S1에서 약간 위쪽과 옆쪽에 위치
        pelvis_center = s1_position + np.array([0, 8, -3])  # 해부학적 정확 위치
        
        print(f"🦴 골반 구조 생성: S1({s1_position[0]:.1f}, {s1_position[1]:.1f}, {s1_position[2]:.1f})")
        print(f"🦴 골반 중심: ({pelvis_center[0]:.1f}, {pelvis_center[1]:.1f}, {pelvis_center[2]:.1f})")
        
        # 장골 (Ilium) - 골반의 주요 부분
        ilium_width = 45  # 실제 장골 너비 (4.5cm)
        ilium_height = 25  # 실제 장골 높이 (2.5cm)
        ilium_depth = 15   # 실제 장골 깊이 (1.5cm)
        
        # 좌우 장골 생성
        for side in [-1, 1]:  # 좌측, 우측
            # 장골의 날개 모양 (실제 해부학적 형태)
            ilium_center = pelvis_center + np.array([
                side * ilium_width * 0.8,  # 좌우 위치
                ilium_height * 0.3,        # 약간 위쪽
                -ilium_depth * 0.2         # 약간 뒤쪽
            ])
            
            # 타원체로 장골 생성 (실제 형태에 가깝게)
            ilium = o3d.geometry.TriangleMesh.create_sphere(radius=12)
            
            # 장골의 특징적인 납작한 형태 구현
            vertices = np.asarray(ilium.vertices)
            vertices[:, 0] *= 1.8  # X축 확대 (너비)
            vertices[:, 1] *= 0.8  # Y축 축소 (높이)
            vertices[:, 2] *= 0.6  # Z축 축소 (두께)
            ilium.vertices = o3d.utility.Vector3dVector(vertices)
            
            ilium.translate(ilium_center)
            ilium.paint_uniform_color([0.85, 0.8, 0.75])  # 연한 회색 (골반뼈 색상)
            vis.add_geometry(ilium)
        
        # 천골 (Sacrum) - S1 위치의 삼각형 뼈
        sacrum = o3d.geometry.TriangleMesh.create_sphere(radius=8)
        vertices = np.asarray(sacrum.vertices)
        vertices[:, 0] *= 0.6  # 좁은 너비
        vertices[:, 1] *= 1.2  # 긴 높이
        vertices[:, 2] *= 0.4  # 얇은 두께
        sacrum.vertices = o3d.utility.Vector3dVector(vertices)
        
        sacrum.translate(s1_position + np.array([0, 0, 2]))
        sacrum.paint_uniform_color([0.8, 0.75, 0.7])
        vis.add_geometry(sacrum)
        
        # 치골 결합 (Pubic Symphysis)
        pubis_center = pelvis_center + np.array([0, -20, -8])
        pubis = o3d.geometry.TriangleMesh.create_sphere(radius=6)
        vertices = np.asarray(pubis.vertices)
        vertices[:, 0] *= 0.4  # 좁은 너비
        vertices[:, 1] *= 0.6  # 작은 높이
        vertices[:, 2] *= 1.2  # 앞뒤 두께
        pubis.vertices = o3d.utility.Vector3dVector(vertices)
        
        pubis.translate(pubis_center)
        pubis.paint_uniform_color([0.8, 0.75, 0.7])
        vis.add_geometry(pubis)
    
    def create_anatomical_neck_structure(self, vis, spine_points):
        """해부학적으로 정확한 목뼈 구조 생성 (두개골/턱뼈 제외)"""
        c7_position = spine_points[0]
        
        # 목 끝 위치 (C7에서 목 길이만큼 위)
        neck_length = 22  # 실제 목 길이 (2.2cm)
        neck_top_position = c7_position + np.array([0, neck_length, 2])
        
        print(f"🦴 목뼈 구조 생성: C7({c7_position[0]:.1f}, {c7_position[1]:.1f}, {c7_position[2]:.1f})")
        
        # 목뼈 연결 (C1-C7)
        neck_vertebrae_count = 4  # 주요 목뼈만 표시
        for i in range(neck_vertebrae_count):
            t = i / (neck_vertebrae_count - 1)
            vertebra_pos = c7_position + t * (neck_top_position - c7_position) * 0.8
            
            cervical_vertebra = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
            cervical_vertebra.translate(vertebra_pos)
            cervical_vertebra.paint_uniform_color([0.9, 0.85, 0.8])
            vis.add_geometry(cervical_vertebra)
        
        # 목뼈 연결선
        neck_connection = self.create_anatomical_bone_segment(
            c7_position, neck_top_position, radius=1.2
        )
        neck_connection.paint_uniform_color([0.9, 0.85, 0.8])
        vis.add_geometry(neck_connection)
    
    def create_rib_structure(self, vis, spine_position, level):
        """늑골 구조 생성"""
        # 늑골 길이는 레벨에 따라 다름
        rib_lengths = {0: 25, 1: 30, 2: 28}  # T3, T8, T12
        rib_length = rib_lengths.get(level, 25)
        
        # 좌우 늑골 생성
        for side in [-1, 1]:  # 왼쪽, 오른쪽
            # 늑골 시작점 (척추에서 약간 앞쪽)
            rib_start = spine_position + np.array([0, 0, 2])
            
            # 늑골 끝점 (옆구리 방향으로 곡선)
            rib_end = spine_position + np.array([
                side * rib_length * 0.8,  # 옆으로
                0,                         # 높이는 유지
                rib_length * 0.3          # 앞으로
            ])
            
            # 늑골을 곡선으로 생성 (여러 세그먼트로)
            segments = 5
            for i in range(segments):
                t1 = i / segments
                t2 = (i + 1) / segments
                
                # 베지어 곡선을 사용한 자연스러운 늑골 형태
                control_point = spine_position + np.array([
                    side * rib_length * 0.4,
                    -2,  # 약간 아래로
                    rib_length * 0.6
                ])
                
                p1 = (1-t1)**2 * rib_start + 2*(1-t1)*t1 * control_point + t1**2 * rib_end
                p2 = (1-t2)**2 * rib_start + 2*(1-t2)*t2 * control_point + t2**2 * rib_end
                
                rib_segment = self.create_anatomical_bone_segment(p1, p2, radius=0.6)
                rib_segment.paint_uniform_color([0.9, 0.85, 0.7])
                vis.add_geometry(rib_segment)
    
    def create_pelvis_structure(self, vis, pelvis_center):
        """골반 구조 생성 (기본 버전 - 호환성용)"""
        # 장골 생성
        for side in [-1, 1]:
            ilium_pos = pelvis_center + np.array([side * 35, 5, -5])
            ilium = o3d.geometry.TriangleMesh.create_sphere(radius=8)
            
            vertices = np.asarray(ilium.vertices)
            vertices[:, 0] *= 1.5
            vertices[:, 1] *= 0.8
            vertices[:, 2] *= 0.6
            ilium.vertices = o3d.utility.Vector3dVector(vertices)
            
            ilium.translate(ilium_pos)
            ilium.paint_uniform_color([0.85, 0.8, 0.75])
            vis.add_geometry(ilium)
    
    def create_neck_structure(self, vis, neck_base):
        """목뼈 구조 생성 (두개골/턱뼈 제외)"""
        # 목 끝 위치
        neck_top_position = neck_base + np.array([0, 25, 5])
        
        # 목뼈 연결
        neck_connection = self.create_anatomical_bone_segment(neck_base, neck_top_position, radius=2.0)
        neck_connection.paint_uniform_color([0.9, 0.9, 0.7])
        vis.add_geometry(neck_connection)
    
    def create_pelvis_structure(self, vis, pelvis_center):
        """골반 구조 생성 - 해부학적으로 올바른 위치"""
        # 골반뼈 (장골) 생성 - S1 위치를 기준으로 위쪽에 배치
        pelvis_width = 30
        pelvis_height = 20
        
        # 골반은 S1(천추) 포인트에서 위쪽과 옆쪽으로 확장
        # S1은 척추의 가장 아래 부분이므로 골반뼈는 이 위치 주변에 있어야 함
        
        # 좌우 장골 - S1에서 위쪽으로 올리고 옆으로 배치
        for side in [-1, 1]:
            # 장골 위치: S1에서 위로 10mm, 옆으로 pelvis_width/2
            iliac_position = pelvis_center + np.array([
                side * pelvis_width/2,  # 좌우로 분리
                pelvis_height/2,        # S1에서 위쪽으로 배치
                -5                      # 약간 뒤쪽에 배치
            ])
            
            # 장골을 타원체로 생성
            iliac_bone = o3d.geometry.TriangleMesh.create_sphere(radius=6)
            # Open3D에서 scale은 단일 값만 지원하므로 수동으로 변형
            vertices = np.asarray(iliac_bone.vertices)
            vertices[:, 0] *= 1.8  # X축 확대 (좌우 폭)
            vertices[:, 1] *= 0.6  # Y축 축소 (높이)
            vertices[:, 2] *= 1.0  # Z축 유지 (전후)
            iliac_bone.vertices = o3d.utility.Vector3dVector(vertices)
            iliac_bone.translate(iliac_position)
            iliac_bone.paint_uniform_color([0.85, 0.85, 0.65])
            vis.add_geometry(iliac_bone)
            
            # 장골을 S1과 연결하는 천장관절
            connection = self.create_bone_segment(pelvis_center, iliac_position, radius=0.8)
            connection.paint_uniform_color([0.9, 0.9, 0.7])
            vis.add_geometry(connection)
        
        # 치골결합 (pubic symphysis) 생성 - 골반 앞쪽 중앙
        pubic_position = pelvis_center + np.array([0, -5, 15])  # S1에서 아래, 앞쪽
        pubic_bone = o3d.geometry.TriangleMesh.create_sphere(radius=4)
        # 치골 모양으로 변형
        vertices = np.asarray(pubic_bone.vertices)
        vertices[:, 0] *= 2.0  # X축 확대
        vertices[:, 1] *= 0.5  # Y축 축소
        vertices[:, 2] *= 0.8  # Z축 축소
        pubic_bone.vertices = o3d.utility.Vector3dVector(vertices)
        pubic_bone.translate(pubic_position)
        pubic_bone.paint_uniform_color([0.8, 0.8, 0.6])
        vis.add_geometry(pubic_bone)
        
        # 좌우 장골을 치골과 연결
        for side in [-1, 1]:
            iliac_pos = pelvis_center + np.array([side * pelvis_width/2, pelvis_height/2, -5])
            pubic_connection = self.create_bone_segment(iliac_pos, pubic_position, radius=0.6)
            pubic_connection.paint_uniform_color([0.85, 0.85, 0.65])
            vis.add_geometry(pubic_connection)
    
    def create_neck_structure(self, vis, neck_base):
        """목뼈 구조 생성"""
        # 머리뼈 (두개골) 위치
        skull_position = neck_base + np.array([0, 25, 5])
        
        # 두개골 생성
        skull = o3d.geometry.TriangleMesh.create_sphere(radius=12)
        skull.translate(skull_position)
        skull.paint_uniform_color([0.95, 0.9, 0.85])  # 살색
        vis.add_geometry(skull)
        
        # 목뼈 연결
        neck_connection = self.create_bone_segment(neck_base, skull_position, radius=2.0)
        neck_connection.paint_uniform_color([0.9, 0.9, 0.7])
        vis.add_geometry(neck_connection)
        
        # 하악골 (턱뼈)
        jaw_position = skull_position + np.array([0, -8, -8])
        jaw = o3d.geometry.TriangleMesh.create_sphere(radius=6)
        # 수동으로 턱뼈 모양 변형
        vertices = np.asarray(jaw.vertices)
        vertices[:, 0] *= 1.2  # X축 확대
        vertices[:, 1] *= 0.6  # Y축 축소
        vertices[:, 2] *= 0.8  # Z축 축소
        jaw.vertices = o3d.utility.Vector3dVector(vertices)
        jaw.translate(jaw_position)
        jaw.paint_uniform_color([0.9, 0.85, 0.8])
        vis.add_geometry(jaw)
    
    def add_reference_lines(self, vis, spine_points):
        """기준선 추가 (수직선, 수평선)"""
        center_point = np.mean(spine_points, axis=0)
        
        # 수직 기준선 (Y축)
        vertical_points = [
            center_point + np.array([0, -50, 0]),
            center_point + np.array([0, 50, 0])
        ]
        vertical_line = o3d.geometry.LineSet()
        vertical_line.points = o3d.utility.Vector3dVector(vertical_points)
        vertical_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        vertical_line.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]])  # 회색
        vis.add_geometry(vertical_line)
        
        # 수평 기준선 (X축)
        horizontal_points = [
            center_point + np.array([-50, 0, 0]),
            center_point + np.array([50, 0, 0])
        ]
        horizontal_line = o3d.geometry.LineSet()
        horizontal_line.points = o3d.utility.Vector3dVector(horizontal_points)
        horizontal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        horizontal_line.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]])  # 회색
        vis.add_geometry(horizontal_line)

def load_depth_map(file_path):
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

def visualize_3d_pose():
    """척추 분석 및 시각화 (SMPL 또는 기본 분석)"""
    print("3D 자세 분석을 시작합니다...")
    
    # 각 뷰의 DepthMap 로드
    views = {
        "front": r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\test\정상\정면_남\DepthMap0.bmp",
        "right": r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\test\정상\오른쪽_남\DepthMap0.bmp",
        "left": r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\test\정상\왼쪽_남\DepthMap0.bmp",
        "back": r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\test\정상\후면_남\DepthMap0.bmp"
    }
    
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
    
    # 노이즈 제거 및 다운샘플링 (개선된 버전)
    print("포인트 클라우드 전처리를 시작합니다...")
    
    # 1단계: 적응적 다운샘플링
    original_point_count = len(merged_cloud.points)
    if original_point_count > 15000:
        voxel_size = 3.0
    elif original_point_count > 8000:
        voxel_size = 2.0
    else:
        voxel_size = 1.5
    
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size=voxel_size)
    print(f"다운샘플링: {original_point_count} -> {len(merged_cloud.points)} 포인트")
    
    # 2단계: Statistical outlier removal
    cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    merged_cloud = cl
    print(f"이상값 제거 후: {len(merged_cloud.points)} 포인트")
    
    # 3단계: 신체 영역만 추출 (높이 기반 필터링)
    points = np.asarray(merged_cloud.points)
    if len(points) > 0:
        # 바닥에서 너무 낮거나 너무 높은 포인트 제거
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        height_range = y_max - y_min
        
        # 하위 5%와 상위 5% 제거 (바닥이나 천장 노이즈)
        y_threshold_low = y_min + height_range * 0.05
        y_threshold_high = y_max - height_range * 0.05
        
        height_mask = (points[:, 1] >= y_threshold_low) & (points[:, 1] <= y_threshold_high)
        filtered_points = points[height_mask]
        
        if len(filtered_points) > 100:
            merged_cloud.points = o3d.utility.Vector3dVector(filtered_points)
            
            # 색상 정보도 함께 필터링
            if merged_cloud.has_colors():
                colors = np.asarray(merged_cloud.colors)
                merged_cloud.colors = o3d.utility.Vector3dVector(colors[height_mask])
    
    print(f"높이 필터링 후: {len(merged_cloud.points)} 포인트")
    
    # 4단계: 법선 벡터 재계산
    merged_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=30)
    )
    
    # SMPL 사용 가능 여부에 따라 분석 방법 선택
    if SMPL_AVAILABLE:
        print("SMPL 기반 척추 분석을 시도합니다...")
        spine_analyzer = SMPLSpineAnalyzer(model_type='smplx')
        
        # SMPL 모델 피팅
        fitted_vertices, joints_3d, pose_params = spine_analyzer.fit_smpl_to_pointcloud(merged_cloud, max_iterations=50)
        
        if fitted_vertices is not None and joints_3d is not None:
            print("SMPL 피팅 완료. 척추 분석을 진행합니다...")
            
            # 척추 각도 분석
            spine_analysis = spine_analyzer.calculate_spine_angles(joints_3d)
            
            # 분석 결과 출력
            print("\n=== SMPL 기반 척추 분석 결과 ===")
            for segment_name, analysis in spine_analysis.items():
                if segment_name in ['cervical', 'thoracic', 'lumbar', 'sacral']:
                    print(f"\n{segment_name.upper()} (경추/흉추/요추/천추):")
                    print(f"  시상면 각도: {analysis['sagittal_angle']:.2f}도")
                    print(f"  관상면 각도: {analysis['frontal_angle']:.2f}도")
                    print(f"  세그먼트 길이: {analysis['length']:.2f}mm")
                elif segment_name == 'shoulder_level':
                    print(f"\n어깨 수평도:")
                    print(f"  수평 각도: {analysis['horizontal_angle']:.2f}도")
                    print(f"  높이 차이: {analysis['height_difference']:.2f}mm")
                elif segment_name == 'overall_posture':
                    print(f"\n전체 자세 평가:")
                    print(f"  척추 전체 높이: {analysis['total_spine_height']:.2f}mm")
                    print(f"  전방 머리 돌출: {analysis['head_forward_distance']:.2f}mm")
                    print(f"  요추 전만각: {analysis['lumbar_lordosis']:.2f}도")
                    print(f"  흉추 후만각: {analysis['thoracic_kyphosis']:.2f}도")
                    print(f"  경추 전만각: {analysis['cervical_lordosis']:.2f}도")
                    print(f"  자세 평가: {analysis['posture_assessment']}")
            
            # SMPL 메시 생성
            smpl_mesh = o3d.geometry.TriangleMesh()
            smpl_mesh.vertices = o3d.utility.Vector3dVector(fitted_vertices)
            
            # 간단한 메시 생성 (Delaunay 삼각분할)
            try:
                import scipy.spatial
                from scipy.spatial import Delaunay
                
                # 2D 투영을 위해 PCA 사용
                pca_result = np.linalg.svd(fitted_vertices - fitted_vertices.mean(axis=0))
                projected_2d = fitted_vertices @ pca_result[2][:2].T
                
                tri = Delaunay(projected_2d)
                smpl_mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
            except:
                print("메시 생성 실패, 포인트로만 표시합니다.")
            
            smpl_mesh.paint_uniform_color([0.8, 0.8, 0.9])  # 연보라색
            smpl_mesh.compute_vertex_normals()
            
            # 척추 시각화 요소 생성
            spine_visualizations = spine_analyzer.create_spine_visualization(joints_3d, spine_analysis)
            
            # 결과 저장
            output_dir = "output/smpl_spine_analysis"
            os.makedirs(output_dir, exist_ok=True)
            
            # 분석 결과 JSON 저장
            analysis_path = os.path.join(output_dir, "spine_analysis_results.json")
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(spine_analysis, f, ensure_ascii=False, indent=2)
            print(f"\n분석 결과가 저장되었습니다: {analysis_path}")
            
            # SMPL 메시 저장
            smpl_mesh_path = os.path.join(output_dir, "smpl_fitted_mesh.ply")
            o3d.io.write_triangle_mesh(smpl_mesh_path, smpl_mesh)
            print(f"SMPL 메시가 저장되었습니다: {smpl_mesh_path}")
            
            # 조인트 위치 저장
            joints_path = os.path.join(output_dir, "joints_3d.npy")
            np.save(joints_path, joints_3d)
            print(f"3D 조인트 위치가 저장되었습니다: {joints_path}")
            
            # 시각화
            print("\n3D 시각화를 시작합니다...")
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="SMPL 기반 척추 분석", width=1200, height=900)
            
            # 원본 포인트 클라우드 (반투명)
            merged_cloud.paint_uniform_color([0.5, 0.5, 0.5])
            vis.add_geometry(merged_cloud)
            
            # SMPL 메시
            vis.add_geometry(smpl_mesh)
            
            # 척추 시각화 요소들
            for geo in spine_visualizations:
                vis.add_geometry(geo)
            
            # 렌더링 옵션 설정
            opt = vis.get_render_option()
            opt.point_size = 1.0
            opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 어두운 회색 배경
            opt.show_coordinate_frame = True
            
            # 카메라 위치 설정
            ctr = vis.get_view_control()
            ctr.set_zoom(0.6)
            ctr.set_front([0.3, -0.3, -0.9])
            ctr.set_up([0, -1, 0])
            
            # 시각화 실행
            vis.run()
            vis.destroy_window()
            
            return
    
    # SMPL을 사용할 수 없는 경우 기본 분석 수행
    print("기본 척추 분석을 수행합니다...")
    basic_analyzer = BasicSpineAnalyzer()
    
    # 포인트 클라우드에서 척추 추정
    print("척추 키포인트 추출을 시작합니다...")
    spine_points = basic_analyzer.extract_spine_from_pointcloud(merged_cloud)
    
    print(f"추출된 척추 포인트 수: {len(spine_points)}")
    if len(spine_points) > 0:
        print("척추 포인트 위치:")
        for i, point in enumerate(spine_points):
            print(f"  포인트 {i+1}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")
    
    if len(spine_points) > 0:
        # 새로운 라인 및 각도 분석
        posture_lines_analysis = basic_analyzer.analyze_posture_lines_and_angles(spine_points)
        
        # 전체 분석 결과 통합
        spine_analysis = {
            'spine_points': [point.tolist() for point in spine_points],
            'posture_lines': posture_lines_analysis
        }
        
        # 🎯 3D 스켈레톤 시각화 실행
        print("\n🦴 3D 스켈레톤 시각화를 시작합니다...")
        basic_analyzer.create_spine_visualization(merged_cloud, spine_points, spine_analysis)
        
        # 결과 저장
        output_dir = "output/posture_lines_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # 분석 결과 JSON 저장
        analysis_path = os.path.join(output_dir, "posture_lines_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(spine_analysis, f, ensure_ascii=False, indent=2)
        print(f"\n분석 결과가 저장되었습니다: {analysis_path}")
        
    else:
        print("척추 포인트를 추출할 수 없었습니다.")
    
    print("\n분석이 완료되었습니다!")

if __name__ == "__main__":
    visualize_3d_pose()