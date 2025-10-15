"""
3D 메쉬 기반 척추 관절 라인 정확 예측 시스템
MediaPipe BlazePose 3D + 메쉬 정보 결합
"""

import cv2
import mediapipe as mp
import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple, Optional
import json


class MediaPipeSpineDetector:
    """
    MediaPipe BlazePose를 활용한 3D 척추 관절 검출기
    - BlazePose Full (33 keypoints) 사용
    - 3D heatmap regression 기반 깊이 추정
    - Visibility 기반 가려진 관절 처리
    - Temporal smoothing으로 안정화
    """
    
    # MediaPipe 척추 관련 랜드마크 인덱스
    SPINE_LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_hip': 23,
        'right_hip': 24,
        # 추가 참조 포인트
        'left_ear': 7,
        'right_ear': 8,
        'mouth_left': 9,
        'mouth_right': 10,
    }
    
    # 척추 세그먼트 정의 (C7, T1-T12, L1-L5, Sacrum)
    SPINE_SEGMENTS = {
        'cervical': ['nose', 'left_shoulder', 'right_shoulder'],  # C7 영역
        'thoracic_upper': ['left_shoulder', 'right_shoulder'],     # T1-T6
        'thoracic_lower': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],  # T7-T12
        'lumbar': ['left_hip', 'right_hip'],                       # L1-L5
    }
    
    def __init__(self, 
                 static_image_mode=True,
                 model_complexity=2,  # 0, 1, 2 (2가 가장 정확)
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Args:
            static_image_mode: True면 각 이미지를 독립적으로 처리
            model_complexity: 0(Lite), 1(Full), 2(Heavy) - 2가 가장 정확
            smooth_landmarks: 랜드마크 스무딩 활성화
            min_detection_confidence: 최소 검출 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=True,  # 세그멘테이션 활성화
            smooth_segmentation=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Temporal smoothing을 위한 버퍼
        self.landmark_history = []
        self.history_size = 5
        
    def load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        """3D 메쉬 로드"""
        mesh = trimesh.load(mesh_path, force='mesh')
        print(f"메쉬 로드 완료: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
    
    def mesh_to_multiview_images(self, 
                                   mesh: trimesh.Trimesh,
                                   image_size: int = 512,
                                   views: List[str] = ['front', 'side', 'back', 'top']) -> Dict[str, np.ndarray]:
        """
        메쉬를 여러 각도의 2D 이미지로 렌더링
        
        Args:
            mesh: 3D 메쉬
            image_size: 렌더링 이미지 크기
            views: 렌더링할 뷰 리스트
            
        Returns:
            각 뷰별 렌더링 이미지 딕셔너리
        """
        images = {}
        
        # 메쉬 중심 및 스케일 정규화
        mesh_centered = mesh.copy()
        mesh_centered.vertices -= mesh_centered.centroid
        scale = np.max(np.abs(mesh_centered.vertices))
        mesh_centered.vertices /= scale
        
        # 각 뷰별 카메라 설정
        camera_configs = {
            'front': {
                'eye': [0, 0, 2],
                'center': [0, 0, 0],
                'up': [0, 1, 0]
            },
            'side': {
                'eye': [2, 0, 0],
                'center': [0, 0, 0],
                'up': [0, 1, 0]
            },
            'back': {
                'eye': [0, 0, -2],
                'center': [0, 0, 0],
                'up': [0, 1, 0]
            },
            'top': {
                'eye': [0, 2, 0],
                'center': [0, 0, 0],
                'up': [0, 0, -1]
            }
        }
        
        for view in views:
            if view not in camera_configs:
                continue
                
            config = camera_configs[view]
            
            # PyRender를 먼저 시도 (더 안정적)
            try:
                images[view] = self._render_with_pyrender(mesh_centered, config, image_size)
                print(f"  ✓ '{view}' 뷰 렌더링 완료 (PyRender)")
            except Exception as e:
                print(f"  PyRender 실패, Trimesh 시도 중... ({e})")
                
                # Trimesh 기본 렌더링 시도
                try:
                    scene = mesh_centered.scene()
                    
                    # PNG로 렌더링
                    png = scene.save_image(resolution=[image_size, image_size])
                    
                    # PIL Image로 변환
                    from PIL import Image
                    import io
                    
                    image = Image.open(io.BytesIO(png))
                    image = image.convert('RGB')
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    images[view] = image_bgr
                    print(f"  ✓ '{view}' 뷰 렌더링 완료 (Trimesh)")
                    
                except Exception as e2:
                    print(f"  ✗ '{view}' 뷰 렌더링 완전 실패: {e2}")
                    # 더미 이미지 생성
                    images[view] = np.ones((image_size, image_size, 3), dtype=np.uint8) * 128
        
        return images
    
    def _render_with_pyrender(self, mesh: trimesh.Trimesh, camera_config: dict, image_size: int) -> np.ndarray:
        """PyRender를 사용한 렌더링 (대체 방법)"""
        try:
            import pyrender
            
            # PyRender 메쉬 생성
            mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            
            # 씬 생성
            scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[255, 255, 255])
            scene.add(mesh_pr)
            
            # 조명 추가
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
            scene.add(light, pose=np.eye(4))
            
            # 추가 조명 (앞쪽)
            light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            light_pose = np.eye(4)
            light_pose[:3, 3] = camera_config['eye']
            scene.add(light2, pose=light_pose)
            
            # 카메라 설정
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            
            # 카메라 포즈 계산
            eye = np.array(camera_config['eye'])
            center = np.array(camera_config['center'])
            up = np.array(camera_config['up'])
            
            z = eye - center
            z = z / np.linalg.norm(z)
            x = np.cross(up, z)
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            
            camera_pose = np.eye(4)
            camera_pose[:3, 0] = x
            camera_pose[:3, 1] = y
            camera_pose[:3, 2] = z
            camera_pose[:3, 3] = eye
            
            scene.add(camera, pose=camera_pose)
            
            # 렌더링
            renderer = pyrender.OffscreenRenderer(image_size, image_size)
            color, depth = renderer.render(scene)
            renderer.delete()
            
            return color
            
        except ImportError:
            print("    PyRender를 사용할 수 없습니다.")
            # Matplotlib을 사용한 간단한 투영
            return self._render_with_matplotlib(mesh, camera_config, image_size)
        except Exception as e:
            print(f"    PyRender 렌더링 오류: {e}")
            return self._render_with_matplotlib(mesh, camera_config, image_size)
    
    def _render_with_matplotlib(self, mesh: trimesh.Trimesh, camera_config: dict, image_size: int) -> np.ndarray:
        """Matplotlib을 사용한 단순 투영 렌더링"""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # Figure 생성 (화면에 표시하지 않음)
        fig = plt.figure(figsize=(5, 5), dpi=image_size/5)
        ax = fig.add_subplot(111, projection='3d')
        
        # 메쉬 표면 그리기
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 간단한 표면 렌더링
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       triangles=faces, color='lightblue', alpha=0.8,
                       edgecolor='none', shade=True)
        
        # 카메라 뷰 설정
        eye = camera_config['eye']
        center = camera_config['center']
        
        # 뷰 각도 설정
        if abs(eye[0]) > abs(eye[2]):  # 측면 뷰
            ax.view_init(elev=0, azim=90 if eye[0] > 0 else -90)
        elif abs(eye[2]) > abs(eye[0]):  # 정면/후면 뷰
            ax.view_init(elev=0, azim=0 if eye[2] > 0 else 180)
        else:  # 상단 뷰
            ax.view_init(elev=90, azim=0)
        
        # 축 숨기기
        ax.set_axis_off()
        
        # 같은 스케일
        max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                             vertices[:, 1].max()-vertices[:, 1].min(),
                             vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
        mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 캔버스를 이미지로 변환
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        
        plt.close(fig)
        
        # RGB로 변환 (알파 채널 제거)
        image_rgb = image[:, :, :3]
        
        # OpenCV BGR로 변환
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    def detect_pose_landmarks(self, image: np.ndarray) -> Optional[mp.solutions.pose.PoseLandmark]:
        """
        이미지에서 포즈 랜드마크 검출
        
        Args:
            image: BGR 이미지
            
        Returns:
            랜드마크 결과 또는 None
        """
        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe 처리
        results = self.pose.process(image_rgb)
        
        return results
    
    def extract_3d_landmarks(self, results) -> Dict[str, Dict[str, float]]:
        """
        MediaPipe 결과에서 3D 랜드마크 추출
        
        Returns:
            각 랜드마크의 3D 좌표와 visibility
        """
        landmarks_3d = {}
        
        if results.pose_world_landmarks:
            for name, idx in self.SPINE_LANDMARKS.items():
                landmark = results.pose_world_landmarks.landmark[idx]
                landmarks_3d[name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
        
        return landmarks_3d
    
    def filter_by_visibility(self, 
                            landmarks: Dict[str, Dict[str, float]], 
                            min_visibility: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Visibility 기반 랜드마크 필터링
        
        Args:
            landmarks: 3D 랜드마크 딕셔너리
            min_visibility: 최소 가시성 임계값
            
        Returns:
            필터링된 랜드마크
        """
        filtered = {}
        for name, data in landmarks.items():
            if data['visibility'] >= min_visibility:
                filtered[name] = data
            else:
                print(f"랜드마크 '{name}' 제외 (visibility: {data['visibility']:.3f})")
        
        return filtered
    
    def apply_temporal_smoothing(self, 
                                 current_landmarks: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Temporal smoothing 적용 (시계열 데이터가 있을 때)
        
        Args:
            current_landmarks: 현재 프레임의 랜드마크
            
        Returns:
            스무딩된 랜드마크
        """
        # 히스토리에 추가
        self.landmark_history.append(current_landmarks)
        
        # 히스토리 크기 제한
        if len(self.landmark_history) > self.history_size:
            self.landmark_history.pop(0)
        
        # 스무딩 (평균)
        if len(self.landmark_history) < 2:
            return current_landmarks
        
        smoothed = {}
        for name in current_landmarks.keys():
            if all(name in lm for lm in self.landmark_history):
                smoothed[name] = {
                    'x': np.mean([lm[name]['x'] for lm in self.landmark_history]),
                    'y': np.mean([lm[name]['y'] for lm in self.landmark_history]),
                    'z': np.mean([lm[name]['z'] for lm in self.landmark_history]),
                    'visibility': current_landmarks[name]['visibility']
                }
            else:
                smoothed[name] = current_landmarks[name]
        
        return smoothed
    
    def multiview_landmark_fusion(self, 
                                   view_landmarks: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """
        여러 뷰의 랜드마크를 융합하여 3D 좌표 정확도 향상
        
        Args:
            view_landmarks: {view_name: {landmark_name: {x, y, z, visibility}}}
            
        Returns:
            융합된 3D 랜드마크
        """
        fused_landmarks = {}
        
        # 모든 랜드마크 이름 수집
        all_landmark_names = set()
        for view_lm in view_landmarks.values():
            all_landmark_names.update(view_lm.keys())
        
        # 각 랜드마크별로 뷰 융합
        for name in all_landmark_names:
            coords = []
            weights = []
            
            for view_name, landmarks in view_landmarks.items():
                if name in landmarks:
                    lm = landmarks[name]
                    coords.append([lm['x'], lm['y'], lm['z']])
                    weights.append(lm['visibility'])
            
            if coords:
                coords = np.array(coords)
                weights = np.array(weights)
                
                # Visibility 가중 평균
                if np.sum(weights) > 0:
                    fused_coord = np.average(coords, axis=0, weights=weights)
                else:
                    fused_coord = np.mean(coords, axis=0)
                
                fused_landmarks[name] = {
                    'x': fused_coord[0],
                    'y': fused_coord[1],
                    'z': fused_coord[2],
                    'visibility': np.mean(weights)
                }
        
        return fused_landmarks
    
    def calculate_spine_keypoints(self, landmarks: Dict[str, Dict[str, float]]) -> List[Dict]:
        """
        척추 랜드마크로부터 척추 키포인트 계산
        해부학적으로 정확한 척추 곡선 생성
        
        Returns:
            척추 키포인트 리스트 (C7, T1-T12, L1-L5, Sacrum)
        """
        spine_keypoints = []
        
        # 필수 랜드마크 확인
        if not all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            return spine_keypoints
        
        # 주요 참조점 계산
        shoulder_center = {
            'x': (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2,
            'y': (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2,
            'z': (landmarks['left_shoulder']['z'] + landmarks['right_shoulder']['z']) / 2,
        }
        
        hip_center = {
            'x': (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2,
            'y': (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2,
            'z': (landmarks['left_hip']['z'] + landmarks['right_hip']['z']) / 2,
        }
        
        # MediaPipe에서 Y는 아래로 증가하므로, shoulder < hip
        # 따라서 shoulder가 위(머리), hip이 아래(다리)
        # spine_height는 양수여야 함
        spine_height = abs(shoulder_center['y'] - hip_center['y'])
        
        # 위쪽(머리) 방향 결정: shoulder가 더 작은 y값 가짐 (MediaPipe 좌표계)
        is_shoulder_top = shoulder_center['y'] < hip_center['y']
        
        # 1. C7 (경추 7번) - 어깨 위 (목 쪽)
        c7 = {
            'name': 'C7',
            'level': 'cervical',
            'x': shoulder_center['x'],
            'y': shoulder_center['y'] - spine_height * 0.05 if is_shoulder_top else shoulder_center['y'] + spine_height * 0.05,
            'z': shoulder_center['z'] - spine_height * 0.02,  # 약간 뒤로
            'confidence': (landmarks['left_shoulder']['visibility'] + 
                         landmarks['right_shoulder']['visibility']) / 2
        }
        spine_keypoints.append(c7)
        
        # 2. T1-T12 (흉추) - 해부학적 곡선 (뒤로 볼록 - 후만)
        # C7(위)에서 엉덩이(아래)로 내려가면서 생성
        for i in range(1, 13):
            t = i / 12.0  # 0.083 ~ 1.0
            
            # Y축: C7(어깨 위)에서 엉덩이로 선형 보간
            if is_shoulder_top:
                # shoulder가 작은 값(위), hip이 큰 값(아래)
                y = c7['y'] * (1 - t) + hip_center['y'] * t
            else:
                # 반대의 경우
                y = c7['y'] * (1 - t) + hip_center['y'] * t
            
            # X축: 중심선 유지
            x = shoulder_center['x'] * (1 - t) + hip_center['x'] * t
            
            # Z축: 흉추 곡선 (후만 - 뒤로 볼록)
            # 중간(T6-T7)에서 가장 뒤로 나옴
            curve_factor = np.sin(t * np.pi) * spine_height * 0.08
            z = shoulder_center['z'] * (1 - t) + hip_center['z'] * t - curve_factor
            
            thoracic = {
                'name': f'T{i}',
                'level': 'thoracic_upper' if i <= 6 else 'thoracic_lower',
                'x': x,
                'y': y,
                'z': z,
                'confidence': 0.85
            }
            spine_keypoints.append(thoracic)
        
        # 3. L1-L5 (요추) - 해부학적 곡선 (앞으로 볼록 - 전만)
        # T12(위)에서 Sacrum(아래)로
        for i in range(1, 6):
            t = i / 5.0  # 0.2 ~ 1.0
            
            # T12의 y 위치
            t12_y = spine_keypoints[-1]['y']  # 마지막 흉추(T12)
            
            # Y축: T12에서 엉덩이로
            y = t12_y * (1 - t) + hip_center['y'] * t
            
            # X축: 중심선
            x = hip_center['x']
            
            # Z축: 요추 곡선 (전만 - 앞으로 볼록)
            # L3에서 가장 앞으로 나옴
            curve_factor = np.sin(t * np.pi) * spine_height * 0.05
            z = hip_center['z'] + curve_factor
            
            lumbar = {
                'name': f'L{i}',
                'level': 'lumbar',
                'x': x,
                'y': y,
                'z': z,
                'confidence': (landmarks['left_hip']['visibility'] + 
                             landmarks['right_hip']['visibility']) / 2
            }
            spine_keypoints.append(lumbar)
        
        # 4. Sacrum (천골) - 엉덩이 중심 (가장 아래)
        sacrum = {
            'name': 'Sacrum',
            'level': 'sacrum',
            'x': hip_center['x'],
            'y': hip_center['y'],
            'z': hip_center['z'],
            'confidence': (landmarks['left_hip']['visibility'] + 
                         landmarks['right_hip']['visibility']) / 2
        }
        spine_keypoints.append(sacrum)
        
        return spine_keypoints
    
    def refine_with_mesh(self, 
                        spine_keypoints: List[Dict],
                        mesh: trimesh.Trimesh,
                        original_mesh: trimesh.Trimesh = None,
                        search_radius: float = 0.15) -> List[Dict]:
        """
        메쉬 표면 정보를 활용하여 척추 키포인트 정제
        MediaPipe 좌표계를 메쉬 좌표계로 올바르게 변환
        
        Args:
            spine_keypoints: 초기 척추 키포인트 (MediaPipe 좌표계)
            mesh: 3D 메쉬
            original_mesh: 원본 메쉬 (스케일링 전)
            search_radius: 탐색 반경
            
        Returns:
            정제된 척추 키포인트 (메쉬 좌표계)
        """
        if original_mesh is None:
            original_mesh = mesh
        
        # 메쉬 정보
        mesh_centered = mesh.copy()
        centroid = mesh_centered.centroid
        mesh_centered.vertices -= centroid
        
        # 메쉬의 실제 스케일 계산
        mesh_bounds = mesh_centered.bounds
        mesh_scale = np.max(mesh_bounds[1] - mesh_bounds[0])
        
        print(f"\n메쉬 정보:")
        print(f"  - 중심: {centroid}")
        print(f"  - 스케일: {mesh_scale:.3f}")
        print(f"  - 범위: {mesh_bounds}")
        
        # MediaPipe 키포인트의 범위 계산
        if spine_keypoints:
            kp_coords = np.array([[kp['x'], kp['y'], kp['z']] for kp in spine_keypoints])
            kp_min = kp_coords.min(axis=0)
            kp_max = kp_coords.max(axis=0)
            kp_center = (kp_min + kp_max) / 2
            kp_scale = np.max(kp_max - kp_min)
            
            print(f"\nMediaPipe 키포인트 정보:")
            print(f"  - 중심: {kp_center}")
            print(f"  - 스케일: {kp_scale:.3f}")
            print(f"  - 범위: [{kp_min}, {kp_max}]")
        
        # KD-Tree 생성 (원본 메쉬)
        kdtree = KDTree(mesh_centered.vertices)
        
        refined_keypoints = []
        
        # 척추는 메쉬 중심선 근처에 위치
        # Y축: 상하 (높이)
        # Z축: 전후 (깊이) 
        # X축: 좌우 (너비)
        
        for idx, kp in enumerate(spine_keypoints):
            # MediaPipe 좌표를 메쉬 좌표계로 변환
            # MediaPipe는 Y가 아래로 증가, 메쉬는 Y가 위로 증가
            
            # 정규화된 좌표를 메쉬 스케일로 변환
            mesh_point = np.array([
                kp['x'] * mesh_scale,           # X: 좌우 (중심선이므로 0 근처)
                -kp['y'] * mesh_scale,          # Y: 상하 (반전)
                kp['z'] * mesh_scale            # Z: 전후
            ])
            
            # 척추는 대략 X=0 근처 (중심선)
            # Y는 어깨(높음)에서 엉덩이(낮음)로
            # Z는 몸의 뒤쪽
            
            # 메쉬에서 가장 가까운 점 탐색
            distance, idx_nearest = kdtree.query(mesh_point, k=100)
            
            # 탐색 반경 내의 정점들 (메쉬 스케일 기준)
            search_rad_scaled = search_radius * mesh_scale
            valid_indices = distance < search_rad_scaled
            
            if np.any(valid_indices):
                nearest_vertices = mesh_centered.vertices[idx_nearest[valid_indices]]
                
                # 척추는 X=0 근처여야 함 (중심선)
                # X좌표가 중심에 가까운 정점들만 선택
                x_threshold = mesh_scale * 0.1  # 메쉬 너비의 10% 이내
                centerline_mask = np.abs(nearest_vertices[:, 0]) < x_threshold
                
                if np.any(centerline_mask):
                    centerline_vertices = nearest_vertices[centerline_mask]
                    
                    # Y좌표(높이) 가중 평균
                    # 예측 높이에 가까운 점에 더 높은 가중치
                    y_diff = np.abs(centerline_vertices[:, 1] - mesh_point[1])
                    weights = np.exp(-y_diff / (mesh_scale * 0.1))
                    weights /= weights.sum()
                    
                    refined_point = np.average(centerline_vertices, axis=0, weights=weights)
                else:
                    # 중심선 정점이 없으면 중앙값 사용
                    refined_point = np.median(nearest_vertices, axis=0)
                
                refined_kp = kp.copy()
                refined_kp['x'] = refined_point[0]
                refined_kp['y'] = refined_point[1]
                refined_kp['z'] = refined_point[2]
                refined_kp['refined'] = True
                refined_kp['mesh_distance'] = np.min(distance[valid_indices])
                refined_kp['refinement_confidence'] = np.sum(valid_indices) / 100.0
                refined_kp['centerline_alignment'] = np.sum(centerline_mask) / np.sum(valid_indices) if np.any(valid_indices) else 0
                
                refined_keypoints.append(refined_kp)
            else:
                # 반경 내에 없으면 가장 가까운 점으로 부드럽게 이동
                nearest_point = mesh_centered.vertices[idx_nearest[0]]
                min_distance = distance[0]
                
                # 거리 기반 블렌딩
                if min_distance < search_rad_scaled * 3:
                    blend_factor = min(0.7, search_rad_scaled / (min_distance + 1e-6))
                    blended_point = mesh_point * (1 - blend_factor) + nearest_point * blend_factor
                    
                    refined_kp = kp.copy()
                    refined_kp['x'] = blended_point[0]
                    refined_kp['y'] = blended_point[1]
                    refined_kp['z'] = blended_point[2]
                    refined_kp['refined'] = True
                    refined_kp['mesh_distance'] = min_distance
                    refined_kp['refinement_confidence'] = 0.3
                    refined_kp['blend_factor'] = blend_factor
                    refined_keypoints.append(refined_kp)
                else:
                    # 너무 멀면 원본 유지 (메쉬 좌표계로 변환만)
                    kp_copy = kp.copy()
                    kp_copy['x'] = mesh_point[0]
                    kp_copy['y'] = mesh_point[1]
                    kp_copy['z'] = mesh_point[2]
                    kp_copy['refined'] = False
                    kp_copy['mesh_distance'] = min_distance
                    refined_keypoints.append(kp_copy)
        
        return refined_keypoints
    
    def detect_spine_from_mesh(self, 
                              mesh_path: str,
                              views: List[str] = ['front', 'side', 'back'],
                              min_visibility: float = 0.5,
                              apply_smoothing: bool = False,
                              refine_with_mesh: bool = True) -> Dict:
        """
        3D 메쉬로부터 척추 관절 라인 검출 (전체 파이프라인)
        
        Args:
            mesh_path: 3D 메쉬 파일 경로
            views: 사용할 뷰 리스트
            min_visibility: 최소 가시성 임계값
            apply_smoothing: 시간적 스무딩 적용 여부
            refine_with_mesh: 메쉬 정보로 정제 여부
            
        Returns:
            척추 검출 결과
        """
        print(f"\n=== 3D 메쉬 척추 검출 시작 ===")
        print(f"메쉬 파일: {mesh_path}")
        
        # 1. 메쉬 로드
        mesh = self.load_mesh(mesh_path)
        
        # 2. 멀티뷰 이미지 렌더링
        print(f"\n다중 뷰 렌더링 중... ({', '.join(views)})")
        view_images = self.mesh_to_multiview_images(mesh, views=views)
        
        # 3. 각 뷰별 포즈 검출
        print("\n각 뷰별 포즈 랜드마크 검출 중...")
        view_landmarks = {}
        
        for view_name, image in view_images.items():
            print(f"  - {view_name} 뷰 처리 중...")
            results = self.detect_pose_landmarks(image)
            
            if results and results.pose_landmarks:
                landmarks_3d = self.extract_3d_landmarks(results)
                filtered_landmarks = self.filter_by_visibility(landmarks_3d, min_visibility)
                
                if apply_smoothing:
                    filtered_landmarks = self.apply_temporal_smoothing(filtered_landmarks)
                
                view_landmarks[view_name] = filtered_landmarks
                print(f"    ✓ {len(filtered_landmarks)}개 랜드마크 검출")
            else:
                print(f"    ✗ 포즈 검출 실패")
        
        if not view_landmarks:
            print("경고: 모든 뷰에서 포즈 검출 실패")
            return {
                'success': False,
                'error': '포즈 검출 실패',
                'spine_keypoints': []
            }
        
        # 4. 멀티뷰 융합
        print("\n멀티뷰 랜드마크 융합 중...")
        fused_landmarks = self.multiview_landmark_fusion(view_landmarks)
        print(f"  ✓ {len(fused_landmarks)}개 융합 랜드마크 생성")
        
        # 5. 척추 키포인트 계산
        print("\n척추 키포인트 계산 중...")
        spine_keypoints = self.calculate_spine_keypoints(fused_landmarks)
        print(f"  ✓ {len(spine_keypoints)}개 척추 키포인트 생성")
        
        # 6. 메쉬 기반 정제
        if refine_with_mesh and spine_keypoints:
            print("\n메쉬 표면 정보로 정제 중...")
            spine_keypoints = self.refine_with_mesh(spine_keypoints, mesh)
            refined_count = sum(1 for kp in spine_keypoints if kp.get('refined', False))
            print(f"  ✓ {refined_count}/{len(spine_keypoints)}개 키포인트 정제됨")
        
        # 7. 결과 반환
        result = {
            'success': True,
            'mesh_path': mesh_path,
            'views_used': list(view_landmarks.keys()),
            'spine_keypoints': spine_keypoints,
            'raw_landmarks': fused_landmarks,
            'view_landmarks': view_landmarks,
            'statistics': {
                'total_keypoints': len(spine_keypoints),
                'refined_keypoints': sum(1 for kp in spine_keypoints if kp.get('refined', False)),
                'avg_confidence': np.mean([kp['confidence'] for kp in spine_keypoints]),
                'views_detected': len(view_landmarks)
            }
        }
        
        print("\n=== 척추 검출 완료 ===")
        print(f"총 {len(spine_keypoints)}개 척추 키포인트 생성")
        print(f"평균 신뢰도: {result['statistics']['avg_confidence']:.3f}")
        
        return result
    
    def visualize_results(self, 
                         result: Dict,
                         output_path: str = None):
        """
        검출 결과 시각화 - 해부학적으로 올바른 척추 곡선
        
        Args:
            result: detect_spine_from_mesh 결과
            output_path: 저장 경로 (None이면 표시만)
        """
        if not result['success']:
            print("시각화 실패: 검출 결과 없음")
            return
        
        spine_keypoints = result['spine_keypoints']
        
        # 3D 플롯
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(18, 6))
        
        # 좌표 추출
        xs = np.array([kp['x'] for kp in spine_keypoints])
        ys = np.array([kp['y'] for kp in spine_keypoints])
        zs = np.array([kp['z'] for kp in spine_keypoints])
        confidences = np.array([kp['confidence'] for kp in spine_keypoints])
        names = [kp['name'] for kp in spine_keypoints]
        
        # 척추 레벨별 색상
        level_colors = {
            'cervical': 'red',
            'thoracic_upper': 'orange',
            'thoracic_lower': 'yellow',
            'lumbar': 'green',
            'sacrum': 'blue'
        }
        colors = [level_colors.get(kp['level'], 'gray') for kp in spine_keypoints]
        
        # 1. 3D 척추 라인
        ax1 = fig.add_subplot(131, projection='3d')
        
        # 척추 곡선 그리기
        ax1.plot(xs, ys, zs, 'b-', alpha=0.6, linewidth=3, label='Spine Curve')
        
        # 키포인트 표시 (신뢰도별 크기)
        sizes = 50 + confidences * 150
        scatter = ax1.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.8, 
                            edgecolors='black', linewidth=1)
        
        # 주요 키포인트 레이블
        important = ['C7', 'T1', 'T6', 'T12', 'L1', 'L5', 'Sacrum']
        for kp in spine_keypoints:
            if kp['name'] in important:
                ax1.text(kp['x'], kp['y'], kp['z'], f" {kp['name']}", 
                        fontsize=9, weight='bold')
        
        ax1.set_xlabel('X (Left-Right)', fontsize=10)
        ax1.set_ylabel('Y (Height)', fontsize=10)
        ax1.set_zlabel('Z (Front-Back)', fontsize=10)
        ax1.set_title('3D Spine Reconstruction', fontsize=12, weight='bold')
        
        # 축 범위 설정 (비율 맞춤)
        max_range = max(xs.ptp(), ys.ptp(), zs.ptp()) / 2
        mid_x, mid_y, mid_z = xs.mean(), ys.mean(), zs.mean()
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 2. 측면 투영 (Z-Y: 척추 곡률 확인)
        ax2 = fig.add_subplot(132)
        ax2.plot(zs, ys, 'b-', alpha=0.6, linewidth=3)
        ax2.scatter(zs, ys, c=colors, s=sizes, alpha=0.8, 
                   edgecolors='black', linewidth=1)
        
        # 척추 영역 표시
        for kp in spine_keypoints:
            if kp['name'] in important:
                ax2.annotate(kp['name'], (kp['z'], kp['y']), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, weight='bold')
        
        ax2.set_xlabel('Z (Front-Back)', fontsize=10)
        ax2.set_ylabel('Y (Height)', fontsize=10)
        ax2.set_title('Side View - Spinal Curvature\n(Kyphosis & Lordosis)', 
                     fontsize=12, weight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_aspect('equal', adjustable='box')
        # Y축 반전: 작은 값이 위(머리), 큰 값이 아래(다리)
        ax2.invert_yaxis()
        
        # 3. 정면 투영 (X-Y: 척추 정렬)
        ax3 = fig.add_subplot(133)
        ax3.plot(xs, ys, 'b-', alpha=0.6, linewidth=3)
        ax3.scatter(xs, ys, c=colors, s=sizes, alpha=0.8,
                   edgecolors='black', linewidth=1)
        
        # 중심선 표시
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.3, label='Centerline')
        
        for kp in spine_keypoints:
            if kp['name'] in important:
                ax3.annotate(kp['name'], (kp['x'], kp['y']),
                           xytext=(5, 0), textcoords='offset points',
                           fontsize=8, weight='bold')
        
        ax3.set_xlabel('X (Left-Right)', fontsize=10)
        ax3.set_ylabel('Y (Height)', fontsize=10)
        ax3.set_title('Front View - Spinal Alignment\n(Scoliosis Check)', 
                     fontsize=12, weight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_aspect('equal', adjustable='box')
        # Y축 반전: 작은 값이 위(머리), 큰 값이 아래(다리)
        ax3.invert_yaxis()
        
        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Cervical (C7)'),
            Patch(facecolor='orange', label='Thoracic Upper (T1-T6)'),
            Patch(facecolor='yellow', label='Thoracic Lower (T7-T12)'),
            Patch(facecolor='green', label='Lumbar (L1-L5)'),
            Patch(facecolor='blue', label='Sacrum')
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=5, frameon=True, fontsize=9)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"시각화 저장: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, result: Dict, output_path: str):
        """결과를 JSON으로 저장"""
        # NumPy 타입을 Python 타입으로 변환
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        result_converted = convert_types(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_converted, f, indent=2, ensure_ascii=False)
        
        print(f"결과 저장 완료: {output_path}")
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='3D 메쉬 척추 관절 검출')
    parser.add_argument('--mesh', type=str, required=True,
                       help='3D 메쉬 파일 경로 (.obj, .ply, .stl)')
    parser.add_argument('--views', type=str, nargs='+', 
                       default=['front', 'side', 'back'],
                       help='사용할 뷰 (front, side, back, top)')
    parser.add_argument('--min-visibility', type=float, default=0.5,
                       help='최소 가시성 임계값 (0.0~1.0)')
    parser.add_argument('--smooth', action='store_true',
                       help='시간적 스무딩 적용')
    parser.add_argument('--no-refine', action='store_true',
                       help='메쉬 정제 비활성화')
    parser.add_argument('--output-dir', type=str, default='3d_file/spine_detection_results',
                       help='출력 디렉토리')
    parser.add_argument('--visualize', action='store_true',
                       help='결과 시각화')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 검출기 생성
    detector = MediaPipeSpineDetector(
        model_complexity=2,  # 최고 품질
        smooth_landmarks=True
    )
    
    # 척추 검출
    result = detector.detect_spine_from_mesh(
        mesh_path=args.mesh,
        views=args.views,
        min_visibility=args.min_visibility,
        apply_smoothing=args.smooth,
        refine_with_mesh=not args.no_refine
    )
    
    # 결과 저장
    mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
    json_output = os.path.join(args.output_dir, f'{mesh_name}_spine_mediapipe.json')
    detector.save_results(result, json_output)
    
    # 시각화
    if args.visualize:
        vis_output = os.path.join(args.output_dir, f'{mesh_name}_spine_mediapipe.png')
        detector.visualize_results(result, vis_output)
    
    print("\n처리 완료!")


if __name__ == '__main__':
    main()
