import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# 1. OBJ 파일 및 깊이 이미지 로드
obj_path = 'skplx_SK_test/3d_file/body_mesh_fpfh.obj'
mesh = trimesh.load(obj_path)
vertices = np.array(mesh.vertices)

# 깊이 이미지 경로
depth_images_paths = {
    'front': 'skplx_SK_test/여성/여_정면.bmp',
    'back': 'skplx_SK_test/여성/여_후면.bmp',
    'left': 'skplx_SK_test/여성/여_왼쪽.bmp',
    'right': 'skplx_SK_test/여성/여_오른쪽.bmp'
}

print("="*80)
print("깊이 이미지 기반 정밀 척추 스켈레톤 추출 시스템")
print("="*80)

print(f"\n메시 정보:")
print(f"  - 정점 수: {len(vertices)}")
print(f"  - 면 수: {len(mesh.faces)}")
print(f"  - 메시 범위: X[{vertices[:,0].min():.2f}, {vertices[:,0].max():.2f}], "
      f"Y[{vertices[:,1].min():.2f}, {vertices[:,1].max():.2f}], "
      f"Z[{vertices[:,2].min():.2f}, {vertices[:,2].max():.2f}]")

# 2. 깊이 이미지 로드 및 전처리
print("\n" + "="*80)
print("깊이 이미지 로딩 및 분석")
print("="*80)

depth_images = {}
for view, path in depth_images_paths.items():
    img = Image.open(path).convert('L')  # 그레이스케일
    depth_array = np.array(img)
    depth_images[view] = depth_array
    print(f"  {view:10s}: {depth_array.shape}, 값 범위: [{depth_array.min()}, {depth_array.max()}]")

# 3. 신체 측정치 계산
y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
height = y_max - y_min

x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
x_center = (x_min + x_max) / 2
x_width = x_max - x_min

z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
z_center = (z_min + z_max) / 2
z_depth = z_max - z_min

print(f"\n[신체 측정]")
print(f"  - 전체 높이 (Y): {height:.2f} mm")
print(f"  - 좌우 폭 (X): {x_width:.2f} mm, 중심: {x_center:.2f}")
print(f"  - 전후 깊이 (Z): {z_depth:.2f} mm, 중심: {z_center:.2f}")

# 4. 깊이 이미지로부터 척추 중심선 추출
print("\n" + "="*80)
print("깊이 이미지 기반 척추 중심선 추출")
print("="*80)

def extract_spine_from_depth_images(depth_front, depth_back, depth_left, depth_right):
    """4방향 깊이 이미지로부터 정확한 척추 위치 추출"""
    
    height_samples = 50  # 더 많은 샘플링
    spine_positions = []
    
    front_h, front_w = depth_front.shape
    back_h, back_w = depth_back.shape
    
    print(f"\n[정면/후면 이미지 분석]")
    
    # 높이별로 샘플링
    for i in range(height_samples):
        y_ratio = 0.15 + (0.75 * i / height_samples)  # 15%~90% 높이
        
        # 1. 정면 이미지에서 X 중심선 찾기 (좌우 대칭)
        front_row = int((1 - y_ratio) * front_h)  # 이미지는 위에서 아래로
        if 0 <= front_row < front_h:
            front_line = depth_front[front_row, :]
            
            # 신체 영역 찾기 (배경 제거)
            threshold = np.percentile(front_line, 20)  # 하위 20% 제외
            body_mask = front_line > threshold
            
            if body_mask.sum() > 10:
                # 신체 좌우 경계
                body_indices = np.where(body_mask)[0]
                left_edge = body_indices[0]
                right_edge = body_indices[-1]
                body_center_col = (left_edge + right_edge) // 2
                
                # 이미지 좌표 → 3D X 좌표 변환
                x_coord = x_min + (body_center_col / front_w) * x_width
            else:
                x_coord = x_center
        else:
            x_coord = x_center
        
        # 2. 후면 이미지에서 Z 위치 찾기 (척추는 등 쪽에 가까움)
        back_row = int((1 - y_ratio) * back_h)
        if 0 <= back_row < back_h:
            back_line = depth_back[back_row, :]
            
            # 등 쪽 표면 감지
            threshold = np.percentile(back_line, 20)
            body_mask = back_line > threshold
            
            if body_mask.sum() > 10:
                body_indices = np.where(body_mask)[0]
                
                # 등 중앙 부분 (중간 40%)
                mid_start = int(len(body_indices) * 0.3)
                mid_end = int(len(body_indices) * 0.7)
                mid_indices = body_indices[mid_start:mid_end]
                
                # 해당 부분의 깊이 값 (밝을수록 가까움)
                mid_depth_values = back_line[mid_indices]
                
                # 가장 튀어나온 부분 (최대 밝기)
                max_depth_col = mid_indices[np.argmax(mid_depth_values)]
                
                # 이미지 좌표 → 3D Z 좌표 변환
                # 후면 이미지에서 밝은 부분 = Z가 큰 부분 (등 쪽)
                z_surface = z_min + (max_depth_col / back_w) * z_depth
                
                # 척추는 표면에서 약간 안쪽 (체감 15-20%)
                z_coord = z_surface - z_depth * 0.18
            else:
                z_coord = z_center
        else:
            z_coord = z_center
        
        # 3. Y 좌표 계산
        y_coord = y_min + height * y_ratio
        
        spine_positions.append([x_coord, y_coord, z_coord])
    
    spine_positions = np.array(spine_positions)
    
    # 스무딩 (자연스러운 척추 곡선)
    print(f"  - 추출된 원시 포인트: {len(spine_positions)}")
    
    if len(spine_positions) > 5:
        # X, Z 좌표만 스무딩 (Y는 고정)
        spine_positions[:, 0] = gaussian_filter1d(spine_positions[:, 0], sigma=2)
        spine_positions[:, 2] = gaussian_filter1d(spine_positions[:, 2], sigma=2)
        print(f"  - 가우시안 스무딩 적용 완료")
    
    return spine_positions

spine_centerline = extract_spine_from_depth_images(
    depth_images['front'], 
    depth_images['back'], 
    depth_images['left'], 
    depth_images['right']
)

print(f"  - 최종 척추 중심선 포인트: {len(spine_centerline)}")

# 5. 척추뼈별 관절 위치 추정
print("\n" + "="*80)
print("의학적 척추 관절 추정")
print("="*80)

def get_spine_position_at_height(y_ratio, spine_centerline, y_min, height):
    """특정 높이에서 척추 위치를 보간"""
    y_target = y_min + height * y_ratio
    
    # 가장 가까운 척추 중심선 포인트 찾기
    distances = np.abs(spine_centerline[:, 1] - y_target)
    closest_idx = np.argmin(distances)
    
    # 선형 보간
    if closest_idx > 0 and closest_idx < len(spine_centerline) - 1:
        # 위아래 포인트로 보간
        lower_idx = closest_idx - 1
        upper_idx = closest_idx + 1
        
        lower_point = spine_centerline[lower_idx]
        upper_point = spine_centerline[upper_idx]
        
        # 선형 보간 비율
        if upper_point[1] != lower_point[1]:
            t = (y_target - lower_point[1]) / (upper_point[1] - lower_point[1])
            t = np.clip(t, 0, 1)
            
            position = lower_point + t * (upper_point - lower_point)
            position[1] = y_target  # Y는 정확히 맞춤
            return position
    
    # 보간 불가능하면 가장 가까운 점
    result = spine_centerline[closest_idx].copy()
    result[1] = y_target
    return result

joints = {}

print(f"\n[경추 (Cervical Vertebrae) - 7개]")
# C1-C7 높이 (상부)
cervical_heights = {
    'C1_atlas': 0.93,
    'C2_axis': 0.91,
    'C3': 0.89,
    'C4': 0.87,
    'C5': 0.85,
    'C6': 0.83,
    'C7': 0.81
}

for name, h in cervical_heights.items():
    joints[name] = get_spine_position_at_height(h, spine_centerline, y_min, height)
    print(f"  {name:15s}: {joints[name]}")

print(f"\n[흉추 (Thoracic Vertebrae) - 12개]")
thoracic_heights = np.linspace(0.79, 0.50, 12)
for i, h in enumerate(thoracic_heights, 1):
    joint_name = f'T{i}'
    joints[joint_name] = get_spine_position_at_height(h, spine_centerline, y_min, height)
    print(f"  {joint_name:4s}:              {joints[joint_name]}")

print(f"\n[요추 (Lumbar Vertebrae) - 5개]")
lumbar_heights = np.linspace(0.48, 0.36, 5)
for i, h in enumerate(lumbar_heights, 1):
    joint_name = f'L{i}'
    joints[joint_name] = get_spine_position_at_height(h, spine_centerline, y_min, height)
    print(f"  {joint_name:4s}:              {joints[joint_name]}")

print(f"\n[천골/미골 (Sacrum/Coccyx)]")
joints['S1_sacrum'] = get_spine_position_at_height(0.34, spine_centerline, y_min, height)
joints['coccyx'] = get_spine_position_at_height(0.31, spine_centerline, y_min, height)
joints['pelvis_center'] = get_spine_position_at_height(0.33, spine_centerline, y_min, height)
print(f"  S1 (Sacrum):       {joints['S1_sacrum']}")
print(f"  Coccyx:            {joints['coccyx']}")
print(f"  Pelvis Center:     {joints['pelvis_center']}")

# 6. 어깨 및 골반 (깊이 이미지 + 메시 데이터 하이브리드)
print(f"\n[어깨 (Shoulder Girdle)]")

def extract_shoulder_hybrid(y_ratio, depth_front, side='left'):
    """깊이 이미지 + 메시 데이터를 결합한 어깨 위치 추출"""
    
    # 1단계: 깊이 이미지에서 대략적 X 위치 파악
    front_h, front_w = depth_front.shape
    img_row_center = int((1 - y_ratio) * front_h)
    img_row_start = max(0, img_row_center - 3)
    img_row_end = min(front_h, img_row_center + 3)
    
    front_lines = depth_front[img_row_start:img_row_end, :]
    front_line = front_lines.mean(axis=0)
    
    threshold = np.percentile(front_line, 10)
    body_mask = front_line > threshold
    
    x_target = None
    if body_mask.sum() > 20:
        body_indices = np.where(body_mask)[0]
        body_left = body_indices[0]
        body_right = body_indices[-1]
        body_center_col = (body_left + body_right) // 2
        
        if side == 'left':
            shoulder_col = int(body_center_col + (body_right - body_center_col) * 0.85)
        else:
            shoulder_col = int(body_center_col - (body_center_col - body_left) * 0.85)
        
        x_target = x_min + (shoulder_col / front_w) * x_width
    
    # 2단계: 메시에서 해당 X, Y 범위의 점들 중 가장 외측 점 찾기
    y_coord = y_min + height * y_ratio
    y_tol = height * 0.04
    
    mask = np.abs(vertices[:, 1] - y_coord) < y_tol
    
    if side == 'left':
        mask &= vertices[:, 0] > x_center  # 왼쪽 (X가 큰 쪽)
        slice_verts = vertices[mask]
        if len(slice_verts) > 0:
            # 가장 왼쪽 점들의 평균
            sorted_indices = np.argsort(slice_verts[:, 0])
            top_5_percent = sorted_indices[-int(len(slice_verts) * 0.05):]
            shoulder_point = slice_verts[top_5_percent].mean(axis=0)
            return shoulder_point
    else:
        mask &= vertices[:, 0] < x_center  # 오른쪽 (X가 작은 쪽)
        slice_verts = vertices[mask]
        if len(slice_verts) > 0:
            sorted_indices = np.argsort(slice_verts[:, 0])
            bottom_5_percent = sorted_indices[:int(len(slice_verts) * 0.05)]
            shoulder_point = slice_verts[bottom_5_percent].mean(axis=0)
            return shoulder_point
    
    # Fallback: 깊이 이미지 결과만 사용
    if x_target is not None:
        spine_pos = get_spine_position_at_height(y_ratio, spine_centerline, y_min, height)
        return np.array([x_target, y_coord, spine_pos[2]])
    
    return None

# 어깨 높이를 C7 기준으로 조정
shoulder_height = 0.78
joints['left_acromion'] = extract_shoulder_hybrid(shoulder_height, depth_images['front'], 'left')
joints['right_acromion'] = extract_shoulder_hybrid(shoulder_height, depth_images['front'], 'right')

if joints['left_acromion'] is not None:
    print(f"  Left Acromion:     {joints['left_acromion']}")
else:
    print(f"  Left Acromion:     Not detected")

if joints['right_acromion'] is not None:
    print(f"  Right Acromion:    {joints['right_acromion']}")
else:
    print(f"  Right Acromion:    Not detected")

# 견갑골 (등 쪽에서 감지 - 메시 기반)
scapula_height = 0.72
mask = np.abs(vertices[:, 1] - (y_min + height * scapula_height)) < height * 0.03
posterior_mask = mask & (vertices[:, 2] > z_center)

left_scapula = vertices[posterior_mask & (vertices[:, 0] > x_center)]
if len(left_scapula) > 5:
    x_sorted = np.sort(left_scapula[:, 0])
    x_median_range = (x_sorted[len(x_sorted)//3], x_sorted[2*len(x_sorted)//3])
    medial_mask = (left_scapula[:, 0] >= x_median_range[0]) & (left_scapula[:, 0] <= x_median_range[1])
    if medial_mask.sum() > 0:
        joints['left_scapula_medial'] = left_scapula[medial_mask].mean(axis=0)
        print(f"  Left Scapula:      {joints['left_scapula_medial']}")

right_scapula = vertices[posterior_mask & (vertices[:, 0] < x_center)]
if len(right_scapula) > 5:
    x_sorted = np.sort(right_scapula[:, 0])
    x_median_range = (x_sorted[len(x_sorted)//3], x_sorted[2*len(x_sorted)//3])
    medial_mask = (right_scapula[:, 0] >= x_median_range[0]) & (right_scapula[:, 0] <= x_median_range[1])
    if medial_mask.sum() > 0:
        joints['right_scapula_medial'] = right_scapula[medial_mask].mean(axis=0)
        print(f"  Right Scapula:     {joints['right_scapula_medial']}")

print(f"\n[골반 (Pelvis)]")

def extract_pelvis_hybrid(y_ratio, depth_front, side='left'):
    """깊이 이미지 + 메시 데이터를 결합한 골반 위치 추출"""
    
    # 1단계: 깊이 이미지에서 대략적 위치
    front_h, front_w = depth_front.shape
    img_row_center = int((1 - y_ratio) * front_h)
    img_row_start = max(0, img_row_center - 3)
    img_row_end = min(front_h, img_row_center + 3)
    
    front_lines = depth_front[img_row_start:img_row_end, :]
    front_line = front_lines.mean(axis=0)
    
    threshold = np.percentile(front_line, 10)
    body_mask = front_line > threshold
    
    x_target = None
    if body_mask.sum() > 20:
        body_indices = np.where(body_mask)[0]
        body_left = body_indices[0]
        body_right = body_indices[-1]
        body_center_col = (body_left + body_right) // 2
        
        if side == 'left':
            pelvis_col = int(body_center_col + (body_right - body_center_col) * 0.75)
        else:
            pelvis_col = int(body_center_col - (body_center_col - body_left) * 0.75)
        
        x_target = x_min + (pelvis_col / front_w) * x_width
    
    # 2단계: 메시에서 해당 영역의 외측 점 찾기
    y_coord = y_min + height * y_ratio
    y_tol = height * 0.04
    
    mask = np.abs(vertices[:, 1] - y_coord) < y_tol
    
    if side == 'left':
        mask &= vertices[:, 0] > x_center + 5
        slice_verts = vertices[mask]
        if len(slice_verts) > 10:
            sorted_indices = np.argsort(slice_verts[:, 0])
            top_portion = sorted_indices[-10:]
            pelvis_point = slice_verts[top_portion].mean(axis=0)
            return pelvis_point
    else:
        mask &= vertices[:, 0] < x_center - 5
        slice_verts = vertices[mask]
        if len(slice_verts) > 10:
            sorted_indices = np.argsort(slice_verts[:, 0])
            bottom_portion = sorted_indices[:10]
            pelvis_point = slice_verts[bottom_portion].mean(axis=0)
            return pelvis_point
    
    # Fallback
    if x_target is not None:
        spine_pos = get_spine_position_at_height(y_ratio, spine_centerline, y_min, height)
        z_coord = spine_pos[2] + z_depth * 0.1
        return np.array([x_target, y_coord, z_coord])
    
    return None

pelvis_height = 0.35
joints['left_iliac_crest'] = extract_pelvis_hybrid(pelvis_height, depth_images['front'], 'left')
joints['right_iliac_crest'] = extract_pelvis_hybrid(pelvis_height, depth_images['front'], 'right')

if joints['left_iliac_crest'] is not None:
    print(f"  Left Iliac Crest:  {joints['left_iliac_crest']}")
else:
    print(f"  Left Iliac Crest:  Not detected")

if joints['right_iliac_crest'] is not None:
    print(f"  Right Iliac Crest: {joints['right_iliac_crest']}")
else:
    print(f"  Right Iliac Crest: Not detected")

print(f"\n[목/머리 (Neck/Head)]")
joints['occipital_base'] = get_spine_position_at_height(0.97, spine_centerline, y_min, height)
print(f"  Occipital Base:    {joints['occipital_base']}")

head_posterior = vertices[vertices[:, 1] > y_min + height * 0.95]
if len(head_posterior) > 0:
    joints['external_occipital_protuberance'] = head_posterior[np.argmax(head_posterior[:, 2])]
    print(f"  Ext. Occipital Protuberance: {joints['external_occipital_protuberance']}")

joints['vertex'] = vertices[np.argmax(vertices[:, 1])]
print(f"  Vertex (Crown):    {joints['vertex']}")

# 7. 척추 연결 구조
skeleton_connections = []
connection_names = []

cervical_sequence = ['occipital_base', 'C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7']
for i in range(len(cervical_sequence)-1):
    connection_names.append((cervical_sequence[i], cervical_sequence[i+1]))

thoracic_sequence = ['C7'] + [f'T{i}' for i in range(1, 13)]
for i in range(len(thoracic_sequence)-1):
    connection_names.append((thoracic_sequence[i], thoracic_sequence[i+1]))

lumbar_sequence = ['T12'] + [f'L{i}' for i in range(1, 6)]
for i in range(len(lumbar_sequence)-1):
    connection_names.append((lumbar_sequence[i], lumbar_sequence[i+1]))

sacral_sequence = ['L5', 'S1_sacrum', 'coccyx', 'pelvis_center']
for i in range(len(sacral_sequence)-1):
    connection_names.append((sacral_sequence[i], sacral_sequence[i+1]))

connection_names.extend([
    ('C7', 'left_acromion'),
    ('C7', 'right_acromion'),
    ('T2', 'left_scapula_medial'),
    ('T2', 'right_scapula_medial'),
    ('pelvis_center', 'left_iliac_crest'),
    ('pelvis_center', 'right_iliac_crest'),
])

joints_list = []
joints_name_to_idx = {}
idx = 0
for name, pos in joints.items():
    if pos is not None:
        joints_list.append(pos)
        joints_name_to_idx[name] = idx
        idx += 1

joints_array = np.array(joints_list)

for start_name, end_name in connection_names:
    if start_name in joints_name_to_idx and end_name in joints_name_to_idx:
        skeleton_connections.append((joints_name_to_idx[start_name], joints_name_to_idx[end_name]))

print(f"\n총 관절: {len(joints_array)}, 총 연결: {len(skeleton_connections)}")

# 8. Open3D 시각화
print("\n" + "="*80)
print("3D 시각화 생성")
print("="*80)

o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
o3d_mesh.compute_vertex_normals()
o3d_mesh.paint_uniform_color([0.9, 0.9, 0.9])

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(joints_array)
line_set.lines = o3d.utility.Vector2iVector(skeleton_connections)
line_set.colors = o3d.utility.Vector3dVector([[0.2, 0.8, 0.2] for _ in skeleton_connections])

# 척추 중심선 시각화 (디버깅용)
spine_line = o3d.geometry.LineSet()
spine_points = o3d.utility.Vector3dVector(spine_centerline)
spine_lines = [[i, i+1] for i in range(len(spine_centerline)-1)]
spine_line.points = spine_points
spine_line.lines = o3d.utility.Vector2iVector(spine_lines)
spine_line.colors = o3d.utility.Vector3dVector([[1, 0, 1] for _ in spine_lines])  # 마젠타

body_height = vertices[:, 1].max() - vertices[:, 1].min()
sphere_radius = body_height * 0.008

cervical_spheres = []
for name in ['occipital_base', 'C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7']:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.5)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([1, 0, 0])
        cervical_spheres.append(sphere)

thoracic_spheres = []
for name in [f'T{i}' for i in range(1, 13)]:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.3)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([1, 1, 0])
        thoracic_spheres.append(sphere)

lumbar_spheres = []
for name in [f'L{i}' for i in range(1, 6)]:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.4)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([0, 0.5, 1])
        lumbar_spheres.append(sphere)

sacral_spheres = []
for name in ['S1_sacrum', 'coccyx', 'pelvis_center']:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.5)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([0.8, 0, 0.8])
        sacral_spheres.append(sphere)

girdle_spheres = []
for name in ['left_acromion', 'right_acromion', 'left_scapula_medial', 'right_scapula_medial', 
             'left_iliac_crest', 'right_iliac_crest']:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.2)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([0, 0.8, 0])
        girdle_spheres.append(sphere)

coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=body_height * 0.1)

print("\n[색상 범례]")
print("  🔴 빨간색: 경추 (Cervical)")
print("  🟡 노란색: 흉추 (Thoracic)")
print("  🔵 파란색: 요추 (Lumbar)")
print("  🟣 자주색: 천골/골반 (Sacrum/Pelvis)")
print("  🟢 녹색: 어깨/골반 외측")
print("  🟣 마젠타: 깊이 이미지 기반 척추 중심선")

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="깊이 이미지 기반 정밀 척추 분석", width=1600, height=1000)

all_geometries = ([o3d_mesh, line_set, spine_line, coordinate_frame] + 
                  cervical_spheres + thoracic_spheres + lumbar_spheres + 
                  sacral_spheres + girdle_spheres)

for geom in all_geometries:
    vis.add_geometry(geom)

render_option = vis.get_render_option()
render_option.mesh_show_back_face = True
render_option.line_width = 10.0
render_option.point_size = 8.0
render_option.background_color = np.array([0.02, 0.02, 0.02])

vis.run()
vis.destroy_window()

# 9. Matplotlib 시각화
print("\n깊이 이미지 분석 결과 시각화 생성 중...")

fig = plt.figure(figsize=(24, 12))

# 9-1. 정면 깊이 이미지 + 척추 중심선
ax1 = fig.add_subplot(2, 4, 1)
ax1.imshow(depth_images['front'], cmap='gray')
# 척추 X 좌표 투영
for point in spine_centerline:
    img_x = int((point[0] - x_min) / x_width * depth_images['front'].shape[1])
    img_y = int((1 - (point[1] - y_min) / height) * depth_images['front'].shape[0])
    ax1.plot(img_x, img_y, 'ro', markersize=2)
ax1.set_title('Front Depth + Spine X-projection', fontsize=12, fontweight='bold')
ax1.axis('off')

# 9-2. 후면 깊이 이미지 + 척추 Z 투영
ax2 = fig.add_subplot(2, 4, 2)
ax2.imshow(depth_images['back'], cmap='gray')
for point in spine_centerline:
    img_x = int((point[2] - z_min) / z_depth * depth_images['back'].shape[1])
    img_y = int((1 - (point[1] - y_min) / height) * depth_images['back'].shape[0])
    ax2.plot(img_x, img_y, 'go', markersize=2)
ax2.set_title('Back Depth + Spine Z-projection', fontsize=12, fontweight='bold')
ax2.axis('off')

# 9-3. 왼쪽 깊이 이미지
ax3 = fig.add_subplot(2, 4, 3)
ax3.imshow(depth_images['left'], cmap='gray')
ax3.set_title('Left Side Depth', fontsize=12, fontweight='bold')
ax3.axis('off')

# 9-4. 오른쪽 깊이 이미지
ax4 = fig.add_subplot(2, 4, 4)
ax4.imshow(depth_images['right'], cmap='gray')
ax4.set_title('Right Side Depth', fontsize=12, fontweight='bold')
ax4.axis('off')

# 9-5. 3D 사선 뷰
ax5 = fig.add_subplot(2, 4, 5, projection='3d')
ax5.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            c='lightgray', alpha=0.08, s=0.2)

# 척추 중심선
ax5.plot(spine_centerline[:, 0], spine_centerline[:, 1], spine_centerline[:, 2],
         'm-', linewidth=4, alpha=0.8, label='Spine Centerline')

# 관절
for start_idx, end_idx in skeleton_connections:
    ax5.plot([joints_array[start_idx, 0], joints_array[end_idx, 0]],
             [joints_array[start_idx, 1], joints_array[end_idx, 1]],
             [joints_array[start_idx, 2], joints_array[end_idx, 2]], 
             'g-', linewidth=2, alpha=0.7)

for name in ['C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7']:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        ax5.scatter(joints_array[idx, 0], joints_array[idx, 1], joints_array[idx, 2], 
                   c='red', s=80, zorder=5)

for name in [f'T{i}' for i in range(1, 13)]:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        ax5.scatter(joints_array[idx, 0], joints_array[idx, 1], joints_array[idx, 2], 
                   c='yellow', s=70, zorder=5)

for name in [f'L{i}' for i in range(1, 6)]:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        ax5.scatter(joints_array[idx, 0], joints_array[idx, 1], joints_array[idx, 2], 
                   c='blue', s=80, zorder=5)

ax5.view_init(elev=5, azim=75)
ax5.set_xlabel('X (mm)')
ax5.set_ylabel('Y (mm)')
ax5.set_zlabel('Z (mm)')
ax5.set_title('3D Oblique View', fontsize=12, fontweight='bold')
ax5.set_facecolor('black')

# 9-6. 측면도 (Z-Y)
ax6 = fig.add_subplot(2, 4, 6)
ax6.scatter(vertices[:, 2], vertices[:, 1], c='lightgray', alpha=0.05, s=0.1)
ax6.plot(spine_centerline[:, 2], spine_centerline[:, 1], 'm-', linewidth=4, alpha=0.8, label='Spine')

cervical_names = ['C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7']
cervical_z = [joints_array[joints_name_to_idx[n], 2] for n in cervical_names if n in joints_name_to_idx]
cervical_y = [joints_array[joints_name_to_idx[n], 1] for n in cervical_names if n in joints_name_to_idx]
ax6.scatter(cervical_z, cervical_y, c='red', s=120, zorder=5, label='Cervical')

thoracic_z = [joints_array[joints_name_to_idx[f'T{i}'], 2] for i in range(1, 13) if f'T{i}' in joints_name_to_idx]
thoracic_y = [joints_array[joints_name_to_idx[f'T{i}'], 1] for i in range(1, 13) if f'T{i}' in joints_name_to_idx]
ax6.scatter(thoracic_z, thoracic_y, c='yellow', s=100, zorder=5, label='Thoracic')

lumbar_z = [joints_array[joints_name_to_idx[f'L{i}'], 2] for i in range(1, 6) if f'L{i}' in joints_name_to_idx]
lumbar_y = [joints_array[joints_name_to_idx[f'L{i}'], 1] for i in range(1, 6) if f'L{i}' in joints_name_to_idx]
ax6.scatter(lumbar_z, lumbar_y, c='blue', s=110, zorder=5, label='Lumbar')

ax6.set_xlabel('Z - Depth (mm)')
ax6.set_ylabel('Y - Height (mm)')
ax6.set_title('Sagittal View (Spine Curve)', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_facecolor('black')

# 9-7. 정면도 (X-Y)
ax7 = fig.add_subplot(2, 4, 7)
ax7.scatter(vertices[:, 0], vertices[:, 1], c='lightgray', alpha=0.05, s=0.1)
ax7.plot(spine_centerline[:, 0], spine_centerline[:, 1], 'm-', linewidth=4, alpha=0.8, label='Spine')

cervical_x = [joints_array[joints_name_to_idx[n], 0] for n in cervical_names if n in joints_name_to_idx]
ax7.scatter(cervical_x, cervical_y, c='red', s=120, zorder=5)

thoracic_x = [joints_array[joints_name_to_idx[f'T{i}'], 0] for i in range(1, 13) if f'T{i}' in joints_name_to_idx]
ax7.scatter(thoracic_x, thoracic_y, c='yellow', s=100, zorder=5)

lumbar_x = [joints_array[joints_name_to_idx[f'L{i}'], 0] for i in range(1, 6) if f'L{i}' in joints_name_to_idx]
ax7.scatter(lumbar_x, lumbar_y, c='blue', s=110, zorder=5)

ax7.axvline(x=x_center, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax7.set_xlabel('X - Left/Right (mm)')
ax7.set_ylabel('Y - Height (mm)')
ax7.set_title('Frontal View (Coronal)', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.set_facecolor('black')

# 9-8. 상단도 (X-Z)
ax8 = fig.add_subplot(2, 4, 8)
ax8.scatter(vertices[:, 0], vertices[:, 2], c='lightgray', alpha=0.05, s=0.1)
ax8.plot(spine_centerline[:, 0], spine_centerline[:, 2], 'm-', linewidth=4, alpha=0.8, label='Spine')

if 'left_acromion' in joints_name_to_idx and 'right_acromion' in joints_name_to_idx:
    l_idx = joints_name_to_idx['left_acromion']
    r_idx = joints_name_to_idx['right_acromion']
    ax8.plot([joints_array[l_idx, 0], joints_array[r_idx, 0]],
             [joints_array[l_idx, 2], joints_array[r_idx, 2]], 'c-', linewidth=3)
    ax8.scatter([joints_array[l_idx, 0], joints_array[r_idx, 0]],
               [joints_array[l_idx, 2], joints_array[r_idx, 2]], c='cyan', s=150)

ax8.axvline(x=x_center, color='red', linestyle='--', alpha=0.5)
ax8.axhline(y=z_center, color='blue', linestyle='--', alpha=0.5)
ax8.set_xlabel('X - Left/Right (mm)')
ax8.set_ylabel('Z - Depth (mm)')
ax8.set_title('Axial View (Transverse)', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.set_facecolor('black')

plt.tight_layout()
plt.savefig('skplx_SK_test/depth_based_spine_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
print("저장 완료: depth_based_spine_analysis.png")
plt.show()

# 10. JSON 저장
import json

medical_data = {
    'metadata': {
        'method': 'Depth image based precise spine extraction',
        'depth_images_used': list(depth_images_paths.keys()),
        'body_height_mm': float(height),
        'spine_centerline_samples': len(spine_centerline)
    },
    'joints': {}
}

for name, pos in joints.items():
    if pos is not None:
        medical_data['joints'][name] = pos.tolist()

medical_data['spine_centerline'] = spine_centerline.tolist()

output_path = 'skplx_SK_test/depth_based_spine_data.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(medical_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print(f"깊이 이미지 기반 척추 데이터 저장: {output_path}")
print("="*80)
print("\n✅ 분석 완료! 깊이 이미지를 활용한 정밀 척추 스켈레톤이 추출되었습니다.")
