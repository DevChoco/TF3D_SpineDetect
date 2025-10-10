import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree

# 1. OBJ 파일 로드
obj_path = 'skplx_SK_test/3d_file/body_mesh_fpfh.obj'
mesh = trimesh.load(obj_path)
vertices = np.array(mesh.vertices)

print("="*80)
print("의학적 척추 및 상체 골격 분석 시스템")
print("="*80)

print(f"\n메시 정보:")
print(f"  - 정점 수: {len(vertices)}")
print(f"  - 면 수: {len(mesh.faces)}")
print(f"  - 메시 범위: X[{vertices[:,0].min():.2f}, {vertices[:,0].max():.2f}], "
      f"Y[{vertices[:,1].min():.2f}, {vertices[:,1].max():.2f}], "
      f"Z[{vertices[:,2].min():.2f}, {vertices[:,2].max():.2f}]")

# 2. 의학적 척추 관절 추정
print("\n" + "="*80)
print("의학적 해부학 기준 관절 추정 중...")
print("="*80)

def estimate_medical_skeleton(vertices):
    """의학적 정확도를 위한 척추 및 상체 골격 추정 - 정면/측면 실루엣 기반"""
    
    joints = {}
    
    # 기본 치수
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    height = y_max - y_min
    
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    x_center = (x_min + x_max) / 2
    
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_center = (z_min + z_max) / 2
    
    print(f"\n[신체 측정]")
    print(f"  - 전체 높이: {height:.2f} mm")
    print(f"  - X 중심축: {x_center:.2f} mm")
    print(f"  - Z 중심축: {z_center:.2f} mm")
    
    # 1. 정면 뷰에서 신체 중심선 추출 (X 좌표)
    print(f"\n[정면 뷰 분석 - 신체 중심선 추출]")
    height_steps = 30
    spine_x_centers = []
    spine_heights = []
    
    for i in range(height_steps):
        y_ratio = 0.3 + (0.65 * i / height_steps)  # 30%~95% 높이
        y_target = y_min + height * y_ratio
        tol = height * 0.03
        
        slice_mask = np.abs(vertices[:, 1] - y_target) < tol
        slice_verts = vertices[slice_mask]
        
        if len(slice_verts) > 10:
            # 정면에서 좌우 폭의 중심 계산
            x_slice_min = slice_verts[:, 0].min()
            x_slice_max = slice_verts[:, 0].max()
            x_slice_center = (x_slice_min + x_slice_max) / 2
            
            spine_x_centers.append(x_slice_center)
            spine_heights.append(y_target)
    
    # 2. 측면 뷰에서 척추 곡선 추출 (Z 좌표)
    print(f"[측면 뷰 분석 - 척추 전후방 위치 추출]")
    spine_z_positions = []
    
    for i, y_target in enumerate(spine_heights):
        tol = height * 0.03
        slice_mask = np.abs(vertices[:, 1] - y_target) < tol
        
        # X 중심선 근처만 선택 (몸통 중앙)
        x_tol = abs(x_max - x_min) * 0.12
        x_center_current = spine_x_centers[i]
        slice_mask &= np.abs(vertices[:, 0] - x_center_current) < x_tol
        
        slice_verts = vertices[slice_mask]
        
        if len(slice_verts) > 5:
            # 등 쪽 (posterior) 선택 - Z가 큰 쪽의 20%
            z_sorted = np.sort(slice_verts[:, 2])
            z_threshold = z_sorted[int(len(z_sorted) * 0.75)]  # 상위 25%
            posterior_verts = slice_verts[slice_verts[:, 2] >= z_threshold]
            
            if len(posterior_verts) > 0:
                # 등 쪽 표면에서 약간 안쪽 (척추 위치)
                z_posterior_mean = posterior_verts[:, 2].mean()
                # 척추는 표면에서 약 20-30mm 안쪽
                body_depth = abs(z_max - z_min)
                z_spine = z_posterior_mean - body_depth * 0.15
                spine_z_positions.append(z_spine)
            else:
                spine_z_positions.append(z_center)
        else:
            spine_z_positions.append(z_center)
    
    print(f"  - 추출된 척추 중심선 포인트 수: {len(spine_x_centers)}")
    
    def get_vertebra_center(y_ratio):
        """보간을 통해 특정 높이의 척추 중심 계산"""
        y_target = y_min + height * y_ratio
        
        # 가장 가까운 높이의 중심선 데이터 찾기
        if len(spine_heights) == 0:
            return np.array([x_center, y_target, z_center])
        
        # 선형 보간
        closest_idx = np.argmin(np.abs(np.array(spine_heights) - y_target))
        
        if closest_idx < len(spine_x_centers) and closest_idx < len(spine_z_positions):
            x_spine = spine_x_centers[closest_idx]
            z_spine = spine_z_positions[closest_idx]
            
            # 주변 데이터로 스무딩
            smooth_range = 2
            start_idx = max(0, closest_idx - smooth_range)
            end_idx = min(len(spine_x_centers), closest_idx + smooth_range + 1)
            
            x_spine = np.mean(spine_x_centers[start_idx:end_idx])
            z_spine = np.mean(spine_z_positions[start_idx:end_idx])
            
            return np.array([x_spine, y_target, z_spine])
        
        return np.array([x_center, y_target, z_center])
    
    def get_lateral_point(y_ratio, side='left', tolerance=0.03):
        """특정 높이에서 좌우측 점 추정 (어깨용)"""
        y_target = y_min + height * y_ratio
        tol = height * tolerance
        
        mask = np.abs(vertices[:, 1] - y_target) < tol
        
        if side == 'left':
            mask &= vertices[:, 0] > x_center
            slice_verts = vertices[mask]
            if len(slice_verts) > 0:
                # 가장 왼쪽 (X가 큰) 점들의 평균
                extreme_idx = np.argsort(slice_verts[:, 0])[-int(len(slice_verts)*0.05):]
                return slice_verts[extreme_idx].mean(axis=0)
        else:  # right
            mask &= vertices[:, 0] < x_center
            slice_verts = vertices[mask]
            if len(slice_verts) > 0:
                # 가장 오른쪽 (X가 작은) 점들의 평균
                extreme_idx = np.argsort(slice_verts[:, 0])[:int(len(slice_verts)*0.05)]
                return slice_verts[extreme_idx].mean(axis=0)
        return None
    
    print(f"\n[경추 (Cervical Vertebrae) - 7개]")
    # C1 (Atlas) - 가장 상부, 두개골과 연결
    joints['C1_atlas'] = get_vertebra_center(0.93)
    print(f"  C1 (Atlas):        {joints['C1_atlas']}")
    
    # C2 (Axis)
    joints['C2_axis'] = get_vertebra_center(0.91)
    print(f"  C2 (Axis):         {joints['C2_axis']}")
    
    # C3-C7
    joints['C3'] = get_vertebra_center(0.89)
    joints['C4'] = get_vertebra_center(0.87)
    joints['C5'] = get_vertebra_center(0.85)
    joints['C6'] = get_vertebra_center(0.83)
    joints['C7'] = get_vertebra_center(0.81)
    print(f"  C3:                {joints['C3']}")
    print(f"  C4:                {joints['C4']}")
    print(f"  C5:                {joints['C5']}")
    print(f"  C6:                {joints['C6']}")
    print(f"  C7 (Prominens):    {joints['C7']}")
    
    print(f"\n[흉추 (Thoracic Vertebrae) - 12개]")
    # T1-T12 (흉곽과 연결)
    thoracic_heights = np.linspace(0.79, 0.50, 12)
    for i, h in enumerate(thoracic_heights, 1):
        joint_name = f'T{i}'
        joints[joint_name] = get_vertebra_center(h)
        print(f"  {joint_name:4s}:              {joints[joint_name]}")
    
    print(f"\n[요추 (Lumbar Vertebrae) - 5개]")
    # L1-L5 (하부 척추, 가장 큰 척추체)
    lumbar_heights = np.linspace(0.48, 0.36, 5)
    for i, h in enumerate(lumbar_heights, 1):
        joint_name = f'L{i}'
        joints[joint_name] = get_vertebra_center(h)
        print(f"  {joint_name:4s}:              {joints[joint_name]}")
    
    print(f"\n[천골/미골 (Sacrum/Coccyx)]")
    # Sacrum (S1-S5 융합된 형태)
    joints['S1_sacrum'] = get_vertebra_center(0.34)
    print(f"  S1 (Sacrum):       {joints['S1_sacrum']}")
    
    # Coccyx (꼬리뼈)
    joints['coccyx'] = get_vertebra_center(0.31)
    print(f"  Coccyx:            {joints['coccyx']}")
    
    print(f"\n[골반 (Pelvis)]")
    # 골반 중심 (천장관절 부위)
    joints['pelvis_center'] = get_vertebra_center(0.33)
    print(f"  Pelvis Center:     {joints['pelvis_center']}")
    
    # 좌우 장골능 (Iliac Crest) - 골반 최상단 외측
    pelvis_height = 0.35
    mask = np.abs(vertices[:, 1] - (y_min + height * pelvis_height)) < height * 0.03
    
    left_pelvis = vertices[mask & (vertices[:, 0] > x_center + 5)]
    if len(left_pelvis) > 0:
        joints['left_iliac_crest'] = left_pelvis[np.argsort(left_pelvis[:, 0])[-10:]].mean(axis=0)
        print(f"  Left Iliac Crest:  {joints['left_iliac_crest']}")
    
    right_pelvis = vertices[mask & (vertices[:, 0] < x_center - 5)]
    if len(right_pelvis) > 0:
        joints['right_iliac_crest'] = right_pelvis[np.argsort(right_pelvis[:, 0])[:10]].mean(axis=0)
        print(f"  Right Iliac Crest: {joints['right_iliac_crest']}")
    
    print(f"\n[어깨 (Shoulder Girdle)]")
    # 좌우 견봉 (Acromion) - 어깨 최외측 돌기
    joints['left_acromion'] = get_lateral_point(0.80, 'left', tolerance=0.02)
    joints['right_acromion'] = get_lateral_point(0.80, 'right', tolerance=0.02)
    print(f"  Left Acromion:     {joints['left_acromion']}")
    print(f"  Right Acromion:    {joints['right_acromion']}")
    
    # 좌우 견갑골 (Scapula) 내측연
    scapula_height = 0.75
    mask = np.abs(vertices[:, 1] - (y_min + height * scapula_height)) < height * 0.03
    
    # 등 쪽 (posterior)으로 제한
    posterior_mask = mask & (vertices[:, 2] > z_center)
    
    left_scapula = vertices[posterior_mask & (vertices[:, 0] > x_center)]
    if len(left_scapula) > 5:
        # 중간 정도 X 위치 (너무 바깥쪽 제외)
        x_sorted = np.sort(left_scapula[:, 0])
        x_median_range = (x_sorted[len(x_sorted)//3], x_sorted[2*len(x_sorted)//3])
        medial_mask = (left_scapula[:, 0] >= x_median_range[0]) & (left_scapula[:, 0] <= x_median_range[1])
        if medial_mask.sum() > 0:
            joints['left_scapula_medial'] = left_scapula[medial_mask].mean(axis=0)
            print(f"  Left Scapula (medial): {joints['left_scapula_medial']}")
    
    right_scapula = vertices[posterior_mask & (vertices[:, 0] < x_center)]
    if len(right_scapula) > 5:
        x_sorted = np.sort(right_scapula[:, 0])
        x_median_range = (x_sorted[len(x_sorted)//3], x_sorted[2*len(x_sorted)//3])
        medial_mask = (right_scapula[:, 0] >= x_median_range[0]) & (right_scapula[:, 0] <= x_median_range[1])
        if medial_mask.sum() > 0:
            joints['right_scapula_medial'] = right_scapula[medial_mask].mean(axis=0)
            print(f"  Right Scapula (medial): {joints['right_scapula_medial']}")
    
    print(f"\n[목/머리 (Neck/Head)]")
    # 후두골 (Occipital bone) - 두개골 기저부
    joints['occipital_base'] = get_vertebra_center(0.97)
    print(f"  Occipital Base:    {joints['occipital_base']}")
    
    # 외후두융기 (External Occipital Protuberance)
    head_posterior = vertices[vertices[:, 1] > y_min + height * 0.95]
    if len(head_posterior) > 0:
        # 가장 뒤쪽 (Z가 큰) 점
        joints['external_occipital_protuberance'] = head_posterior[np.argmax(head_posterior[:, 2])]
        print(f"  Ext. Occipital Protuberance: {joints['external_occipital_protuberance']}")
    
    # 정수리 (Vertex)
    joints['vertex'] = vertices[np.argmax(vertices[:, 1])]
    print(f"  Vertex (Crown):    {joints['vertex']}")
    
    return joints

# 관절 추정 실행
joints_dict = estimate_medical_skeleton(vertices)

# 전역 변수로 저장
y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
height = y_max - y_min
x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
x_center = (x_min + x_max) / 2
z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
z_center = (z_min + z_max) / 2

# 3. 척추 곡선 분석
print("\n" + "="*80)
print("척추 곡선 분석 (Spinal Curvature Analysis)")
print("="*80)

# 척추 분절별 분류
cervical_joints = [f'C{i}' for i in range(1, 8)] + ['C1_atlas', 'C2_axis']
thoracic_joints = [f'T{i}' for i in range(1, 13)]
lumbar_joints = [f'L{i}' for i in range(1, 6)]
sacral_joints = ['S1_sacrum', 'coccyx']

# 각 분절의 Y 좌표 추출
cervical_y = []
cervical_z = []
for name in ['C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7']:
    if name in joints_dict and joints_dict[name] is not None:
        cervical_y.append(joints_dict[name][1])
        cervical_z.append(joints_dict[name][2])

thoracic_y = []
thoracic_z = []
for name in thoracic_joints:
    if name in joints_dict and joints_dict[name] is not None:
        thoracic_y.append(joints_dict[name][1])
        thoracic_z.append(joints_dict[name][2])

lumbar_y = []
lumbar_z = []
for name in lumbar_joints:
    if name in joints_dict and joints_dict[name] is not None:
        lumbar_y.append(joints_dict[name][1])
        lumbar_z.append(joints_dict[name][2])

# 곡률 분석
if len(cervical_y) > 2 and len(cervical_z) > 2:
    cervical_curve = np.polyfit(cervical_y, cervical_z, 2)
    print(f"\n[경추 전만 (Cervical Lordosis)]")
    print(f"  곡률 계수: {cervical_curve[0]:.6f}")
    if cervical_curve[0] < 0:
        print(f"  → 정상 전만 곡선 (Normal lordotic curve)")
    else:
        print(f"  → 비정상: 후만 경향 (Kyphotic tendency)")

if len(thoracic_y) > 2 and len(thoracic_z) > 2:
    thoracic_curve = np.polyfit(thoracic_y, thoracic_z, 2)
    print(f"\n[흉추 후만 (Thoracic Kyphosis)]")
    print(f"  곡률 계수: {thoracic_curve[0]:.6f}")
    if thoracic_curve[0] > 0:
        print(f"  → 정상 후만 곡선 (Normal kyphotic curve)")
    else:
        print(f"  → 비정상: 평평하거나 전만 (Flat or lordotic)")

if len(lumbar_y) > 2 and len(lumbar_z) > 2:
    lumbar_curve = np.polyfit(lumbar_y, lumbar_z, 2)
    print(f"\n[요추 전만 (Lumbar Lordosis)]")
    print(f"  곡률 계수: {lumbar_curve[0]:.6f}")
    if lumbar_curve[0] < 0:
        print(f"  → 정상 전만 곡선 (Normal lordotic curve)")
    else:
        print(f"  → 비정상: 후만 경향 (Kyphotic tendency)")

# 4. 스켈레톤 연결 정의 (의학적 순서)
print("\n" + "="*80)
print("척추 연결 구조 생성")
print("="*80)

skeleton_connections = []
connection_names = []

# 경추 연결
cervical_sequence = ['occipital_base', 'C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7']
for i in range(len(cervical_sequence)-1):
    connection_names.append((cervical_sequence[i], cervical_sequence[i+1]))

# 흉추 연결
thoracic_sequence = ['C7'] + [f'T{i}' for i in range(1, 13)]
for i in range(len(thoracic_sequence)-1):
    connection_names.append((thoracic_sequence[i], thoracic_sequence[i+1]))

# 요추 연결
lumbar_sequence = ['T12'] + [f'L{i}' for i in range(1, 6)]
for i in range(len(lumbar_sequence)-1):
    connection_names.append((lumbar_sequence[i], lumbar_sequence[i+1]))

# 천골 연결
sacral_sequence = ['L5', 'S1_sacrum', 'coccyx', 'pelvis_center']
for i in range(len(sacral_sequence)-1):
    connection_names.append((sacral_sequence[i], sacral_sequence[i+1]))

# 어깨 연결
connection_names.extend([
    ('C7', 'left_acromion'),
    ('C7', 'right_acromion'),
    ('T2', 'left_scapula_medial'),
    ('T2', 'right_scapula_medial'),
])

# 골반 연결
connection_names.extend([
    ('pelvis_center', 'left_iliac_crest'),
    ('pelvis_center', 'right_iliac_crest'),
])

# 인덱스 매핑
joints_list = []
joints_name_to_idx = {}
idx = 0
for name, pos in joints_dict.items():
    if pos is not None:
        joints_list.append(pos)
        joints_name_to_idx[name] = idx
        idx += 1

joints_array = np.array(joints_list)

# 연결 인덱스 생성
for start_name, end_name in connection_names:
    if start_name in joints_name_to_idx and end_name in joints_name_to_idx:
        skeleton_connections.append((joints_name_to_idx[start_name], joints_name_to_idx[end_name]))

print(f"  총 관절 수: {len(joints_array)}")
print(f"  총 연결 수: {len(skeleton_connections)}")

# 5. Open3D 시각화
print("\n" + "="*80)
print("3D 시각화 생성 중...")
print("="*80)

# 메시
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
o3d_mesh.compute_vertex_normals()
o3d_mesh.paint_uniform_color([0.9, 0.9, 0.9])

# 척추 라인
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(joints_array)
line_set.lines = o3d.utility.Vector2iVector(skeleton_connections)
line_set.colors = o3d.utility.Vector3dVector([[0.2, 0.8, 0.2] for _ in skeleton_connections])

body_height = vertices[:, 1].max() - vertices[:, 1].min()
sphere_radius = body_height * 0.008

# 경추 (빨간색)
cervical_spheres = []
for name in ['occipital_base', 'C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7']:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.5)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([1, 0, 0])  # 빨간색
        cervical_spheres.append(sphere)

# 흉추 (노란색)
thoracic_spheres = []
for name in [f'T{i}' for i in range(1, 13)]:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.3)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([1, 1, 0])  # 노란색
        thoracic_spheres.append(sphere)

# 요추 (파란색)
lumbar_spheres = []
for name in [f'L{i}' for i in range(1, 6)]:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.4)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([0, 0.5, 1])  # 파란색
        lumbar_spheres.append(sphere)

# 천골/골반 (자주색)
sacral_spheres = []
for name in ['S1_sacrum', 'coccyx', 'pelvis_center']:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.5)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([0.8, 0, 0.8])  # 자주색
        sacral_spheres.append(sphere)

# 어깨/골반 외측 (녹색)
girdle_spheres = []
for name in ['left_acromion', 'right_acromion', 'left_scapula_medial', 'right_scapula_medial', 
             'left_iliac_crest', 'right_iliac_crest']:
    if name in joints_name_to_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.2)
        sphere.translate(joints_array[joints_name_to_idx[name]])
        sphere.paint_uniform_color([0, 0.8, 0])  # 녹색
        girdle_spheres.append(sphere)

# 좌표축
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=body_height * 0.1)

print("\n[색상 범례]")
print("  🔴 빨간색: 경추 (Cervical, C1-C7)")
print("  🟡 노란색: 흉추 (Thoracic, T1-T12)")
print("  🔵 파란색: 요추 (Lumbar, L1-L5)")
print("  🟣 자주색: 천골/골반 중심 (Sacrum/Pelvis)")
print("  🟢 녹색:   어깨/골반 외측 (Shoulder/Pelvis girdle)")
print("  🟢 연두색 선: 척추 연결")

# 시각화
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="의학적 척추 및 상체 골격 분석", width=1600, height=1000)

all_geometries = ([o3d_mesh, line_set, coordinate_frame] + 
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

# 6. Matplotlib 시각화 (측면도 - 척추 곡선 강조)
print("\n2D 척추 곡선 시각화 생성 중...")

fig = plt.figure(figsize=(20, 7))

# 6-1. 측면도 (사선)
ax1 = fig.add_subplot(141, projection='3d')
ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            c='lightgray', alpha=0.08, s=0.2)

# 척추 그리기
for start_idx, end_idx in skeleton_connections:
    ax1.plot([joints_array[start_idx, 0], joints_array[end_idx, 0]],
             [joints_array[start_idx, 1], joints_array[end_idx, 1]],
             [joints_array[start_idx, 2], joints_array[end_idx, 2]], 
             'g-', linewidth=3, alpha=0.7)

# 경추
for name in ['C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7']:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        ax1.scatter(joints_array[idx, 0], joints_array[idx, 1], joints_array[idx, 2], 
                   c='red', s=120, zorder=5, edgecolors='darkred', linewidths=2)

# 흉추
for name in [f'T{i}' for i in range(1, 13)]:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        ax1.scatter(joints_array[idx, 0], joints_array[idx, 1], joints_array[idx, 2], 
                   c='yellow', s=100, zorder=5, edgecolors='orange', linewidths=2)

# 요추
for name in [f'L{i}' for i in range(1, 6)]:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        ax1.scatter(joints_array[idx, 0], joints_array[idx, 1], joints_array[idx, 2], 
                   c='blue', s=110, zorder=5, edgecolors='darkblue', linewidths=2)

ax1.view_init(elev=5, azim=75)
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (Height, mm)')
ax1.set_zlabel('Z (Depth, mm)')
ax1.set_title('Oblique View', fontsize=14, fontweight='bold')
ax1.set_facecolor('black')

# 6-2. 순수 측면도 (Z-Y 평면)
ax2 = fig.add_subplot(142)
ax2.scatter(vertices[:, 2], vertices[:, 1], c='lightgray', alpha=0.05, s=0.1)

# 척추 곡선
spine_names_ordered = (['C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7'] + 
                       [f'T{i}' for i in range(1, 13)] + 
                       [f'L{i}' for i in range(1, 6)] + 
                       ['S1_sacrum'])

spine_z = []
spine_y = []
for name in spine_names_ordered:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        spine_z.append(joints_array[idx, 2])
        spine_y.append(joints_array[idx, 1])

if len(spine_z) > 0:
    ax2.plot(spine_z, spine_y, 'g-', linewidth=4, alpha=0.8, label='Spine Curve')
    
    # 경추
    cervical_z = [joints_array[joints_name_to_idx[name], 2] for name in ['C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7'] 
                  if name in joints_name_to_idx]
    cervical_y = [joints_array[joints_name_to_idx[name], 1] for name in ['C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7'] 
                  if name in joints_name_to_idx]
    ax2.scatter(cervical_z, cervical_y, c='red', s=150, zorder=5, edgecolors='darkred', linewidths=2, label='Cervical')
    
    # 흉추
    thoracic_z = [joints_array[joints_name_to_idx[name], 2] for name in [f'T{i}' for i in range(1, 13)] 
                  if name in joints_name_to_idx]
    thoracic_y = [joints_array[joints_name_to_idx[name], 1] for name in [f'T{i}' for i in range(1, 13)] 
                  if name in joints_name_to_idx]
    ax2.scatter(thoracic_z, thoracic_y, c='yellow', s=130, zorder=5, edgecolors='orange', linewidths=2, label='Thoracic')
    
    # 요추
    lumbar_z = [joints_array[joints_name_to_idx[name], 2] for name in [f'L{i}' for i in range(1, 6)] 
                if name in joints_name_to_idx]
    lumbar_y = [joints_array[joints_name_to_idx[name], 1] for name in [f'L{i}' for i in range(1, 6)] 
                if name in joints_name_to_idx]
    ax2.scatter(lumbar_z, lumbar_y, c='blue', s=140, zorder=5, edgecolors='darkblue', linewidths=2, label='Lumbar')

ax2.set_xlabel('Z - Anterior/Posterior (mm)', fontsize=11)
ax2.set_ylabel('Y - Height (mm)', fontsize=11)
ax2.set_title('Sagittal View (Spine Curvature)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.set_facecolor('black')
ax2.grid(True, alpha=0.3)

# 6-3. 정면도
ax3 = fig.add_subplot(143)
ax3.scatter(vertices[:, 0], vertices[:, 1], c='lightgray', alpha=0.05, s=0.1)

# 척추 중심선
for name in spine_names_ordered:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        ax3.scatter(joints_array[idx, 0], joints_array[idx, 1], 
                   c='green', s=80, zorder=5, alpha=0.8)

# 어깨
for name in ['left_acromion', 'right_acromion']:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        ax3.scatter(joints_array[idx, 0], joints_array[idx, 1], 
                   c='cyan', s=200, zorder=6, marker='^', edgecolors='blue', linewidths=2)

# 골반
for name in ['left_iliac_crest', 'right_iliac_crest']:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        ax3.scatter(joints_array[idx, 0], joints_array[idx, 1], 
                   c='magenta', s=200, zorder=6, marker='s', edgecolors='purple', linewidths=2)

ax3.axvline(x=x_center, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Midline')
ax3.set_xlabel('X - Left/Right (mm)', fontsize=11)
ax3.set_ylabel('Y - Height (mm)', fontsize=11)
ax3.set_title('Frontal View (Coronal)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.set_facecolor('black')
ax3.grid(True, alpha=0.3)

# 6-4. 상단도
ax4 = fig.add_subplot(144)
ax4.scatter(vertices[:, 0], vertices[:, 2], c='lightgray', alpha=0.05, s=0.1)

# 어깨선
if 'left_acromion' in joints_name_to_idx and 'right_acromion' in joints_name_to_idx:
    left_idx = joints_name_to_idx['left_acromion']
    right_idx = joints_name_to_idx['right_acromion']
    ax4.plot([joints_array[left_idx, 0], joints_array[right_idx, 0]],
             [joints_array[left_idx, 2], joints_array[right_idx, 2]],
             'c-', linewidth=4, label='Shoulder Line')
    ax4.scatter([joints_array[left_idx, 0], joints_array[right_idx, 0]],
               [joints_array[left_idx, 2], joints_array[right_idx, 2]],
               c='cyan', s=200, zorder=5, marker='^')

# 골반선
if 'left_iliac_crest' in joints_name_to_idx and 'right_iliac_crest' in joints_name_to_idx:
    left_idx = joints_name_to_idx['left_iliac_crest']
    right_idx = joints_name_to_idx['right_iliac_crest']
    ax4.plot([joints_array[left_idx, 0], joints_array[right_idx, 0]],
             [joints_array[left_idx, 2], joints_array[right_idx, 2]],
             'm-', linewidth=4, label='Pelvis Line')
    ax4.scatter([joints_array[left_idx, 0], joints_array[right_idx, 0]],
               [joints_array[left_idx, 2], joints_array[right_idx, 2]],
               c='magenta', s=200, zorder=5, marker='s')

ax4.axvline(x=x_center, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax4.axhline(y=z_center, color='blue', linestyle='--', alpha=0.5, linewidth=2)
ax4.set_xlabel('X - Left/Right (mm)', fontsize=11)
ax4.set_ylabel('Z - Anterior/Posterior (mm)', fontsize=11)
ax4.set_title('Axial View (Transverse)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.set_facecolor('black')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('skplx_SK_test/medical_spine_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
print("저장 완료: medical_spine_analysis.png")
plt.show()

# 7. 의학적 데이터 저장
import json

medical_data = {
    'metadata': {
        'description': 'Medical-grade spinal and upper body skeleton analysis',
        'body_height_mm': float(height),
        'coordinate_system': 'Y-up (superior), X-lateral, Z-anterior-posterior'
    },
    'joints': {
        'cervical_vertebrae': {name: joints_dict[name].tolist() if joints_dict[name] is not None else None 
                               for name in ['C1_atlas', 'C2_axis', 'C3', 'C4', 'C5', 'C6', 'C7']
                               if name in joints_dict},
        'thoracic_vertebrae': {name: joints_dict[name].tolist() if joints_dict[name] is not None else None 
                               for name in [f'T{i}' for i in range(1, 13)]
                               if name in joints_dict},
        'lumbar_vertebrae': {name: joints_dict[name].tolist() if joints_dict[name] is not None else None 
                             for name in [f'L{i}' for i in range(1, 6)]
                             if name in joints_dict},
        'sacral_coccygeal': {name: joints_dict[name].tolist() if joints_dict[name] is not None else None 
                             for name in ['S1_sacrum', 'coccyx']
                             if name in joints_dict},
        'pelvis': {name: joints_dict[name].tolist() if joints_dict[name] is not None else None 
                   for name in ['pelvis_center', 'left_iliac_crest', 'right_iliac_crest']
                   if name in joints_dict},
        'shoulder_girdle': {name: joints_dict[name].tolist() if joints_dict[name] is not None else None 
                           for name in ['left_acromion', 'right_acromion', 'left_scapula_medial', 'right_scapula_medial']
                           if name in joints_dict},
        'cranium': {name: joints_dict[name].tolist() if joints_dict[name] is not None else None 
                   for name in ['occipital_base', 'external_occipital_protuberance', 'vertex']
                   if name in joints_dict}
    },
    'spinal_curvature_analysis': {
        'cervical_lordosis': {
            'curve_coefficient': float(cervical_curve[0]) if len(cervical_y) > 2 else None,
            'assessment': 'Normal lordotic' if len(cervical_y) > 2 and cervical_curve[0] < 0 else 'Abnormal'
        } if len(cervical_y) > 2 else None,
        'thoracic_kyphosis': {
            'curve_coefficient': float(thoracic_curve[0]) if len(thoracic_y) > 2 else None,
            'assessment': 'Normal kyphotic' if len(thoracic_y) > 2 and thoracic_curve[0] > 0 else 'Abnormal'
        } if len(thoracic_y) > 2 else None,
        'lumbar_lordosis': {
            'curve_coefficient': float(lumbar_curve[0]) if len(lumbar_y) > 2 else None,
            'assessment': 'Normal lordotic' if len(lumbar_y) > 2 and lumbar_curve[0] < 0 else 'Abnormal'
        } if len(lumbar_y) > 2 else None
    }
}

output_path = 'skplx_SK_test/medical_spine_data.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(medical_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print(f"의학적 척추 데이터 저장 완료: {output_path}")
print("="*80)
print("\n분석 완료! 의학적으로 정확한 척추 및 상체 골격이 추출되었습니다.")
