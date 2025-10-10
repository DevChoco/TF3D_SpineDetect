import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 1. OBJ 파일 로드
obj_path = 'skplx_SK_test/3d_file/body_mesh_fpfh.obj'
mesh = trimesh.load(obj_path)
vertices = np.array(mesh.vertices)

print(f"메시 정보:")
print(f"  - 정점 수: {len(vertices)}")
print(f"  - 면 수: {len(mesh.faces)}")
print(f"  - 메시 중심: {vertices.mean(axis=0)}")
print(f"  - 메시 범위: X[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}], Y[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}], Z[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

# 2. 메시에서 직접 주요 관절 추정
print("\n메시 분석을 통한 관절 추정 중...")

def estimate_skeleton_from_mesh(vertices):
    """메시의 형태를 분석하여 스켈레톤 추정"""
    
    joints = {}
    
    # Y축 기준 높이 정보
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    height = y_max - y_min
    
    # X축 기준 좌우 정보
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    x_center = (x_min + x_max) / 2
    
    # Z축 기준 전후 정보
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_center = (z_min + z_max) / 2
    
    print(f"  - 신체 높이: {height:.2f}")
    print(f"  - X 중심: {x_center:.2f}, Z 중심: {z_center:.2f}")
    
    # 높이별 슬라이스로 관절 추정
    def get_slice_center(y_ratio, x_range=None, z_range=None):
        """특정 높이의 정점들의 중심 계산"""
        y_target = y_min + height * y_ratio
        tolerance = height * 0.05  # 5% 범위
        
        mask = np.abs(vertices[:, 1] - y_target) < tolerance
        
        if x_range:
            mask &= (vertices[:, 0] >= x_range[0]) & (vertices[:, 0] <= x_range[1])
        if z_range:
            mask &= (vertices[:, 2] >= z_range[0]) & (vertices[:, 2] <= z_range[1])
        
        slice_verts = vertices[mask]
        if len(slice_verts) > 0:
            return slice_verts.mean(axis=0)
        return None
    
    # 머리 (가장 높은 지점)
    head_idx = np.argmax(vertices[:, 1])
    joints['head'] = vertices[head_idx]
    
    # 목 (85% 높이)
    joints['neck'] = get_slice_center(0.85)
    
    # 어깨 (80% 높이, 좌우)
    shoulder_height = 0.80
    left_shoulder_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * shoulder_height)) < height * 0.03) &
        (vertices[:, 0] > x_center)
    ]
    right_shoulder_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * shoulder_height)) < height * 0.03) &
        (vertices[:, 0] < x_center)
    ]
    
    if len(left_shoulder_verts) > 0:
        # 가장 왼쪽 (X가 큰) 점들의 평균
        left_extreme = left_shoulder_verts[np.argsort(left_shoulder_verts[:, 0])[-int(len(left_shoulder_verts)*0.1):]]
        joints['left_shoulder'] = left_extreme.mean(axis=0)
    
    if len(right_shoulder_verts) > 0:
        # 가장 오른쪽 (X가 작은) 점들의 평균
        right_extreme = right_shoulder_verts[np.argsort(right_shoulder_verts[:, 0])[:int(len(right_shoulder_verts)*0.1)]]
        joints['right_shoulder'] = right_extreme.mean(axis=0)
    
    # 팔꿈치 (65% 높이, 좌우 끝)
    elbow_height = 0.65
    left_elbow_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * elbow_height)) < height * 0.05) &
        (vertices[:, 0] > x_center)
    ]
    right_elbow_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * elbow_height)) < height * 0.05) &
        (vertices[:, 0] < x_center)
    ]
    
    if len(left_elbow_verts) > 0:
        left_extreme = left_elbow_verts[np.argsort(left_elbow_verts[:, 0])[-int(len(left_elbow_verts)*0.05):]]
        joints['left_elbow'] = left_extreme.mean(axis=0)
    
    if len(right_elbow_verts) > 0:
        right_extreme = right_elbow_verts[np.argsort(right_elbow_verts[:, 0])[:int(len(right_elbow_verts)*0.05)]]
        joints['right_elbow'] = right_extreme.mean(axis=0)
    
    # 손목 (55% 높이, 좌우 끝)
    wrist_height = 0.55
    left_wrist_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * wrist_height)) < height * 0.05) &
        (vertices[:, 0] > x_center)
    ]
    right_wrist_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * wrist_height)) < height * 0.05) &
        (vertices[:, 0] < x_center)
    ]
    
    if len(left_wrist_verts) > 0:
        left_extreme = left_wrist_verts[np.argsort(left_wrist_verts[:, 0])[-int(len(left_wrist_verts)*0.05):]]
        joints['left_wrist'] = left_extreme.mean(axis=0)
    
    if len(right_wrist_verts) > 0:
        right_extreme = right_wrist_verts[np.argsort(right_wrist_verts[:, 0])[:int(len(right_wrist_verts)*0.05)]]
        joints['right_wrist'] = right_extreme.mean(axis=0)
    
    # 척추 (70%, 60%, 50%, 40% 높이)
    joints['spine3'] = get_slice_center(0.70)  # 상부 척추
    joints['spine2'] = get_slice_center(0.60)  # 중부 척추
    joints['spine1'] = get_slice_center(0.50)  # 하부 척추
    joints['pelvis'] = get_slice_center(0.40)   # 골반
    
    # 엉덩이 (35% 높이, 좌우)
    hip_height = 0.35
    left_hip_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * hip_height)) < height * 0.03) &
        (vertices[:, 0] > x_center + 5)
    ]
    right_hip_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * hip_height)) < height * 0.03) &
        (vertices[:, 0] < x_center - 5)
    ]
    
    if len(left_hip_verts) > 0:
        joints['left_hip'] = left_hip_verts.mean(axis=0)
    if len(right_hip_verts) > 0:
        joints['right_hip'] = right_hip_verts.mean(axis=0)
    
    # 무릎 (20% 높이, 좌우)
    knee_height = 0.20
    left_knee_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * knee_height)) < height * 0.03) &
        (vertices[:, 0] > x_center)
    ]
    right_knee_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * knee_height)) < height * 0.03) &
        (vertices[:, 0] < x_center)
    ]
    
    if len(left_knee_verts) > 0:
        joints['left_knee'] = left_knee_verts.mean(axis=0)
    if len(right_knee_verts) > 0:
        joints['right_knee'] = right_knee_verts.mean(axis=0)
    
    # 발목 (5% 높이, 좌우)
    ankle_height = 0.05
    left_ankle_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * ankle_height)) < height * 0.03) &
        (vertices[:, 0] > x_center)
    ]
    right_ankle_verts = vertices[
        (np.abs(vertices[:, 1] - (y_min + height * ankle_height)) < height * 0.03) &
        (vertices[:, 0] < x_center)
    ]
    
    if len(left_ankle_verts) > 0:
        joints['left_ankle'] = left_ankle_verts.mean(axis=0)
    if len(right_ankle_verts) > 0:
        joints['right_ankle'] = right_ankle_verts.mean(axis=0)
    
    # 발 (가장 낮은 지점)
    foot_idx = np.argmin(vertices[:, 1])
    joints['left_foot'] = vertices[foot_idx]
    joints['right_foot'] = vertices[foot_idx]
    
    return joints

# 관절 추정
joints_dict = estimate_skeleton_from_mesh(vertices)

print(f"\n추정된 관절 개수: {len(joints_dict)}")
for name, pos in joints_dict.items():
    if pos is not None:
        print(f"  {name:20s}: [{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]")

# 3. 스켈레톤 연결 정의
skeleton_connections = []
connection_names = [
    # 척추
    ('pelvis', 'spine1'),
    ('spine1', 'spine2'),
    ('spine2', 'spine3'),
    ('spine3', 'neck'),
    ('neck', 'head'),
    
    # 왼팔
    ('spine3', 'left_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    
    # 오른팔
    ('spine3', 'right_shoulder'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    
    # 왼다리
    ('pelvis', 'left_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('left_ankle', 'left_foot'),
    
    # 오른다리
    ('pelvis', 'right_hip'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('right_ankle', 'right_foot'),
]

# 연결 생성 (존재하는 관절만)
joints_list = []
joints_name_to_idx = {}
idx = 0
for name in joints_dict:
    if joints_dict[name] is not None:
        joints_list.append(joints_dict[name])
        joints_name_to_idx[name] = idx
        idx += 1

joints_array = np.array(joints_list)

# 연결 인덱스 생성
for start_name, end_name in connection_names:
    if start_name in joints_name_to_idx and end_name in joints_name_to_idx:
        skeleton_connections.append((joints_name_to_idx[start_name], joints_name_to_idx[end_name]))

print(f"\n스켈레톤 연결 개수: {len(skeleton_connections)}")

# 4. Open3D 시각화
print("\n시각화 준비 중...")

# 메시
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
o3d_mesh.compute_vertex_normals()
o3d_mesh.paint_uniform_color([0.85, 0.85, 0.85])

# 스켈레톤 선
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(joints_array)
line_set.lines = o3d.utility.Vector2iVector(skeleton_connections)
line_set.colors = o3d.utility.Vector3dVector([[1, 0.3, 0.3] for _ in skeleton_connections])

# 관절 구
body_height = vertices[:, 1].max() - vertices[:, 1].min()
sphere_radius = body_height * 0.015

spheres = []
for joint_pos in joints_array:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.translate(joint_pos)
    sphere.paint_uniform_color([0.3, 0.6, 1])
    spheres.append(sphere)

# 척추 관절 강조
spine_names = ['pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head']
spine_spheres = []
for name in spine_names:
    if name in joints_name_to_idx:
        idx = joints_name_to_idx[name]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.8)
        sphere.translate(joints_array[idx])
        sphere.paint_uniform_color([1, 1, 0])
        spine_spheres.append(sphere)

# 좌표축
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=body_height * 0.15)

print("\nOpen3D 뷰어 열기...")
print("  - 회색: OBJ 메시")
print("  - 빨간색 선: 스켈레톤 본")
print("  - 파란색 구: 일반 관절")
print("  - 노란색 구 (큰): 척추 관절")

# 시각화
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="메시 기반 직접 스켈레톤 추정", width=1400, height=900)

all_geometries = [o3d_mesh, line_set, coordinate_frame] + spheres + spine_spheres
for geom in all_geometries:
    vis.add_geometry(geom)

render_option = vis.get_render_option()
render_option.mesh_show_back_face = True
render_option.line_width = 15.0
render_option.point_size = 10.0
render_option.background_color = np.array([0.05, 0.05, 0.05])

vis.run()
vis.destroy_window()

# 5. Matplotlib 시각화
fig = plt.figure(figsize=(18, 6))

# 정면 뷰
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            c='lightgray', alpha=0.15, s=0.3, label='Mesh')

# 척추 연결
spine_connections = [('pelvis', 'spine1'), ('spine1', 'spine2'), ('spine2', 'spine3'), ('spine3', 'neck'), ('neck', 'head')]
for start_name, end_name in spine_connections:
    if start_name in joints_name_to_idx and end_name in joints_name_to_idx:
        start_idx = joints_name_to_idx[start_name]
        end_idx = joints_name_to_idx[end_name]
        ax1.plot([joints_array[start_idx, 0], joints_array[end_idx, 0]],
                 [joints_array[start_idx, 1], joints_array[end_idx, 1]],
                 [joints_array[start_idx, 2], joints_array[end_idx, 2]], 
                 'y-', linewidth=6)

# 나머지 연결
for start_idx, end_idx in skeleton_connections:
    start_name = [k for k, v in joints_name_to_idx.items() if v == start_idx][0]
    end_name = [k for k, v in joints_name_to_idx.items() if v == end_idx][0]
    if (start_name, end_name) not in spine_connections:
        ax1.plot([joints_array[start_idx, 0], joints_array[end_idx, 0]],
                 [joints_array[start_idx, 1], joints_array[end_idx, 1]],
                 [joints_array[start_idx, 2], joints_array[end_idx, 2]], 
                 'r-', linewidth=4, alpha=0.8)

ax1.scatter(joints_array[:, 0], joints_array[:, 1], joints_array[:, 2], 
            c='blue', s=60, zorder=5, label='Joints')

# 척추 관절 강조
spine_indices = [joints_name_to_idx[name] for name in spine_names if name in joints_name_to_idx]
if len(spine_indices) > 0:
    spine_joints = joints_array[spine_indices]
    ax1.scatter(spine_joints[:, 0], spine_joints[:, 1], spine_joints[:, 2], 
                c='yellow', s=150, marker='o', edgecolors='black', linewidths=2, zorder=6, label='Spine')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Front View (Direct Skeleton)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.set_facecolor('black')

# 측면 뷰
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            c='lightgray', alpha=0.15, s=0.3)

for start_name, end_name in spine_connections:
    if start_name in joints_name_to_idx and end_name in joints_name_to_idx:
        start_idx = joints_name_to_idx[start_name]
        end_idx = joints_name_to_idx[end_name]
        ax2.plot([joints_array[start_idx, 0], joints_array[end_idx, 0]],
                 [joints_array[start_idx, 1], joints_array[end_idx, 1]],
                 [joints_array[start_idx, 2], joints_array[end_idx, 2]], 
                 'y-', linewidth=6)

for start_idx, end_idx in skeleton_connections:
    start_name = [k for k, v in joints_name_to_idx.items() if v == start_idx][0]
    end_name = [k for k, v in joints_name_to_idx.items() if v == end_idx][0]
    if (start_name, end_name) not in spine_connections:
        ax2.plot([joints_array[start_idx, 0], joints_array[end_idx, 0]],
                 [joints_array[start_idx, 1], joints_array[end_idx, 1]],
                 [joints_array[start_idx, 2], joints_array[end_idx, 2]], 
                 'r-', linewidth=4, alpha=0.8)

ax2.scatter(joints_array[:, 0], joints_array[:, 1], joints_array[:, 2], 
            c='blue', s=60, zorder=5)
if len(spine_indices) > 0:
    ax2.scatter(spine_joints[:, 0], spine_joints[:, 1], spine_joints[:, 2], 
                c='yellow', s=150, marker='o', edgecolors='black', linewidths=2, zorder=6)

ax2.view_init(elev=0, azim=90)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Side View', fontsize=14, fontweight='bold')
ax2.set_facecolor('black')

# 상단 뷰
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            c='lightgray', alpha=0.15, s=0.3)

for start_name, end_name in spine_connections:
    if start_name in joints_name_to_idx and end_name in joints_name_to_idx:
        start_idx = joints_name_to_idx[start_name]
        end_idx = joints_name_to_idx[end_name]
        ax3.plot([joints_array[start_idx, 0], joints_array[end_idx, 0]],
                 [joints_array[start_idx, 1], joints_array[end_idx, 1]],
                 [joints_array[start_idx, 2], joints_array[end_idx, 2]], 
                 'y-', linewidth=6)

for start_idx, end_idx in skeleton_connections:
    start_name = [k for k, v in joints_name_to_idx.items() if v == start_idx][0]
    end_name = [k for k, v in joints_name_to_idx.items() if v == end_idx][0]
    if (start_name, end_name) not in spine_connections:
        ax3.plot([joints_array[start_idx, 0], joints_array[end_idx, 0]],
                 [joints_array[start_idx, 1], joints_array[end_idx, 1]],
                 [joints_array[start_idx, 2], joints_array[end_idx, 2]], 
                 'r-', linewidth=4, alpha=0.8)

ax3.scatter(joints_array[:, 0], joints_array[:, 1], joints_array[:, 2], 
            c='blue', s=60, zorder=5)
if len(spine_indices) > 0:
    ax3.scatter(spine_joints[:, 0], spine_joints[:, 1], spine_joints[:, 2], 
                c='yellow', s=150, marker='o', edgecolors='black', linewidths=2, zorder=6)

ax3.view_init(elev=90, azim=0)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Top View', fontsize=14, fontweight='bold')
ax3.set_facecolor('black')

plt.tight_layout()
plt.savefig('skplx_SK_test/direct_skeleton_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n2D 시각화 이미지 저장 완료: direct_skeleton_visualization.png")
plt.show()

# 6. 스켈레톤 데이터 저장
skeleton_data = {
    'joints': {name: joints_dict[name].tolist() if joints_dict[name] is not None else None 
               for name in joints_dict},
    'skeleton_connections': [(list(joints_name_to_idx.keys())[start], 
                              list(joints_name_to_idx.keys())[end]) 
                             for start, end in skeleton_connections]
}

import json
output_path = 'skplx_SK_test/direct_skeleton_data.json'
with open(output_path, 'w') as f:
    json.dump(skeleton_data, f, indent=2)
print(f"\n스켈레톤 데이터 저장 완료: {output_path}")

print("\n=== 완료 ===")
print("이 방법은 SMPLX 모델 없이 OBJ 메시의 형태를 직접 분석하여 스켈레톤을 추정합니다.")
print("실제 포즈에 더 가까운 스켈레톤을 얻을 수 있습니다.")
