import torch
import numpy as np
import trimesh
import open3d as o3d
from smplx import SMPLX, body_models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from scipy.optimize import minimize

# 1. OBJ 파일 로드
obj_path = 'skplx_SK_test/3d_file/body_mesh_fpfh.obj'
mesh = trimesh.load(obj_path)
vertices = np.array(mesh.vertices)

print(f"메시 정보:")
print(f"  - 정점 수: {len(vertices)}")
print(f"  - 면 수: {len(mesh.faces)}")
print(f"  - 메시 중심: {vertices.mean(axis=0)}")
print(f"  - 메시 범위: X[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}], Y[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}], Z[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

# 2. SMPLX 모델 불러오기
smplx_model = SMPLX(
    model_path='skplx_SK_test/smplx',  # 모델 경로 (폴더)
    gender='neutral',  # 또는 'male', 'female'
    batch_size=1,
    use_pca=False,
    flat_hand_mean=True
)

# 3. SMPLX 모델로 스켈레톤 추출
# SMPLX 모델의 기본 포즈로 출력
with torch.no_grad():
    output = smplx_model()
    
    # 관절 위치 (joints)
    joints_original = output.joints.detach().cpu().numpy()[0]  # [127, 3]
    
    # 신체 정점
    smplx_vertices = output.vertices.detach().cpu().numpy()[0]  # [10475, 3]

print(f"\nSMPLX 정보 (원본):")
print(f"  - 관절 수: {len(joints_original)}")
print(f"  - SMPLX 정점 수: {len(smplx_vertices)}")
print(f"  - SMPLX 중심: {smplx_vertices.mean(axis=0)}")
print(f"  - SMPLX 범위: X[{smplx_vertices[:,0].min():.3f}, {smplx_vertices[:,0].max():.3f}], Y[{smplx_vertices[:,1].min():.3f}, {smplx_vertices[:,1].max():.3f}], Z[{smplx_vertices[:,2].min():.3f}, {smplx_vertices[:,2].max():.3f}]")

# 4. OBJ 메시에서 스켈레톤 추출 (포즈 피팅)
print("\nOBJ 메시에서 스켈레톤 추출 중...")

# 4-1. 스케일과 평행이동 추정
smplx_center = smplx_vertices.mean(axis=0)
obj_center = vertices.mean(axis=0)

smplx_scale = np.ptp(smplx_vertices, axis=0)
obj_scale = np.ptp(vertices, axis=0)
scale_factors = obj_scale / (smplx_scale + 1e-8)
scale = np.mean(scale_factors)

print(f"  - 스케일 비율: {scale:.3f}")
print(f"  - 평행이동: {obj_center - smplx_center}")

# 4-2. 간단한 ICP 기반 정렬로 포즈 파라미터 추정
# OBJ 메시의 정점을 SMPLX 모델에 피팅
def fit_smplx_to_mesh(obj_vertices, smplx_model, initial_scale, initial_translation):
    """OBJ 메시에 SMPLX를 피팅하여 포즈 파라미터 추출"""
    
    # 초기 파라미터
    body_pose = torch.zeros(1, 63, dtype=torch.float32)  # 21 joints * 3 (axis-angle)
    global_orient = torch.zeros(1, 3, dtype=torch.float32)
    transl = torch.tensor([initial_translation], dtype=torch.float32)
    
    # 최적화할 파라미터 설정
    body_pose.requires_grad = True
    global_orient.requires_grad = True
    
    # 옵티마이저
    optimizer = torch.optim.Adam([body_pose, global_orient], lr=0.01)
    
    # 목표 정점 (샘플링)
    if len(obj_vertices) > 5000:
        sample_idx = np.random.choice(len(obj_vertices), 5000, replace=False)
        target_vertices = torch.tensor(obj_vertices[sample_idx], dtype=torch.float32)
    else:
        target_vertices = torch.tensor(obj_vertices, dtype=torch.float32)
    
    print("  - 포즈 피팅 중 (간략화 버전)...")
    
    # 간단한 최적화 (10 iterations만)
    for i in range(10):
        optimizer.zero_grad()
        
        # SMPLX 모델 출력
        output = smplx_model(body_pose=body_pose, global_orient=global_orient, return_verts=True)
        pred_vertices = output.vertices[0]
        
        # 스케일 및 평행이동 적용
        pred_vertices_scaled = pred_vertices * initial_scale + transl
        
        # 최근접 정점 손실
        tree = cKDTree(pred_vertices_scaled.detach().numpy())
        distances, _ = tree.query(target_vertices.numpy())
        loss = torch.tensor(distances.mean(), requires_grad=True)
        
        if i % 3 == 0:
            print(f"    Iteration {i}: Loss = {loss.item():.4f}")
        
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
    
    return body_pose.detach(), global_orient.detach()

# 포즈 피팅 수행 (시간이 오래 걸릴 수 있으므로 간략화)
print("  - 참고: 정확한 피팅은 시간이 오래 걸립니다. 간단한 정렬만 수행합니다.")

# 더 간단한 방법: 스케일과 평행이동만 적용
with torch.no_grad():
    output = smplx_model()
    joints_original = output.joints[0].cpu().numpy()
    smplx_vertices_original = output.vertices[0].cpu().numpy()

# 스켈레톤 정렬 적용
joints = (joints_original - smplx_center) * scale + obj_center
smplx_vertices_aligned = (smplx_vertices_original - smplx_center) * scale + obj_center

print(f"\n정렬된 SMPLX:")
print(f"  - 정렬된 중심: {smplx_vertices_aligned.mean(axis=0)}")
print(f"  - 정렬된 범위: X[{smplx_vertices_aligned[:,0].min():.3f}, {smplx_vertices_aligned[:,0].max():.3f}], Y[{smplx_vertices_aligned[:,1].min():.3f}, {smplx_vertices_aligned[:,1].max():.3f}], Z[{smplx_vertices_aligned[:,2].min():.3f}, {smplx_vertices_aligned[:,2].max():.3f}]")

# 4-3. OBJ 메시의 주요 지점을 기반으로 스켈레톤 추정
print("\nOBJ 메시에서 주요 관절 위치 추정...")

def estimate_joint_from_mesh(vertices, joint_type='head'):
    """메시에서 특정 관절 위치 추정"""
    if joint_type == 'head':
        # 가장 높은 Y 좌표
        return vertices[np.argmax(vertices[:, 1])]
    elif joint_type == 'pelvis':
        # 중간 높이, 중앙 위치
        mid_height = (vertices[:, 1].min() + vertices[:, 1].max()) * 0.35
        pelvis_candidates = vertices[np.abs(vertices[:, 1] - mid_height) < 10]
        return pelvis_candidates.mean(axis=0)
    elif joint_type == 'feet':
        # 가장 낮은 Y 좌표
        return vertices[np.argmin(vertices[:, 1])]
    return vertices.mean(axis=0)

# 주요 관절 추정
head_pos = estimate_joint_from_mesh(vertices, 'head')
pelvis_pos = estimate_joint_from_mesh(vertices, 'pelvis')
feet_pos = estimate_joint_from_mesh(vertices, 'feet')

print(f"  - 추정된 머리 위치: {head_pos}")
print(f"  - 추정된 골반 위치: {pelvis_pos}")
print(f"  - 추정된 발 위치: {feet_pos}")

# 스켈레톤 보정
body_height = head_pos[1] - feet_pos[1]
smplx_height = joints[15, 1] - joints[7, 1]  # head - ankle
height_scale = body_height / smplx_height if smplx_height > 0 else 1.0

print(f"  - 신체 높이 보정 비율: {height_scale:.3f}")

# 높이 기준으로 스켈레톤 재조정
joints_centered = joints - joints[0]  # pelvis를 원점으로
joints_rescaled = joints_centered * height_scale
joints_final = joints_rescaled + pelvis_pos  # 추정된 골반 위치로 이동

joints = joints_final  # 최종 스켈레톤

# 5. 주요 관절 인덱스 (SMPLX 기준)
joint_names = {
    0: 'pelvis',
    1: 'left_hip',
    2: 'right_hip',
    3: 'spine1',
    4: 'left_knee',
    5: 'right_knee',
    6: 'spine2',
    7: 'left_ankle',
    8: 'right_ankle',
    9: 'spine3',
    12: 'neck',
    15: 'head',
    16: 'left_collar',
    17: 'right_collar',
    18: 'left_shoulder',
    19: 'right_shoulder',
    20: 'left_elbow',
    21: 'right_elbow',
    22: 'left_wrist',
    23: 'right_wrist'
}

# 6. 스켈레톤 연결 정의 (본 구조)
skeleton_connections = [
    (0, 3),   # pelvis -> spine1
    (3, 6),   # spine1 -> spine2
    (6, 9),   # spine2 -> spine3
    (9, 12),  # spine3 -> neck
    (12, 15), # neck -> head
    (0, 1),   # pelvis -> left_hip
    (0, 2),   # pelvis -> right_hip
    (1, 4),   # left_hip -> left_knee
    (2, 5),   # right_hip -> right_knee
    (4, 7),   # left_knee -> left_ankle
    (5, 8),   # right_knee -> right_ankle
    (9, 16),  # spine3 -> left_collar
    (9, 17),  # spine3 -> right_collar
    (16, 18), # left_collar -> left_shoulder
    (17, 19), # right_collar -> right_shoulder
    (18, 20), # left_shoulder -> left_elbow
    (19, 21), # right_shoulder -> right_elbow
    (20, 22), # left_elbow -> left_wrist
    (21, 23), # right_elbow -> right_wrist
]

# 7. Open3D를 사용한 시각화
print("\n시각화 준비 중...")

# OBJ 메시만 표시 (SMPLX 메시 제거)
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
o3d_mesh.compute_vertex_normals()
o3d_mesh.paint_uniform_color([0.85, 0.85, 0.85])  # 밝은 회색

# 스켈레톤 시각화를 위한 LineSet 생성 (더 굵고 밝게)
points = joints[:24]  # 주요 관절만 사용
lines = skeleton_connections
colors = [[1, 0.2, 0.2] for _ in range(len(lines))]  # 밝은 빨간색

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

# 관절을 구(sphere)로 표시 (크기 조정)
spheres = []
sphere_radius = body_height * 0.012  # 신체 높이에 비례하여 크기 설정
for i, (idx, name) in enumerate(joint_names.items()):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.translate(joints[idx])
    sphere.paint_uniform_color([0.2, 0.5, 1])  # 밝은 파란색
    spheres.append(sphere)

# 척추 관절을 강조 (더 큰 구)
spine_joints = [0, 3, 6, 9, 12, 15]  # pelvis, spine1, spine2, spine3, neck, head
spine_spheres = []
for idx in spine_joints:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.5)  # 척추는 더 크게
    sphere.translate(joints[idx])
    sphere.paint_uniform_color([1, 1, 0])  # 노란색으로 강조
    spine_spheres.append(sphere)

print("\nOpen3D 뷰어 열기...")
print("  - 회색: 원본 OBJ 메시 (당신의 모델)")
print("  - 밝은 빨간색 선: 스켈레톤 본")
print("  - 밝은 파란색 구: 일반 관절")
print("  - 노란색 구 (큰 것): 척추 관절")

# 좌표축 추가
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=body_height * 0.15)

# 시각화 함수 정의 (반투명 메시를 위한 커스텀 렌더링)
def custom_draw_geometry_with_transparency(geometries):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="OBJ 메시의 스켈레톤 시각화", width=1400, height=900)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # 렌더 옵션 설정
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.line_width = 15.0  # 선 굵기 증가
    render_option.point_size = 10.0
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # 어두운 배경
    
    # 초기 카메라 위치 설정
    view_control = vis.get_view_control()
    view_control.set_zoom(0.7)
    
    vis.run()
    vis.destroy_window()

# 모든 요소 시각화 (OBJ 메시 + 스켈레톤만)
all_geometries = [o3d_mesh, line_set, coordinate_frame] + spheres + spine_spheres
custom_draw_geometry_with_transparency(all_geometries)

# 7. Matplotlib을 사용한 2D 시각화 (보조)
fig = plt.figure(figsize=(18, 6))

# 7-1. 정면 뷰
ax1 = fig.add_subplot(131, projection='3d')
# OBJ 메시만 표시
ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            c='lightgray', alpha=0.15, s=0.5, label='Your Model')
# 척추 강조
spine_indices = [0, 3, 6, 9, 12, 15]
for i in range(len(spine_indices)-1):
    start, end = spine_indices[i], spine_indices[i+1]
    ax1.plot([joints[start, 0], joints[end, 0]],
             [joints[start, 1], joints[end, 1]],
             [joints[start, 2], joints[end, 2]], 
             'y-', linewidth=6, label='Spine' if i == 0 else '')
# 나머지 본
for start, end in skeleton_connections:
    if (start, end) not in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]:
        ax1.plot([joints[start, 0], joints[end, 0]],
                 [joints[start, 1], joints[end, 1]],
                 [joints[start, 2], joints[end, 2]], 
                 'r-', linewidth=4, alpha=0.9)
ax1.scatter(joints[:24, 0], joints[:24, 1], joints[:24, 2], 
            c='blue', s=50, label='Joints', zorder=5)
ax1.scatter(joints[spine_indices, 0], joints[spine_indices, 1], joints[spine_indices, 2], 
            c='yellow', s=120, marker='o', edgecolors='black', linewidths=2, label='Spine Joints', zorder=6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Front View', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_facecolor('black')

# 7-2. 측면 뷰
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            c='lightgray', alpha=0.15, s=0.5)
# 척추 강조
for i in range(len(spine_indices)-1):
    start, end = spine_indices[i], spine_indices[i+1]
    ax2.plot([joints[start, 0], joints[end, 0]],
             [joints[start, 1], joints[end, 1]],
             [joints[start, 2], joints[end, 2]], 
             'y-', linewidth=6)
# 나머지 본
for start, end in skeleton_connections:
    if (start, end) not in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]:
        ax2.plot([joints[start, 0], joints[end, 0]],
                 [joints[start, 1], joints[end, 1]],
                 [joints[start, 2], joints[end, 2]], 
                 'r-', linewidth=4, alpha=0.9)
ax2.scatter(joints[:24, 0], joints[:24, 1], joints[:24, 2], 
            c='blue', s=50, zorder=5)
ax2.scatter(joints[spine_indices, 0], joints[spine_indices, 1], joints[spine_indices, 2], 
            c='yellow', s=120, marker='o', edgecolors='black', linewidths=2, zorder=6)
ax2.view_init(elev=0, azim=90)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Side View', fontsize=14, fontweight='bold')
ax2.set_facecolor('black')

# 7-3. 상단 뷰
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            c='lightgray', alpha=0.15, s=0.5)
# 척추 강조
for i in range(len(spine_indices)-1):
    start, end = spine_indices[i], spine_indices[i+1]
    ax3.plot([joints[start, 0], joints[end, 0]],
             [joints[start, 1], joints[end, 1]],
             [joints[start, 2], joints[end, 2]], 
             'y-', linewidth=6)
# 나머지 본
for start, end in skeleton_connections:
    if (start, end) not in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]:
        ax3.plot([joints[start, 0], joints[end, 0]],
                 [joints[start, 1], joints[end, 1]],
                 [joints[start, 2], joints[end, 2]], 
                 'r-', linewidth=4, alpha=0.9)
ax3.scatter(joints[:24, 0], joints[:24, 1], joints[:24, 2], 
            c='blue', s=50, zorder=5)
ax3.scatter(joints[spine_indices, 0], joints[spine_indices, 1], joints[spine_indices, 2], 
            c='yellow', s=120, marker='o', edgecolors='black', linewidths=2, zorder=6)
ax3.view_init(elev=90, azim=0)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Top View', fontsize=14, fontweight='bold')
ax3.set_facecolor('black')

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig('skplx_SK_test/obj_skeleton_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n2D 시각화 이미지 저장 완료: obj_skeleton_visualization.png")
plt.show()

# 8. 관절 정보 출력
print("\n=== 주요 관절 좌표 ===")
for idx, name in joint_names.items():
    coord = joints[idx]
    spine_marker = " *** SPINE ***" if idx in [0, 3, 6, 9, 12, 15] else ""
    print(f"{name:20s}: [{coord[0]:7.3f}, {coord[1]:7.3f}, {coord[2]:7.3f}]{spine_marker}")

print("\n=== 척추 관절 상세 정보 ===")
spine_joint_names = {
    0: 'pelvis (골반)',
    3: 'spine1 (하부 척추)',
    6: 'spine2 (중부 척추)',
    9: 'spine3 (상부 척추)',
    12: 'neck (목)',
    15: 'head (머리)'
}
for idx, name in spine_joint_names.items():
    coord = joints[idx]
    print(f"{name:30s}: [{coord[0]:7.3f}, {coord[1]:7.3f}, {coord[2]:7.3f}]")

# 9. 스켈레톤 데이터 저장
skeleton_data = {
    'joints': joints.tolist(),
    'joint_names': joint_names,
    'skeleton_connections': skeleton_connections
}

import json
output_path = 'skplx_SK_test/skeleton_data.json'
with open(output_path, 'w') as f:
    json.dump(skeleton_data, f, indent=2)
print(f"\n스켈레톤 데이터 저장 완료: {output_path}")

