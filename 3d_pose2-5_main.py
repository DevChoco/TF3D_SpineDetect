import numpy as np
import cv2
import open3d as o3d
import os
import json
from smpl_joint_extractor import SMPLJointExtractor

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

def create_skeleton_visualization(core_joints):
    """
    SMPL 핵심 관절로부터 스켈레톤 시각화 객체 생성
    
    Args:
        core_joints: SMPL 핵심 관절 딕셔너리
    
    Returns:
        tuple: (관절 포인트들, 연결선들)
    """
    geometries = []
    
    if not core_joints:
        return geometries
    
    # 관절 포인트 생성
    joint_spheres = []
    for joint_name, position in core_joints.items():
        # 관절별 색상 설정
        if 'spine' in joint_name or joint_name == 'pelvis' or joint_name == 'neck' or joint_name == 'head':
            color = [1, 0, 0]  # 척추 - 빨간색
        elif 'shoulder' in joint_name:
            color = [0, 1, 0]  # 어깨 - 초록색
        elif 'hip' in joint_name:
            color = [0, 0, 1]  # 골반 - 파란색
        else:
            color = [1, 1, 0]  # 기타 - 노란색
        
        # 관절 구체 생성
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        joint_spheres.append(sphere)
    
    # 척추 연결선 생성
    spine_joints = ['pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head']
    spine_points = []
    spine_indices = []
    
    for i, joint_name in enumerate(spine_joints):
        if joint_name in core_joints:
            spine_points.append(core_joints[joint_name])
            if i > 0:  # 이전 점과 연결
                spine_indices.append([len(spine_points)-2, len(spine_points)-1])
    
    if len(spine_points) > 1:
        # 척추 연결선
        spine_line = o3d.geometry.LineSet()
        spine_line.points = o3d.utility.Vector3dVector(spine_points)
        spine_line.lines = o3d.utility.Vector2iVector(spine_indices)
        spine_line.paint_uniform_color([1, 0, 0])  # 빨간색
        geometries.append(spine_line)
    
    # 어깨 연결선 생성
    if 'left_shoulder' in core_joints and 'right_shoulder' in core_joints:
        shoulder_points = [core_joints['left_shoulder'], core_joints['right_shoulder']]
        shoulder_line = o3d.geometry.LineSet()
        shoulder_line.points = o3d.utility.Vector3dVector(shoulder_points)
        shoulder_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        shoulder_line.paint_uniform_color([0, 1, 0])  # 초록색
        geometries.append(shoulder_line)
    
    # 골반 연결선 생성
    if 'left_hip' in core_joints and 'right_hip' in core_joints:
        hip_points = [core_joints['left_hip'], core_joints['right_hip']]
        hip_line = o3d.geometry.LineSet()
        hip_line.points = o3d.utility.Vector3dVector(hip_points)
        hip_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        hip_line.paint_uniform_color([0, 0, 1])  # 파란색
        geometries.append(hip_line)
    
    # 어깨-척추 연결
    if 'left_shoulder' in core_joints and 'spine3' in core_joints:
        connection_points = [core_joints['spine3'], core_joints['left_shoulder']]
        connection_line = o3d.geometry.LineSet()
        connection_line.points = o3d.utility.Vector3dVector(connection_points)
        connection_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        connection_line.paint_uniform_color([0.5, 0.5, 0.5])  # 회색
        geometries.append(connection_line)
    
    if 'right_shoulder' in core_joints and 'spine3' in core_joints:
        connection_points = [core_joints['spine3'], core_joints['right_shoulder']]
        connection_line = o3d.geometry.LineSet()
        connection_line.points = o3d.utility.Vector3dVector(connection_points)
        connection_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        connection_line.paint_uniform_color([0.5, 0.5, 0.5])  # 회색
        geometries.append(connection_line)
    
    # 골반-척추 연결
    if 'left_hip' in core_joints and 'pelvis' in core_joints:
        connection_points = [core_joints['pelvis'], core_joints['left_hip']]
        connection_line = o3d.geometry.LineSet()
        connection_line.points = o3d.utility.Vector3dVector(connection_points)
        connection_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        connection_line.paint_uniform_color([0.5, 0.5, 0.5])  # 회색
        geometries.append(connection_line)
    
    if 'right_hip' in core_joints and 'pelvis' in core_joints:
        connection_points = [core_joints['pelvis'], core_joints['right_hip']]
        connection_line = o3d.geometry.LineSet()
        connection_line.points = o3d.utility.Vector3dVector(connection_points)
        connection_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        connection_line.paint_uniform_color([0.5, 0.5, 0.5])  # 회색
        geometries.append(connection_line)
    
    # 관절 구체들 추가
    geometries.extend(joint_spheres)
    
    return geometries

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
    
    # 노이즈 제거 및 다운샘플링
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size=2.0)
    
    # Statistical outlier removal을 이용한 노이즈 제거
    # nb_neighbors: 통계 계산에 사용할 이웃 점들의 수
    # std_ratio: 표준편차의 배수 (이 값을 벗어나는 점들을 제거)
    cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    merged_cloud = cl
    
    # 법선 벡터 재계산
    merged_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    # SMPL 관절 추출 및 스켈레톤 생성
    print("SMPL 관절 추출 및 스켈레톤 생성 중...")
    try:
        # SMPL 관절 추출기 초기화
        extractor = SMPLJointExtractor(model_type='smpl', device='cpu')
        
        # 깊이 맵에서 관절 추출
        depth_maps = {}
        for view_name, file_path in views.items():
            depth_map = load_depth_map(file_path)
            if depth_map is not None:
                depth_maps[view_name] = depth_map
        
        # 관절 추출
        joint_results = extractor.extract_joints_from_depth(depth_maps)
        core_joints = joint_results.get('core_joints', {})
        
        # 포인트 클라우드 좌표계에 맞게 스켈레톤 스케일 조정
        if core_joints:
            # 포인트 클라우드의 바운딩 박스 계산
            points = np.asarray(merged_cloud.points)
            if len(points) > 0:
                min_coords = np.min(points, axis=0)
                max_coords = np.max(points, axis=0)
                cloud_center = (min_coords + max_coords) / 2
                cloud_height = max_coords[1] - min_coords[1]
                
                # 스켈레톤을 포인트 클라우드 크기에 맞게 스케일링
                skeleton_scale = cloud_height / 0.75  # 대략적인 사람 키 (0.75m)
                
                # 스켈레톤 위치 조정
                adjusted_joints = {}
                for joint_name, joint_pos in core_joints.items():
                    # 스케일링 및 위치 조정
                    scaled_pos = np.array(joint_pos) * skeleton_scale
                    # Y축 오프셋 조정 (골반을 포인트 클라우드 하단에 맞춤)
                    scaled_pos[1] += min_coords[1] - (core_joints['pelvis'][1] * skeleton_scale)
                    # X, Z축을 포인트 클라우드 중심에 맞춤
                    scaled_pos[0] += cloud_center[0]
                    scaled_pos[2] += cloud_center[2]
                    adjusted_joints[joint_name] = scaled_pos
                
                # 스켈레톤 시각화 객체 생성
                skeleton_geometries = create_skeleton_visualization(adjusted_joints)
                
                print(f"스켈레톤 생성 완료: {len(skeleton_geometries)}개의 요소")
                
                # 자세 분석 결과 출력
                posture_metrics = joint_results.get('posture_metrics', {})
                if posture_metrics:
                    print(f"자세 점수: {posture_metrics.get('posture_score', 0):.1f}/100")
                
            else:
                skeleton_geometries = []
                print("포인트 클라우드가 비어있어 스켈레톤을 생성할 수 없습니다.")
        else:
            skeleton_geometries = []
            print("관절 데이터가 없어 스켈레톤을 생성할 수 없습니다.")
            
    except Exception as e:
        print(f"스켈레톤 생성 중 오류: {e}")
        skeleton_geometries = []
    
    # 메시 생성
    print("포인트 클라우드를 메시로 변환 중...")
    mesh = create_mesh_from_pointcloud(merged_cloud)
    
    # 메시 저장
    if mesh is not None:
        output_dir = "output/3d_models"
        os.makedirs(output_dir, exist_ok=True)
        
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
    vis.create_window(window_name="3D Pose Visualization with Skeleton", width=1200, height=800)
    
    # 포인트 클라우드 추가
    vis.add_geometry(merged_cloud)
    
    # 메시 추가 (있는 경우)
    if mesh is not None:
        # 메시를 반투명하게 설정
        mesh.paint_uniform_color([0.7, 0.7, 0.9])
        vis.add_geometry(mesh)
    
    # 스켈레톤 추가
    for skeleton_geom in skeleton_geometries:
        vis.add_geometry(skeleton_geom)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 어두운 회색 배경
    opt.line_width = 3.0  # 스켈레톤 선 두께
    
    # 조명 설정
    opt.light_on = True
    
    # 카메라 위치 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    ctr.set_front([0.4, -0.6, -0.6])
    ctr.set_up([0, -1, 0])
    
    print("\n=== 시각화 조작법 ===")
    print("마우스 왼쪽 버튼: 회전")
    print("마우스 오른쪽 버튼: 줌")
    print("마우스 가운데 버튼: 이동")
    print("ESC 또는 창 닫기: 종료")
    print("========================")
    
    # 시각화
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_3d_pose()