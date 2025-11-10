import numpy as np
import cv2
import open3d as o3d
import os
import copy

def create_mask_from_depth(depth_map, threshold_low=0.1, threshold_high=0.9):
    """
    깊이맵에서 이진 마스크를 생성합니다.
    
    Args:
        depth_map: 정규화된 깊이맵 (0-1 범위)
        threshold_low: 하한 임계값 (이 값 이하는 배경으로 간주)
        threshold_high: 상한 임계값 (이 값 이상은 노이즈로 간주)
    
    Returns:
        이진 마스크 (True: 유효한 영역, False: 배경/노이즈)
    """
    # 기본 임계값 적용
    mask = (depth_map > threshold_low) & (depth_map < threshold_high)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Opening (erosion followed by dilation) - 작은 노이즈 제거
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Closing (dilation followed by erosion) - 작은 구멍 채우기
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # 가장 큰 연결 구성 요소만 유지 (주요 객체만 보존)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    if num_labels > 1:  # 배경(0) 제외하고 다른 구성 요소가 있는 경우
        # 가장 큰 구성 요소 찾기 (배경 제외)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_uint8 = (labels == largest_component).astype(np.uint8) * 255
    
    return mask_uint8 > 0

def remove_noise_from_pointcloud(pcd, method="statistical", verbose=True):
    """
    포인트 클라우드에서 이상치만 제거합니다.
    
    Args:
        pcd: Open3D PointCloud 객체
        method: 노이즈 제거 방법 ("statistical", "radius", "all")
        verbose: 로그 출력 여부
    
    Returns:
        이상치가 제거된 PointCloud 객체
    """
    if pcd is None or len(pcd.points) == 0:
        return pcd
    
    original_count = len(pcd.points)
    cleaned_pcd = pcd
    
    if method in ["statistical", "all"]:
        # Statistical Outlier Removal - 이상치만 제거 (매우 관대하게)
        # 각 포인트의 이웃들과의 거리 분포를 분석하여 극단적인 이상치만 제거
        cl, ind = cleaned_pcd.remove_statistical_outlier(
            nb_neighbors=10,  # 분석할 이웃 포인트 수 (더 적게)
            std_ratio=3.0     # 표준편차 배수 (매우 관대하게, 극단적 이상치만 제거)
        )
        cleaned_pcd = cl
        if verbose:
            stat_removed = original_count - len(cleaned_pcd.points)
            print(f"  Statistical outlier removal: {stat_removed}개 극단적 이상치 제거")
    
    # Radius Outlier Removal과 Voxel downsampling은 제거하여 이상치만 처리
    
    return cleaned_pcd

def preprocess_for_icp(pcd, aggressive=False):
    """
    ICP 정렬을 위한 포인트 클라우드 전처리 - 이상치만 제거
    
    Args:
        pcd: Open3D PointCloud 객체
        aggressive: True면 더 강력한 이상치 제거 적용
    
    Returns:
        전처리된 PointCloud 객체
    """
    if pcd is None:
        return None
    
    print(f"  ICP 전처리 시작: {len(pcd.points)}개 포인트")
    
    # 이상치만 제거 (다운샘플링 최소화)
    if aggressive:
        # 더 엄격한 이상치 제거
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=1.5)
        pcd = cl
    else:
        # 기본 이상치 제거
        pcd = remove_noise_from_pointcloud(pcd, method="statistical", verbose=False)
    
    # 법선 벡터 재계산 (ICP 품질 향상을 위해)
    if len(pcd.points) > 50:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    print(f"  ICP 전처리 완료: {len(pcd.points)}개 포인트")
    return pcd

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
    
    # 마스크 생성
    mask = create_mask_from_depth(depth_map, threshold_low=0.2, threshold_high=0.95)
    
    size = depth_map.shape[0]  # 정사각형이므로 한 변의 길이만 필요
    y, x = np.mgrid[0:size, 0:size]
    
    # 포인트 수를 줄이기 위해 다운샘플링
    step = 1
    x = x[::step, ::step]
    y = y[::step, ::step]
    depth_map = depth_map[::step, ::step]
    mask = mask[::step, ::step]
    
    # 중심점 조정을 위한 오프셋 계산
    x = x - size/2
    y = y - size/2
    
    scale = 100  # 스케일 조정
    
    # 뷰에 따라 좌표 변환
    if view == "front":
        points = np.stack([x, -y, depth_map * scale], axis=-1)
    elif view == "right":
        points = np.stack([depth_map * scale * 2.5, -y, -x], axis=-1)  # 우측 깊이 2배
    elif view == "left":
        points = np.stack([-depth_map * scale * 2.5, -y, x], axis=-1)  # 좌측 깊이 2배
    elif view == "back":
        points = np.stack([-x, -y, -depth_map * scale], axis=-1)

    # 마스크를 적용하여 유효한 포인트만 선택
    valid_points = points[mask]
    
    # 추가적으로 깊이 임계값도 적용 (마스크와 함께 사용)
    valid_depths = depth_map[mask]
    depth_threshold = 0.01
    final_valid_mask = valid_depths > depth_threshold
    valid_points = valid_points[final_valid_mask]
    
    print(f"{view} 뷰: 마스크 적용 전 {np.sum(mask)} 포인트, 적용 후 {len(valid_points)} 포인트")
    
    # 너무 많은 포인트가 있는 경우 추가 다운샘플링
    if len(valid_points) > 20000:
        indices = np.random.choice(len(valid_points), 20000, replace=False)
        valid_points = valid_points[indices]
    
    # 포인트가 너무 적으면 None 반환
    if len(valid_points) < 100:
        print(f"경고: {view} 뷰의 유효한 포인트가 너무 적습니다 ({len(valid_points)}개)")
        return None
    
    # Open3D 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    colors = {
        "front": [1, 0, 0],  # 빨간색
        "right": [0, 1, 0],  # 초록색
        "left": [0, 0, 1],   # 파란색
        "back": [1, 1, 0]    # 노란색
    }
    
    pcd.paint_uniform_color(colors[view])
    
    # 이상치 제거만 적용 (매우 보수적)
    print(f"  {view} 뷰 이상치 제거 시작...")
    # 극단적인 이상치만 제거
    if len(valid_points) > 500:  # 포인트가 충분할 때만 이상치 제거
        pcd = remove_noise_from_pointcloud(pcd, method="statistical", verbose=True)
    else:
        print(f"  포인트가 적어 이상치 제거 생략: {len(valid_points)}개")
    
    return pcd

def align_point_clouds(source, target, threshold=10, use_preprocessing=True):
    """
    개선된 ICP 정렬 함수
    
    Args:
        source: 정렬할 소스 포인트 클라우드
        target: 타겟 포인트 클라우드  
        threshold: ICP 대응점 최대 거리
        use_preprocessing: ICP 전 전처리 적용 여부
    
    Returns:
        정렬된 포인트 클라우드
    """
    if source is None or target is None:
        return source
    
    print(f"  ICP 정렬 시작: source {len(source.points)}, target {len(target.points)} 포인트")
    
    # ICP를 위한 전처리
    if use_preprocessing:
        # Open3D PointCloud 복사
        source_processed = o3d.geometry.PointCloud()
        source_processed.points = o3d.utility.Vector3dVector(np.asarray(source.points))
        source_processed.colors = o3d.utility.Vector3dVector(np.asarray(source.colors))
        if source.has_normals():
            source_processed.normals = o3d.utility.Vector3dVector(np.asarray(source.normals))
        
        target_processed = o3d.geometry.PointCloud()
        target_processed.points = o3d.utility.Vector3dVector(np.asarray(target.points))
        target_processed.colors = o3d.utility.Vector3dVector(np.asarray(target.colors))
        if target.has_normals():
            target_processed.normals = o3d.utility.Vector3dVector(np.asarray(target.normals))
        
        source_processed = preprocess_for_icp(source_processed, aggressive=False)
        target_processed = preprocess_for_icp(target_processed, aggressive=False)
    else:
        source_processed = source
        target_processed = target
    
    if source_processed is None or target_processed is None:
        print("  전처리 실패, 원본 반환")
        return source
    
    # 초기 변환 행렬
    init_transformation = np.eye(4)
    
    # 1단계: 거친 정렬 (Point-to-Point ICP)
    print("  1단계: Point-to-Point ICP 정렬...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_processed, target_processed,
        max_correspondence_distance=threshold * 2,  # 초기에는 더 큰 거리 허용
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50
        )
    )
    
    print(f"  1단계 결과: fitness={reg_p2p.fitness:.4f}, rmse={reg_p2p.inlier_rmse:.4f}")
    
    # 2단계: 정교한 정렬 (Point-to-Plane ICP, 법선이 있는 경우)
    if (source_processed.has_normals() and target_processed.has_normals() and 
        reg_p2p.fitness > 0.05):  # 1단계가 어느 정도 성공한 경우만
        
        print("  2단계: Point-to-Plane ICP 정렬...")
        reg_p2plane = o3d.pipelines.registration.registration_icp(
            source_processed, target_processed,
            max_correspondence_distance=threshold,
            init=reg_p2p.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=100
            )
        )
        
        print(f"  2단계 결과: fitness={reg_p2plane.fitness:.4f}, rmse={reg_p2plane.inlier_rmse:.4f}")
        
        # 더 나은 결과 선택
        if reg_p2plane.fitness > reg_p2p.fitness:
            final_transformation = reg_p2plane.transformation
            final_fitness = reg_p2plane.fitness
        else:
            final_transformation = reg_p2p.transformation
            final_fitness = reg_p2p.fitness
    else:
        final_transformation = reg_p2p.transformation
        final_fitness = reg_p2p.fitness
    
    # 결과가 유효한 경우에만 변환 적용
    if final_fitness > 0.02:  # 2% 이상의 품질
        print(f"  정렬 성공: fitness={final_fitness:.4f}")
        return source.transform(final_transformation)
    else:
        print(f"  정렬 실패: fitness={final_fitness:.4f}, 원본 반환")
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
    # views = {
    #     "front": r"d:\Lab2\3D_Body_Posture_Analysis\test\정상\정면_여\DepthMap0.bmp",
    #     "right": r"d:\Lab2\3D_Body_Posture_Analysis\test\정상\오른쪽_여\DepthMap0.bmp",
    #     "left": r"d:\Lab2\3D_Body_Posture_Analysis\test\정상\왼쪽_여\DepthMap0.bmp",
    #     "back": r"d:\Lab2\3D_Body_Posture_Analysis\test\정상\후면_여\DepthMap0.bmp"
    # }
    
    views = {
        "front": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_정면.bmp",
        "right": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_오른쪽.bmp",
        "left": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_왼쪽.bmp",
        "back": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_후면.bmp"
    }

    
    # 각 뷰의 포인트 클라우드 생성
    point_clouds = {}
    for view_name, file_path in views.items():
        print(f"\n{view_name} 뷰 처리 중...")
        depth_map = load_depth_map(file_path)
        if depth_map is not None:
            # 디버깅을 위해 마스크도 저장
            mask = create_mask_from_depth(depth_map, threshold_low=0.2, threshold_high=0.95)
            
            # 마스크 시각화를 위해 저장 (선택사항)
            debug_dir = "output/debug"
            os.makedirs(debug_dir, exist_ok=True)
            mask_path = os.path.join(debug_dir, f"{view_name}_mask.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
            print(f"마스크 저장됨: {mask_path}")
            
            pcd = create_point_cloud_from_depth(depth_map, view_name)
            if pcd is not None:
                # 법선 벡터 계산
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
                point_clouds[view_name] = pcd
    
    # 정면을 기준으로 정렬 시작
    aligned_clouds = [point_clouds["front"]]
    front_target = point_clouds["front"]
    
    print("\n=== ICP 정렬 단계 ===")
    
    # 좌측과 우측을 정면과 정렬
    left_aligned = None
    right_aligned = None
    
    if "left" in point_clouds:
        print("\n좌측 뷰를 정면과 정렬...")
        left_aligned = align_point_clouds(point_clouds["left"], front_target, threshold=50, use_preprocessing=True)
        aligned_clouds.append(left_aligned)
    
    if "right" in point_clouds:
        print("\n우측 뷰를 정면과 정렬...")
        right_aligned = align_point_clouds(point_clouds["right"], front_target, threshold=50, use_preprocessing=True)
        aligned_clouds.append(right_aligned)
    
    # 후면은 정렬된 좌우 포인트들과 함께 정렬
    if "back" in point_clouds and (left_aligned is not None or right_aligned is not None):
        print("\n후면 뷰를 좌우측과 정렬...")
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
        back_aligned = align_point_clouds(point_clouds["back"], side_target, threshold=80, use_preprocessing=True)
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
    
    print(f"\n=== 최종 병합 및 이상치 제거 ===")
    print(f"병합된 포인트 수: {len(merged_cloud.points)}")
    
    # 극단적인 이상치만 제거 (다운샘플링 최소화)
    cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.5)  # 매우 관대한 기준
    merged_cloud = cl
    print(f"극단적 이상치 제거 후: {len(merged_cloud.points)} 포인트")
    
    # 법선 벡터 재계산
    merged_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
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
    vis.create_window(window_name="3D Pose Visualization", width=1024, height=768)
    
    # 포인트 클라우드와 메시 모두 추가
    vis.add_geometry(merged_cloud)
    if mesh is not None:
        vis.add_geometry(mesh)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([1, 1, 1])  # 검은색 배경
    
    # 카메라 위치 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([-0.3, 0.1, 0.5])
    ctr.set_up([0, 1, 0])

    # 시각화
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_3d_pose()