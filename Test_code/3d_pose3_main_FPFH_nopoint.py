import numpy as np
import cv2
import open3d as o3d
import os
import copy
import importlib
from scipy.spatial import cKDTree

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
        points = np.stack([x, -y, depth_map * scale * 1.1], axis=-1)
    elif view == "right":
        points = np.stack([depth_map * scale * 3, -y, -x], axis=-1)  # 우측 깊이 2배
    elif view == "left":
        points = np.stack([-depth_map * scale * 3, -y, x], axis=-1)  # 좌측 깊이 2배
    elif view == "back":
        points = np.stack([-x, -y, -depth_map * scale * 1.1], axis=-1)

    # 마스크를 적용하여 유효한 포인트만 선택
    valid_points = points[mask]
    
    # 추가적으로 깊이 임계값도 적용 (마스크와 함께 사용)
    valid_depths = depth_map[mask]
    depth_threshold = 0.01
    final_valid_mask = valid_depths > depth_threshold
    valid_points = valid_points[final_valid_mask]
    
    print(f"{view} 뷰: 마스크 적용 전 {np.sum(mask)} 포인트, 적용 후 {len(valid_points)} 포인트")
    
    # 너무 많은 포인트가 있는 경우 추가 다운샘플링
    if len(valid_points) > 50000:
        indices = np.random.choice(len(valid_points), 50000, replace=False)
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

def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh

def global_registration_fpfh_ransac(source, target, voxel_size=5.0, ransac_iter=20000):
    """FPFH 특징 + RANSAC을 이용한 전역 초기 정합 (Open3D)
       반환: transformation (4x4) 및 result obj
    """
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)
    if len(src_down.points) == 0 or len(tgt_down.points) == 0:
        return np.eye(4), None
    src_fpfh = compute_fpfh(src_down, voxel_size)
    tgt_fpfh = compute_fpfh(tgt_down, voxel_size)

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=ransac_iter, confidence=0.999)
    )
    return result.transformation, result

def multi_scale_icp(source, target, voxel_list=[20.0, 10.0, 5.0], use_point_to_plane=True, max_iter_per_scale=[50,50,100]):
    """voxel-based multi-scale ICP: coarse -> fine (데이터 단위가 픽셀/스케일 100 근처 가정)
       source/target are o3d.geometry.PointCloud (원본)
       반환: final_transformation, final_result
    """
    current_trans = np.eye(4)
    final_result = None
    for i, voxel in enumerate(voxel_list):
        src_down = source.voxel_down_sample(voxel)
        tgt_down = target.voxel_down_sample(voxel)
        if len(src_down.points) == 0 or len(tgt_down.points) == 0:
            continue
        if use_point_to_plane:
            src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
            tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))

        distance_threshold = voxel * 1.5
        estimation = (o3d.pipelines.registration.TransformationEstimationPointToPlane() if use_point_to_plane
                      else o3d.pipelines.registration.TransformationEstimationPointToPoint())

        result = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=distance_threshold,
            init=current_trans,
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter_per_scale[i if i < len(max_iter_per_scale) else -1])
        )
        current_trans = result.transformation
        final_result = result

    if final_result is None:
        # 다운샘플 결과가 빈 경우 원본으로 1회 시도
        estimation = (o3d.pipelines.registration.TransformationEstimationPointToPlane() if use_point_to_plane
                      else o3d.pipelines.registration.TransformationEstimationPointToPoint())
        final_result = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=voxel_list[-1] * 1.5,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        current_trans = final_result.transformation

    return current_trans, final_result

def _get_centroid(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return np.zeros(3, dtype=np.float64)
    return pts.mean(axis=0)

def _clamp_rotation(R: np.ndarray, max_angle_rad: float) -> np.ndarray:
    """Rodrigues 축-각 표현으로 회전 각을 max_angle_rad로 제한합니다."""
    rvec, _ = cv2.Rodrigues(R)
    angle = float(np.linalg.norm(rvec))
    if angle <= 1e-12 or angle <= max_angle_rad:
        return R
    rvec_unit = (rvec / angle) * max_angle_rad
    R_limited, _ = cv2.Rodrigues(rvec_unit)
    return R_limited

def small_rotation_icp(source, target, voxel_list=[20.0, 10.0, 5.0], max_iter_per_scale=[30,30,60],
                       max_angle_deg_per_scale=[2.0, 1.0, 0.5], init_trans: np.ndarray | None = None):
    """아주 미세한 각도만 허용하는 ICP.
    - 각 스케일에서 Open3D ICP를 수행하되, 결과의 회전 증가분을 각 스케일 최대 각도로 클램프합니다.
    - 누적 변환 current_trans을 유지하며, delta 회전만 제한.
    반환: (4x4 transformation, stub_result)
    """
    # 초기 변환: 전달된 init_trans 사용, 없으면 중심 정렬 번역으로 시작
    if init_trans is None:
        current_trans = np.eye(4)
        t0 = _get_centroid(target) - _get_centroid(source)
        current_trans[:3, 3] = t0
    else:
        current_trans = init_trans.copy()

    last_fitness, last_rmse = 0.0, np.inf

    for i, voxel in enumerate(voxel_list):
        src_down = source.voxel_down_sample(voxel)
        tgt_down = target.voxel_down_sample(voxel)
        if len(src_down.points) == 0 or len(tgt_down.points) == 0:
            continue

        # 법선 (point-to-plane가 보통 더 안정적이나 여기서는 제한 회전이므로 point-to-plane 사용)
        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
        tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))

        distance_threshold = voxel * 1.5
        result = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=distance_threshold,
            init=current_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter_per_scale[i if i < len(max_iter_per_scale) else -1])
        )

        prev_trans = current_trans
        T_est = result.transformation
        # delta = T_est * inv(prev)
        delta = T_est @ np.linalg.inv(prev_trans)
        R_delta = delta[:3, :3]
        t_delta = delta[:3, 3]

        # 회전 각도 제한
        max_ang_rad = np.deg2rad(max_angle_deg_per_scale[i if i < len(max_angle_deg_per_scale) else -1])
        R_delta_limited = _clamp_rotation(R_delta, max_ang_rad)

        # 제한된 delta 재조합
        delta_limited = np.eye(4)
        delta_limited[:3, :3] = R_delta_limited
        delta_limited[:3, 3] = t_delta

        current_trans = delta_limited @ prev_trans

        # 스케일별 평가 (최근접 거리 기반)
        src_eval = copy.deepcopy(source).transform(current_trans)
        dists = np.asarray(target.compute_point_cloud_distance(src_eval))
        if dists.size > 0:
            inliers = dists < distance_threshold
            last_fitness = float(np.sum(inliers)) / max(1, len(dists))
            last_rmse = float(np.sqrt(np.mean((dists[inliers] ** 2))) if np.any(inliers) else np.inf)

    result_stub = type("ICPResult", (), {})()
    result_stub.fitness = last_fitness
    result_stub.inlier_rmse = last_rmse
    return current_trans, result_stub

def translation_only_icp(source, target, voxel_list=[20.0, 10.0, 5.0], max_iter_per_scale=[20, 20, 30], max_corr_factor=1.5):
    """회전을 허용하지 않는 번역(translation) 전용 ICP.
    - 각 스케일(voxel)에서 최근접 이웃 대응을 만들고, 평균 변위로 번역 벡터만 갱신
    - R은 항상 항등행렬 I, t만 누적
    반환: (4x4 transformation, {'fitness':..., 'rmse':...})
    """
    # 초기 번역: 중심 정렬
    t = (_get_centroid(target) - _get_centroid(source)).astype(np.float64)
    last_fitness, last_rmse = 0.0, np.inf

    for i, voxel in enumerate(voxel_list):
        # 다운샘플
        src_down = source.voxel_down_sample(voxel)
        tgt_down = target.voxel_down_sample(voxel)
        if len(src_down.points) == 0 or len(tgt_down.points) == 0:
            continue

        src_np = np.asarray(src_down.points)
        tgt_np = np.asarray(tgt_down.points)
        tree = cKDTree(tgt_np)
        max_corr_dist = voxel * max_corr_factor

        for it in range(max_iter_per_scale[i if i < len(max_iter_per_scale) else -1]):
            src_trans = src_np + t  # 회전 없이 번역만 적용
            dists, idx = tree.query(src_trans, k=1)
            inlier_mask = dists < max_corr_dist
            if not np.any(inlier_mask):
                break
            src_in = src_trans[inlier_mask]
            tgt_in = tgt_np[idx[inlier_mask]]
            # 평균 변위
            delta = (tgt_in - src_in).mean(axis=0)
            t += delta
            # 수렴 체크
            if np.linalg.norm(delta) < max(1e-3, voxel * 1e-3):
                break

        # 스케일별 평가
        if len(tgt_np) > 0:
            src_final = src_np + t
            dists, _ = tree.query(src_final, k=1)
            inliers = dists < max_corr_dist
            last_fitness = float(np.sum(inliers)) / max(1, len(dists))
            last_rmse = float(np.sqrt(np.mean((dists[inliers] ** 2))) if np.any(inliers) else np.inf)

    # 최종 4x4 변환 (R=I, t=t)
    T = np.eye(4)
    T[:3, 3] = t
    result_stub = type("ICPResult", (), {})()
    result_stub.fitness = last_fitness
    result_stub.inlier_rmse = last_rmse
    return T, result_stub

def cpd_refine_np(source_pcd, target_pcd, max_points=2000, cpd_beta=2.0, cpd_lambda=2.0, cpd_iter=40, w=0.0):
    """pycpd 기반 비강직 보정 (DeformableRegistration)
       source_pcd, target_pcd: o3d PointCloud
       반환: warped o3d PointCloud (source 기준으로 변형)
    """
    try:
        cpd_mod = importlib.import_module("pycpd")
        DeformableRegistration = getattr(cpd_mod, "DeformableRegistration")
    except Exception as e:
        raise RuntimeError("pycpd가 설치되어 있지 않습니다. requirements.txt에 pycpd 추가 후 설치하세요.") from e
    source_np = np.asarray(source_pcd.points)
    target_np = np.asarray(target_pcd.points)

    if source_np.shape[0] == 0 or target_np.shape[0] == 0:
        return copy.deepcopy(source_pcd)

    # 서브샘플링(계산량 제어)
    if source_np.shape[0] > max_points:
        idx = np.random.choice(source_np.shape[0], max_points, replace=False)
        source_sub = source_np[idx]
    else:
        source_sub = source_np.copy()
    if target_np.shape[0] > max_points:
        idx2 = np.random.choice(target_np.shape[0], max_points, replace=False)
        target_sub = target_np[idx2]
    else:
        target_sub = target_np.copy()

    reg = DeformableRegistration(X=target_sub, Y=source_sub, max_iterations=cpd_iter, w=w, beta=cpd_beta, lambd=cpd_lambda)
    TY, _ = reg.register()  # TY shape: (N_sub, 3)

    # 간단 보간: 각 원본 source 점을 nearest neighbor로 매핑하여 displacement 적용
    tree = cKDTree(source_sub)
    dists, idx_nn = tree.query(source_np, k=1)
    displacement = TY[idx_nn] - source_sub[idx_nn]
    warped = source_np + displacement

    warped_pcd = o3d.geometry.PointCloud()
    warped_pcd.points = o3d.utility.Vector3dVector(warped)
    if source_pcd.has_colors():
        warped_pcd.colors = copy.deepcopy(source_pcd.colors)
    return warped_pcd

def align_point_clouds_v2(source, target, params=None):
    """개선된 정렬: (1) global FPFH+RANSAC -> (2) multi-scale ICP -> (3) optional CPD refine
       params: dict으로 옵션 전달
    """
    if source is None or target is None or len(source.points) == 0 or len(target.points) == 0:
        return source
    if params is None:
        params = {}
    voxel_coarse = params.get('voxel_coarse', 5.0)
    voxel_list = params.get('voxel_list', [20.0, 10.0, 5.0])
    ransac_iter = params.get('ransac_iter', 20000)
    use_cpd = params.get('use_cpd', True)
    cpd_params = params.get('cpd_params', {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40})
    fitness_threshold_accept = params.get('fitness_threshold_accept', 0.02)
    allow_rotation = params.get('allow_rotation', True)
    allow_small_rotation = params.get('allow_small_rotation', False)

    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)

    if allow_rotation:
        # 1) Global 초기 정합 (FPFH + RANSAC)
        try:
            init_trans, ransac_result = global_registration_fpfh_ransac(src, tgt, voxel_size=voxel_coarse, ransac_iter=ransac_iter)
            if ransac_result is not None:
                print(f"  Global RANSAC fitness={getattr(ransac_result,'fitness',None)}, inlier_rmse={getattr(ransac_result,'inlier_rmse',None)}")
            src.transform(init_trans)
        except Exception as e:
            print("  Global RANSAC 실패:", e)

        # 2) Multi-scale ICP (회전 허용)
        trans_icp, icp_result = multi_scale_icp(src, tgt, voxel_list=voxel_list, use_point_to_plane=True)
        print(f"  Multi-scale ICP fitness={icp_result.fitness:.4f}, rmse={icp_result.inlier_rmse:.4f}")
    else:
        print("  회전 비허용 모드: translation-only ICP 수행")
        # 전역 RANSAC은 사용하지 않고 바로 번역 전용 ICP
        trans_icp, icp_result = translation_only_icp(src, tgt, voxel_list=voxel_list, max_iter_per_scale=[30,30,50])
        print(f"  Translation-Only ICP fitness={icp_result.fitness:.4f}, rmse={icp_result.inlier_rmse:.4f}")
    
    # allow_small_rotation이 요청된 경우, translation-only 결과에서 아주 미세 회전만 허용해 추가 정련
    if not allow_rotation and allow_small_rotation:
        print("  소각 회전 허용 모드: small-rotation ICP로 미세 보정")
        trans_icp, icp_result = small_rotation_icp(src, tgt, voxel_list=voxel_list,
                                                   max_iter_per_scale=[20,20,40],
                                                   max_angle_deg_per_scale=[2.0, 1.0, 0.5],
                                                   init_trans=trans_icp)
        print(f"  Small-Rotation ICP fitness={icp_result.fitness:.4f}, rmse={icp_result.inlier_rmse:.4f}")

    # 최종 변환 적용
    src_transformed = src.transform(trans_icp)

    # 정합 품질 평가 (최근접 거리 기반 추정 fitness)
    distances = np.asarray(tgt.compute_point_cloud_distance(src_transformed))
    if distances.size == 0:
        return src_transformed
    inliers = distances < (voxel_list[-1] * 2.0)
    fitness = float(np.sum(inliers)) / max(1, len(distances))
    print(f"  정렬 평가: estimated fitness (nearest dist<{voxel_list[-1]*2.0:.4f}) = {fitness:.4f}")

    # 3) 조건부 CPD 비강직 보정 (옵션)
    if use_cpd and allow_rotation:
        attempt_cpd = params.get('force_cpd', False) or (fitness < 0.25)
        if attempt_cpd:
            print("  CPD 비강직 보정 시도 (조건부)...")
            try:
                warped = cpd_refine_np(src_transformed, tgt, max_points=cpd_params.get('max_points',1500),
                                       cpd_beta=cpd_params.get('cpd_beta',2.0), cpd_lambda=cpd_params.get('cpd_lambda',2.0),
                                       cpd_iter=cpd_params.get('cpd_iter',40))
                distances2 = np.asarray(tgt.compute_point_cloud_distance(warped))
                fitness2 = float(np.sum(distances2 < (voxel_list[-1]*2.0))) / max(1, len(distances2))
                print(f"  CPD 후 fitness = {fitness2:.4f}")
                if fitness2 >= fitness or fitness < fitness_threshold_accept:
                    print("  CPD 결과 채택")
                    return warped
                else:
                    print("  CPD 결과 기각 (개선 없음)")
            except Exception as e:
                print("  CPD 오류:", e)

    return src_transformed

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
        "front": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_정면.bmp",
        "right": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_오른쪽.bmp",
        "left": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_왼쪽.bmp",
        "back": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_후면.bmp"
    }

    # views = {
    #     "front": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\남성\남_정면.bmp",
    #     "right": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\남성\남_오른쪽.bmp",
    #     "left": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\남성\남_왼쪽.bmp",
    #     "back": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\남성\남_후면.bmp"
    # }
    
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
        params_align = {
            'voxel_coarse': 5.0,
            'voxel_list': [20.0, 10.0, 5.0],
            'ransac_iter': 20000,
            'use_cpd': False,
            'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
            'fitness_threshold_accept': 0.02,
            'force_cpd': False,
            'allow_rotation': False,
            'allow_small_rotation': True
        }
        left_aligned = align_point_clouds_v2(point_clouds["left"], front_target, params=params_align)
        aligned_clouds.append(left_aligned)
    
    if "right" in point_clouds:
        print("\n우측 뷰를 정면과 정렬...")
        params_align = {
            'voxel_coarse': 5.0,
            'voxel_list': [20.0, 10.0, 5.0],
            'ransac_iter': 20000,
            'use_cpd': False,
            'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
            'fitness_threshold_accept': 0.02,
            'force_cpd': False,
            'allow_rotation': False,
            'allow_small_rotation': True
        }
        right_aligned = align_point_clouds_v2(point_clouds["right"], front_target, params=params_align)
        aligned_clouds.append(right_aligned)
    
    # 후면은 무조건 좌/우 포인트 클라우드(정렬 결과)에만 정렬
    if "back" in point_clouds:
        print("\n후면 뷰를 좌/우 누적 클라우드에 정렬...")
        side_target = o3d.geometry.PointCloud()
        st_points = []
        st_colors = []

        # 좌측 정렬 결과
        if left_aligned is not None and len(left_aligned.points) > 0:
            st_points.extend(np.asarray(left_aligned.points))
            if left_aligned.has_colors():
                st_colors.extend(np.asarray(left_aligned.colors))

        # 우측 정렬 결과
        if right_aligned is not None and len(right_aligned.points) > 0:
            st_points.extend(np.asarray(right_aligned.points))
            if right_aligned.has_colors():
                st_colors.extend(np.asarray(right_aligned.colors))

        if len(st_points) == 0:
            print("  좌/우 타겟이 비어 있어 후면 정렬을 건너뜁니다.")
        else:
            side_target.points = o3d.utility.Vector3dVector(np.array(st_points))
            if len(st_colors) == len(st_points) and len(st_colors) > 0:
                side_target.colors = o3d.utility.Vector3dVector(np.array(st_colors))

            # 후면을 좌/우 타겟에 정렬
            params_align = {
                'voxel_coarse': 5.0,
                'voxel_list': [25.0, 12.0, 6.0],
                'ransac_iter': 30000,
                'use_cpd': False,
                'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
                'fitness_threshold_accept': 0.02,
                'force_cpd': False,
                'allow_rotation': False,
                'allow_small_rotation': True
            }
            back_aligned = align_point_clouds_v2(point_clouds["back"], side_target, params=params_align)
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
        
        # OBJ 형식으로 저장
        mesh_path = os.path.join(output_dir, "body_mesh.obj")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"메시가 저장되었습니다 (OBJ): {mesh_path}")
        
        # PLY 형식으로 저장
        mesh_ply_path = os.path.join(output_dir, "body_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_ply_path, mesh)
        print(f"메시가 저장되었습니다 (PLY): {mesh_ply_path}")
        
        # STL 형식으로 저장 (3D 프린팅 및 CAD용 - 품질 향상)
        mesh_stl_path = os.path.join(output_dir, "body_mesh.stl")
        try:
            # STL 저장 전 메시 최적화
            mesh_optimized = mesh.simplify_quadric_decimation(target_number_of_triangles=len(mesh.triangles))
            mesh_optimized.remove_degenerate_triangles()
            mesh_optimized.remove_duplicated_triangles()
            mesh_optimized.remove_duplicated_vertices()
            mesh_optimized.remove_non_manifold_edges()
            
            # 법선 벡터 재계산 (STL 품질 향상)
            mesh_optimized.compute_vertex_normals()
            mesh_optimized.compute_triangle_normals()
            
            # STL 저장
            success = o3d.io.write_triangle_mesh(mesh_stl_path, mesh_optimized)
            if success:
                print(f"메시가 저장되었습니다 (STL): {mesh_stl_path}")
                print(f"  STL 메시 정보: {len(mesh_optimized.vertices)}개 정점, {len(mesh_optimized.triangles)}개 삼각형")
            else:
                print(f"STL 저장 실패: {mesh_stl_path}")
        except Exception as e:
            print(f"STL 저장 중 오류 발생: {e}")
            # 최적화 없이 원본 메시로 다시 시도
            try:
                o3d.io.write_triangle_mesh(mesh_stl_path, mesh)
                print(f"STL 저장 성공 (최적화 없음): {mesh_stl_path}")
            except Exception as e2:
                print(f"STL 저장 완전 실패: {e2}")
    
    # 시각화 (메시만 표시)
    if mesh is not None:
        # 초기 카메라 뷰포인트 설정
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Body Mesh Visualization", width=1024, height=768)
        
        # 메시만 추가 (포인트 클라우드는 제외)
        vis.add_geometry(mesh)
        
        # 렌더링 옵션 설정
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 어두운 회색 배경
        opt.mesh_show_wireframe = False  # 와이어프레임 숨김
        opt.mesh_show_back_face = True   # 뒷면도 표시
        opt.light_on = True              # 조명 활성화
        
        # 카메라 위치 설정
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.5, -0.5, -0.5])
        ctr.set_up([0, -1, 0])
        
        # 시각화
        print("\n3D 메시 시각화 창이 열렸습니다. 창을 닫으면 프로그램이 종료됩니다.")
        vis.run()
        vis.destroy_window()
    else:
        print("메시가 생성되지 않아 시각화를 건너뜁니다.")

if __name__ == "__main__":
    # 여기에 성별 및 파일 경로를 설정하여 테스트할 수 있습니다.
    # 예: gender = 'male', view = 'front', file_path = 'test/정상/정면_남/DepthMap0.bmp'
    
    # posture_analysis 모듈의 함수를 호출하기 위한 예시 코드
    # 이 코드를 직접 실행하기보다는, posture_analysis.py를 메인으로 사용하고
    # 이 파일의 함수들을 임포트해서 사용하는 것을 권장합니다.
    
    # 1. 성별 결정 (파일 경로 기반)
    # 이 예시에서는 '남' 또는 '여'가 포함된 경로로 성별을 추론합니다.
    # 실제 애플리케이션에서는 더 견고한 방법이 필요할 수 있습니다.
    file_path_example = 'test/정상/정면_남/DepthMap0.bmp'
    gender = 'female' if '여' in file_path_example else 'male'

    # 2. 4개 뷰의 포인트 클라우드 생성 및 병합
    # (이 부분은 3d_pose3_main_FPFH.py의 기존 로직을 그대로 사용)
    # ... (중략) ...
    # final_pcd = ... (병합 및 정렬된 최종 포인트 클라우드)
    
    # 3. 자세 분석 실행
    # from posture_analysis import analyze_posture
    # analyze_posture(final_pcd, gender)
    pass
    visualize_3d_pose()