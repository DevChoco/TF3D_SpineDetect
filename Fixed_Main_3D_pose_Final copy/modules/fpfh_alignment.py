import numpy as np
import cv2
import open3d as o3d
import copy
import importlib
from scipy.spatial import cKDTree

# 상대 임포트 (패키지로 사용될 때)
try:
    from .pointcloud_generator import preprocess_for_icp
except ImportError:
    # 절대 임포트 (직접 실행될 때)
    try:
        from pointcloud_generator import preprocess_for_icp
    except ImportError:
        # preprocess_for_icp가 실제로 사용되지 않는 경우 None으로 설정
        preprocess_for_icp = None
        print("Warning: preprocess_for_icp를 임포트할 수 없습니다.")


def compute_fpfh(pcd, voxel_size):
    """
    포인트 클라우드에서 FPFH 특징을 계산합니다.
    
    Args:
        pcd (o3d.geometry.PointCloud): 포인트 클라우드
        voxel_size (float): 복셀 크기
        
    Returns:
        o3d.pipelines.registration.Feature: FPFH 특징
    """
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def global_registration_fpfh_ransac(source, target, voxel_size=5.0, ransac_iter=20000):
    """
    FPFH 특징 + RANSAC을 이용한 전역 초기 정합 (Open3D)
    
    Args:
        source (o3d.geometry.PointCloud): 소스 포인트 클라우드
        target (o3d.geometry.PointCloud): 타겟 포인트 클라우드
        voxel_size (float): 다운샘플링 복셀 크기
        ransac_iter (int): RANSAC 반복 횟수
        
    Returns:
        tuple: (transformation matrix, result object)
    """
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)
    if len(src_down.points) == 0 or len(tgt_down.points) == 0:
        return np.eye(4), None
    
    src_fpfh = compute_fpfh(src_down, voxel_size)
    tgt_fpfh = compute_fpfh(tgt_down, voxel_size)

    distance_threshold = voxel_size * 2.5  # 1.5에서 2.5로 증가: 더 넓은 correspondence 허용
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, mutual_filter=False,  # mutual_filter 비활성화: 단방향 매칭 허용
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,  # 4에서 3으로 감소: 더 적은 점으로 변환 추정
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),  # 0.9에서 0.8로 완화
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=ransac_iter, confidence=0.95)  # 0.999에서 0.95로 완화
    )
    return result.transformation, result


def multi_scale_icp(source, target, voxel_list=[20.0, 10.0, 5.0], use_point_to_plane=True, max_iter_per_scale=[150, 200, 250, 300]):
    """
    복셀 기반 다중 스케일 ICP: 거친 정렬에서 정밀 정렬로 진행
    
    Args:
        source (o3d.geometry.PointCloud): 소스 포인트 클라우드
        target (o3d.geometry.PointCloud): 타겟 포인트 클라우드
        voxel_list (list): 복셀 크기 리스트 (큰 것부터 작은 것 순서)
        use_point_to_plane (bool): Point-to-Plane ICP 사용 여부
        max_iter_per_scale (list): 각 스케일별 최대 반복 횟수
        
    Returns:
        tuple: (final transformation, final result)
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

        distance_threshold = voxel * 3.0  # 2.5에서 3.0으로 증가: 매우 관대한 correspondence
        estimation = (o3d.pipelines.registration.TransformationEstimationPointToPlane() if use_point_to_plane
                      else o3d.pipelines.registration.TransformationEstimationPointToPoint())

        result = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=distance_threshold,
            init=current_trans,
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter_per_scale[i if i < len(max_iter_per_scale) else -1]
            )
        )
        current_trans = result.transformation
        final_result = result

    if final_result is None:
        # 다운샘플 결과가 빈 경우 원본으로 1회 시도
        estimation = (o3d.pipelines.registration.TransformationEstimationPointToPlane() if use_point_to_plane
                      else o3d.pipelines.registration.TransformationEstimationPointToPoint())
        final_result = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=voxel_list[-1] * 2.0,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        current_trans = final_result.transformation

    return current_trans, final_result


def _get_centroid(pcd):
    """포인트 클라우드의 중심점을 계산합니다."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return np.zeros(3, dtype=np.float64)
    return pts.mean(axis=0)


def _clamp_rotation(R, max_angle_rad):
    """Rodrigues 축-각 표현으로 회전 각을 max_angle_rad로 제한합니다."""
    rvec, _ = cv2.Rodrigues(R)
    angle = float(np.linalg.norm(rvec))
    if angle <= 1e-12 or angle <= max_angle_rad:
        return R
    rvec_unit = (rvec / angle) * max_angle_rad
    R_limited, _ = cv2.Rodrigues(rvec_unit)
    return R_limited


def small_rotation_icp(source, target, voxel_list=[20.0, 10.0, 5.0], max_iter_per_scale=[30, 30, 60],
                       max_angle_deg_per_scale=[2.0, 1.0, 0.5], init_trans=None):
    """
    아주 미세한 각도만 허용하는 ICP.
    
    Args:
        source (o3d.geometry.PointCloud): 소스 포인트 클라우드
        target (o3d.geometry.PointCloud): 타겟 포인트 클라우드
        voxel_list (list): 복셀 크기 리스트
        max_iter_per_scale (list): 각 스케일별 최대 반복 횟수
        max_angle_deg_per_scale (list): 각 스케일별 최대 허용 각도 (도 단위)
        init_trans (np.ndarray): 초기 변환 행렬
        
    Returns:
        tuple: (transformation matrix, result stub)
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

        # 법선 계산
        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
        tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))

        distance_threshold = voxel * 1.5
        result = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=distance_threshold,
            init=current_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter_per_scale[i if i < len(max_iter_per_scale) else -1]
            )
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
    """
    회전을 허용하지 않는 번역(translation) 전용 ICP.
    
    Args:
        source (o3d.geometry.PointCloud): 소스 포인트 클라우드
        target (o3d.geometry.PointCloud): 타겟 포인트 클라우드
        voxel_list (list): 복셀 크기 리스트
        max_iter_per_scale (list): 각 스케일별 최대 반복 횟수
        max_corr_factor (float): 최대 대응점 거리 계수
        
    Returns:
        tuple: (transformation matrix, result dict)
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
    """
    pycpd 기반 비강직 보정 (DeformableRegistration)
    
    Args:
        source_pcd (o3d.geometry.PointCloud): 소스 포인트 클라우드
        target_pcd (o3d.geometry.PointCloud): 타겟 포인트 클라우드
        max_points (int): 최대 포인트 수 (계산량 제어)
        cpd_beta (float): CPD beta 매개변수
        cpd_lambda (float): CPD lambda 매개변수
        cpd_iter (int): 최대 반복 횟수
        w (float): 노이즈 비율 매개변수
        
    Returns:
        o3d.geometry.PointCloud: 변형된 포인트 클라우드
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


def align_point_clouds_fpfh(source, target, params=None):
    """
    개선된 FPFH 기반 정렬: (1) global FPFH+RANSAC -> (2) multi-scale ICP -> (3) optional CPD refine
    
    Args:
        source (o3d.geometry.PointCloud): 소스 포인트 클라우드
        target (o3d.geometry.PointCloud): 타겟 포인트 클라우드
        params (dict): 정렬 매개변수 딕셔너리
        
    Returns:
        o3d.geometry.PointCloud: 정렬된 포인트 클라우드
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
    use_point_to_plane_icp = params.get('use_point_to_plane_icp', True)  # Point-to-Plane ICP 사용 여부 (아니면 point-to-point로)

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

        # 2) Multi-scale ICP (회전 허용, Point-to-Plane 또는 Point-to-Point)
        icp_type = "Point-to-Plane" if use_point_to_plane_icp else "Point-to-Point"
        print(f"  Multi-scale {icp_type} ICP 수행 중...")
        trans_icp, icp_result = multi_scale_icp(src, tgt, voxel_list=voxel_list, use_point_to_plane=use_point_to_plane_icp)
        print(f"  Multi-scale {icp_type} ICP fitness={icp_result.fitness:.4f}, rmse={icp_result.inlier_rmse:.4f}")
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

    # 정합 품질 평가: ICP 결과 객체에서 직접 가져오기 (Point-to-Plane과 Point-to-Point 구분)
    if icp_result is not None:
        fitness = icp_result.fitness
        rmse = icp_result.inlier_rmse
        icp_type = "Point-to-Plane" if use_point_to_plane_icp else "Point-to-Point"
        print(f"  정렬 평가 ({icp_type} ICP): fitness={fitness:.4f}, RMSE={rmse:.4f}")
    else:
        # fallback: 수동 계산 (ICP 결과가 없는 경우)
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