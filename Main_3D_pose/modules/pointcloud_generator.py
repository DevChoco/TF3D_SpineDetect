"""
포인트 클라우드 생성 및 전처리 모듈

이 모듈은 다음 기능을 제공합니다:
- 깊이맵에서 포인트 클라우드 생성
- 마스크 생성 및 적용
- 노이즈 제거 및 전처리
"""

import numpy as np
import cv2
import open3d as o3d
from PIL import Image


def load_depth_map(file_path):
    """
    깊이맵 이미지를 로드하고 정사각형으로 잘라 정규화합니다.
    
    Args:
        file_path (str): 깊이맵 이미지 파일 경로
        
    Returns:
        np.ndarray: 정규화된 깊이맵 (0-1 범위) 또는 None
    """
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


def create_mask_from_depth(depth_map, threshold_low=0.1, threshold_high=0.9):
    """
    깊이맵에서 이진 마스크를 생성합니다.
    
    Args:
        depth_map (np.ndarray): 정규화된 깊이맵 (0-1 범위)
        threshold_low (float): 하한 임계값 (이 값 이하는 배경으로 간주)
        threshold_high (float): 상한 임계값 (이 값 이상은 노이즈로 간주)
    
    Returns:
        np.ndarray: 이진 마스크 (True: 유효한 영역, False: 배경/노이즈)
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


def create_point_cloud_from_depth(depth_map, view):
    """
    깊이맵에서 포인트 클라우드를 생성합니다.
    
    Args:
        depth_map (np.ndarray): 정규화된 깊이맵
        view (str): 뷰 이름 ("front", "right", "left", "back")
        
    Returns:
        o3d.geometry.PointCloud: 포인트 클라우드 또는 None
    """
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
    if len(valid_points) > 100000:
        indices = np.random.choice(len(valid_points), 100000, replace=False)
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
    
    # 극단적인 이상치만 제거
    print(f"  {view} 뷰 이상치 제거 시작...")
    if len(valid_points) > 500:  # 포인트가 충분할 때만 이상치 제거
        pcd = remove_noise_from_pointcloud(pcd, method="statistical", verbose=True)
    else:
        print(f"  포인트가 적어 이상치 제거 생략: {len(valid_points)}개")
    
    return pcd


def remove_noise_from_pointcloud(pcd, method="statistical", verbose=True):
    """
    포인트 클라우드에서 이상치만 제거합니다.
    
    Args:
        pcd (o3d.geometry.PointCloud): Open3D PointCloud 객체
        method (str): 노이즈 제거 방법 ("statistical", "radius", "all")
        verbose (bool): 로그 출력 여부
    
    Returns:
        o3d.geometry.PointCloud: 이상치가 제거된 PointCloud 객체
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
    
    return cleaned_pcd


def preprocess_for_icp(pcd, aggressive=False):
    """
    ICP 정렬을 위한 포인트 클라우드 전처리 - 이상치만 제거
    
    Args:
        pcd (o3d.geometry.PointCloud): Open3D PointCloud 객체
        aggressive (bool): True면 더 강력한 이상치 제거 적용
    
    Returns:
        o3d.geometry.PointCloud: 전처리된 PointCloud 객체
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