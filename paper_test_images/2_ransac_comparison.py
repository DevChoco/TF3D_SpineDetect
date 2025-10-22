"""
논문용 테스트 이미지 생성 #2: RANSAC 적용 전후 정렬 비교

이 스크립트는 RANSAC 기반 전역 정렬의 효과를 시각적으로 보여줍니다.
- 좌측: 초기 상태 (정렬 전)
- 중앙: ICP만 사용 (로컬 최적화만, 초기 위치에 민감)
- 우측: RANSAC + ICP (전역 정렬 후 정밀화)
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import copy

# 상위 디렉토리의 modules를 import하기 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Fixed_Main_3D_pose'))

from modules.pointcloud_generator import load_depth_map, create_point_cloud_from_depth


def compute_fpfh(pcd, voxel_size):
    """FPFH 특징 계산"""
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0
    
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def ransac_alignment(source, target, voxel_size=5.0):
    """RANSAC 기반 전역 정렬"""
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)
    
    src_fpfh = compute_fpfh(src_down, voxel_size)
    tgt_fpfh = compute_fpfh(tgt_down, voxel_size)
    
    distance_threshold = voxel_size * 2.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=20000, confidence=0.95)
    )
    
    return result.transformation, result


def icp_refinement(source, target, init_transform=np.eye(4)):
    """ICP 정밀화"""
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=10.0,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    return result.transformation, result


def calculate_alignment_error(source, target):
    """정렬 오차 계산"""
    distances = np.asarray(target.compute_point_cloud_distance(source))
    mean_error = np.mean(distances)
    std_error = np.std(distances)
    return mean_error, std_error


def capture_alignment_image(source, target, title=""):
    """정렬된 포인트 클라우드를 이미지로 캡처"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=600)
    vis.add_geometry(source)
    vis.add_geometry(target)
    
    # 렌더링 옵션
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 3.0
    
    # 카메라 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    ctr.set_front([0.5, -0.3, -0.8])
    ctr.set_up([0, 1, 0])  # Y축을 위로
    
    # 이미지 캡처
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    
    return np.asarray(image)


def main():
    print("="*60)
    print("논문용 이미지 생성 #2: RANSAC 정렬 성능 비교")
    print("="*60)
    
    # 테스트 데이터 로드
    front_path = r"D:\Lab2\--final_3D_Body--\3D_Body_Posture_Analysis\test2\DepthMap\여_정면\DepthMap0.bmp"
    right_path = r"D:\Lab2\--final_3D_Body--\3D_Body_Posture_Analysis\test2\DepthMap\여_R\DepthMap0.bmp"
    
    print("\n깊이맵 로딩 및 포인트 클라우드 생성 중...")
    depth_front = load_depth_map(front_path)
    depth_right = load_depth_map(right_path)
    
    if depth_front is None or depth_right is None:
        print("깊이맵 로드 실패!")
        return
    
    # 포인트 클라우드 생성
    pcd_front = create_point_cloud_from_depth(depth_front, "front")
    pcd_right = create_point_cloud_from_depth(depth_right, "right")
    
    # 법선 계산
    pcd_front.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    pcd_right.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    # 색상 설정 (비교를 위해)
    pcd_front.paint_uniform_color([0.8, 0.2, 0.2])  # 빨간색 (타겟)
    
    print(f"정면 포인트: {len(pcd_front.points):,}개")
    print(f"우측 포인트: {len(pcd_right.points):,}개")
    
    # 1. 초기 상태 (정렬 전)
    print("\n1. 초기 상태 캡처 중...")
    pcd_right_initial = copy.deepcopy(pcd_right)
    pcd_right_initial.paint_uniform_color([0.2, 0.2, 0.8])  # 파란색
    img_initial = capture_alignment_image(pcd_right_initial, pcd_front, "Initial")
    error_initial_mean, error_initial_std = calculate_alignment_error(pcd_right_initial, pcd_front)
    print(f"  초기 정렬 오차: {error_initial_mean:.2f} ± {error_initial_std:.2f} mm")
    
    # 2. ICP만 사용 (RANSAC 없이)
    print("\n2. ICP만 사용한 정렬...")
    pcd_right_icp_only = copy.deepcopy(pcd_right)
    transform_icp_only, result_icp_only = icp_refinement(pcd_right_icp_only, pcd_front)
    pcd_right_icp_only.transform(transform_icp_only)
    pcd_right_icp_only.paint_uniform_color([0.8, 0.5, 0.2])  # 주황색
    img_icp_only = capture_alignment_image(pcd_right_icp_only, pcd_front, "ICP Only")
    error_icp_mean, error_icp_std = calculate_alignment_error(pcd_right_icp_only, pcd_front)
    print(f"  ICP 전용 오차: {error_icp_mean:.2f} ± {error_icp_std:.2f} mm")
    print(f"  ICP fitness: {result_icp_only.fitness:.4f}")
    
    # 3. RANSAC + ICP 사용
    print("\n3. RANSAC + ICP 정렬...")
    pcd_right_ransac = copy.deepcopy(pcd_right)
    
    # RANSAC 전역 정렬
    print("  RANSAC 전역 정렬 중...")
    transform_ransac, result_ransac = ransac_alignment(pcd_right_ransac, pcd_front, voxel_size=5.0)
    pcd_right_ransac.transform(transform_ransac)
    print(f"  RANSAC fitness: {result_ransac.fitness:.4f}")
    
    # ICP 정밀화
    print("  ICP 정밀화 중...")
    transform_icp, result_icp = icp_refinement(pcd_right_ransac, pcd_front, init_transform=np.eye(4))
    pcd_right_ransac.transform(transform_icp)
    pcd_right_ransac.paint_uniform_color([0.2, 0.8, 0.2])  # 초록색
    img_ransac = capture_alignment_image(pcd_right_ransac, pcd_front, "RANSAC + ICP")
    error_ransac_mean, error_ransac_std = calculate_alignment_error(pcd_right_ransac, pcd_front)
    print(f"  RANSAC+ICP 오차: {error_ransac_mean:.2f} ± {error_ransac_std:.2f} mm")
    print(f"  최종 ICP fitness: {result_icp.fitness:.4f}")
    
    # 4. 비교 이미지 생성
    print("\n비교 이미지 생성 중...")
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(2, 3, height_ratios=[1, 0.2], hspace=0.3, wspace=0.2)
    
    # 초기 상태
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_initial)
    ax1.set_title('(a) Initial State (No Alignment)', fontsize=12, fontweight='bold')
    ax1.text(0.5, -0.08, f'Mean Error: {error_initial_mean:.2f} mm\nStd: {error_initial_std:.2f} mm', 
             ha='center', va='top', transform=ax1.transAxes, fontsize=10, color='blue')
    ax1.axis('off')
    
    # ICP만 사용
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_icp_only)
    ax2.set_title('(b) ICP Only (Local Optimization)', fontsize=12, fontweight='bold')
    ax2.text(0.5, -0.08, f'Mean Error: {error_icp_mean:.2f} mm\nFitness: {result_icp_only.fitness:.4f}', 
             ha='center', va='top', transform=ax2.transAxes, fontsize=10, color='orange')
    ax2.axis('off')
    
    # RANSAC + ICP
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_ransac)
    ax3.set_title('(c) RANSAC + ICP (Global + Local)', fontsize=12, fontweight='bold')
    ax3.text(0.5, -0.08, f'Mean Error: {error_ransac_mean:.2f} mm\nFitness: {result_icp.fitness:.4f}', 
             ha='center', va='top', transform=ax3.transAxes, fontsize=10, color='green')
    ax3.axis('off')
    
    # 통계 비교
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis('off')
    
    improvement_icp = ((error_initial_mean - error_icp_mean) / error_initial_mean) * 100
    improvement_ransac = ((error_initial_mean - error_ransac_mean) / error_initial_mean) * 100
    
    stats_text = f"""
    Alignment Quality Comparison:
    
    Method              | Mean Error (mm) | Improvement | Fitness  | Notes
    --------------------|-----------------|-------------|----------|------------------------------------------
    Initial (No Align)  | {error_initial_mean:7.2f}        | -           | -        | Misaligned clouds
    ICP Only            | {error_icp_mean:7.2f}        | {improvement_icp:5.1f}%      | {result_icp_only.fitness:.4f}   | Local minimum (depends on initial pose)
    RANSAC + ICP        | {error_ransac_mean:7.2f}        | {improvement_ransac:5.1f}%      | {result_icp.fitness:.4f}   | Global optimum (robust to initial pose)
    
    Conclusion: RANSAC provides robust global initialization, preventing ICP from converging to local minima.
    """
    
    ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', 
                 fontsize=10, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Point Cloud Alignment Comparison: RANSAC vs ICP-Only', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # 저장
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, 'output_2_ransac_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n결과 이미지 저장: {output_path}")
    
    print("\n완료!")
    plt.show()


if __name__ == "__main__":
    main()
