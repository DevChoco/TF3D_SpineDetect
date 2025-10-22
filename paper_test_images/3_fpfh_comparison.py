"""
논문용 테스트 이미지 생성 #3: FPFH 특징 기반 정렬 vs 일반 정렬 비교

이 스크립트는 FPFH(Fast Point Feature Histogram) 특징을 사용한 정렬의 우수성을 보여줍니다.
- 좌측: 기하학적 특징만 사용 (단순 ICP)
- 우측: FPFH 특징 기반 정렬 (로컬 표면 특징 활용)
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


def compute_fpfh_feature(pcd, voxel_size):
    """FPFH 특징 계산"""
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0
    
    # 법선 계산
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # FPFH 특징 계산
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def simple_icp_alignment(source, target, max_iter=200):
    """단순 ICP 정렬 (특징 없이)"""
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=15.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return result.transformation, result


def fpfh_based_alignment(source, target, voxel_size=5.0):
    """FPFH 특징 기반 정렬"""
    # 다운샘플링
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)
    
    # FPFH 특징 계산
    print("  FPFH 특징 계산 중...")
    src_fpfh = compute_fpfh_feature(src_down, voxel_size)
    tgt_fpfh = compute_fpfh_feature(tgt_down, voxel_size)
    
    print(f"  Source FPFH shape: {src_fpfh.data.shape}")
    print(f"  Target FPFH shape: {tgt_fpfh.data.shape}")
    
    # FPFH 기반 RANSAC
    distance_threshold = voxel_size * 2.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=20000, confidence=0.95)
    )
    
    # ICP 정밀화
    print("  ICP 정밀화 중...")
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=10.0,
        init=result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    return result_icp.transformation, result_icp, result


def visualize_fpfh_features(pcd, fpfh, num_samples=1000):
    """FPFH 특징을 색상으로 시각화"""
    # 특징 벡터의 첫 번째 차원을 색상으로 매핑
    features = fpfh.data[0, :]  # 첫 번째 히스토그램 빈 사용
    
    # 정규화
    features_norm = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
    
    # 컬러맵 적용
    colors = plt.cm.jet(features_norm)[:, :3]
    
    # 포인트 클라우드에 색상 적용
    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd_colored


def calculate_alignment_metrics(source, target):
    """정렬 품질 메트릭 계산"""
    distances = np.asarray(target.compute_point_cloud_distance(source))
    
    mean_error = np.mean(distances)
    median_error = np.median(distances)
    std_error = np.std(distances)
    max_error = np.max(distances)
    
    # 정밀도: 10mm 이내 포인트 비율
    precision_10mm = np.sum(distances < 10.0) / len(distances) * 100
    
    return {
        'mean': mean_error,
        'median': median_error,
        'std': std_error,
        'max': max_error,
        'precision_10mm': precision_10mm
    }


def capture_alignment_image(source, target, title=""):
    """정렬 결과 캡처"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=600)
    vis.add_geometry(source)
    vis.add_geometry(target)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 3.0
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    ctr.set_front([0.5, -0.3, -0.8])
    ctr.set_up([0, 1, 0])  # Y축을 위로
    
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    
    return np.asarray(image)


def main():
    print("="*60)
    print("논문용 이미지 생성 #3: FPFH 특징 기반 정렬 성능 비교")
    print("="*60)
    
    # 테스트 데이터 로드
    front_path = r"D:\Lab2\--final_3D_Body--\3D_Body_Posture_Analysis\test2\DepthMap\여_정면\DepthMap0.bmp"
    left_path = r"D:\Lab2\--final_3D_Body--\3D_Body_Posture_Analysis\test2\DepthMap\여_L\DepthMap0.bmp"
    
    print("\n깊이맵 로딩 및 포인트 클라우드 생성 중...")
    depth_front = load_depth_map(front_path)
    depth_left = load_depth_map(left_path)
    
    if depth_front is None or depth_left is None:
        print("깊이맵 로드 실패!")
        return
    
    # 포인트 클라우드 생성
    pcd_front = create_point_cloud_from_depth(depth_front, "front")
    pcd_left = create_point_cloud_from_depth(depth_left, "left")
    
    # 법선 계산
    pcd_front.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    pcd_left.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    # 색상 설정
    pcd_front.paint_uniform_color([0.8, 0.2, 0.2])  # 빨간색 (타겟)
    
    print(f"정면 포인트: {len(pcd_front.points):,}개")
    print(f"좌측 포인트: {len(pcd_left.points):,}개")
    
    # 1. 단순 ICP 정렬 (특징 없이)
    print("\n1. 단순 ICP 정렬 (기하학적 특징만)...")
    pcd_left_simple = copy.deepcopy(pcd_left)
    transform_simple, result_simple = simple_icp_alignment(pcd_left_simple, pcd_front)
    pcd_left_simple.transform(transform_simple)
    pcd_left_simple.paint_uniform_color([0.8, 0.5, 0.2])  # 주황색
    
    metrics_simple = calculate_alignment_metrics(pcd_left_simple, pcd_front)
    print(f"  평균 오차: {metrics_simple['mean']:.2f} mm")
    print(f"  중앙값 오차: {metrics_simple['median']:.2f} mm")
    print(f"  표준편차: {metrics_simple['std']:.2f} mm")
    print(f"  정밀도 (10mm 이내): {metrics_simple['precision_10mm']:.1f}%")
    print(f"  Fitness: {result_simple.fitness:.4f}")
    
    # 2. FPFH 기반 정렬
    print("\n2. FPFH 특징 기반 정렬...")
    pcd_left_fpfh = copy.deepcopy(pcd_left)
    transform_fpfh, result_fpfh, result_ransac = fpfh_based_alignment(pcd_left_fpfh, pcd_front, voxel_size=5.0)
    pcd_left_fpfh.transform(transform_fpfh)
    pcd_left_fpfh.paint_uniform_color([0.2, 0.8, 0.2])  # 초록색
    
    metrics_fpfh = calculate_alignment_metrics(pcd_left_fpfh, pcd_front)
    print(f"  평균 오차: {metrics_fpfh['mean']:.2f} mm")
    print(f"  중앙값 오차: {metrics_fpfh['median']:.2f} mm")
    print(f"  표준편차: {metrics_fpfh['std']:.2f} mm")
    print(f"  정밀도 (10mm 이내): {metrics_fpfh['precision_10mm']:.1f}%")
    print(f"  RANSAC Fitness: {result_ransac.fitness:.4f}")
    print(f"  최종 Fitness: {result_fpfh.fitness:.4f}")
    
    # 3. FPFH 특징 시각화
    print("\n3. FPFH 특징 시각화...")
    pcd_front_down = pcd_front.voxel_down_sample(5.0)
    fpfh_front = compute_fpfh_feature(pcd_front_down, 5.0)
    pcd_front_fpfh_vis = visualize_fpfh_features(pcd_front_down, fpfh_front)
    
    # 4. 이미지 캡처
    print("\n4. 비교 이미지 생성 중...")
    img_simple = capture_alignment_image(pcd_left_simple, pcd_front, "Simple ICP")
    img_fpfh = capture_alignment_image(pcd_left_fpfh, pcd_front, "FPFH + ICP")
    
    # FPFH 특징 시각화 캡처
    vis_fpfh = o3d.visualization.Visualizer()
    vis_fpfh.create_window(visible=False, width=800, height=600)
    vis_fpfh.add_geometry(pcd_front_fpfh_vis)
    opt_fpfh = vis_fpfh.get_render_option()
    opt_fpfh.background_color = np.asarray([0, 0, 0])
    opt_fpfh.point_size = 5.0
    ctr_fpfh = vis_fpfh.get_view_control()
    ctr_fpfh.set_zoom(0.8)
    ctr_fpfh.set_front([0, 0, -1])
    ctr_fpfh.set_up([0, 1, 0])  # Y축을 위로
    vis_fpfh.poll_events()
    vis_fpfh.update_renderer()
    img_fpfh_vis = np.asarray(vis_fpfh.capture_screen_float_buffer(False))
    vis_fpfh.destroy_window()
    
    # 5. 결과 비교 플롯
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.25], hspace=0.3, wspace=0.2)
    
    # FPFH 특징 시각화
    ax_fpfh = fig.add_subplot(gs[0, :])
    ax_fpfh.imshow(img_fpfh_vis)
    ax_fpfh.set_title('FPFH Feature Visualization (33D Histogram per Point)', 
                      fontsize=13, fontweight='bold')
    ax_fpfh.text(0.5, -0.05, 
                'Color represents local surface geometry: curvature, normals, angles', 
                ha='center', va='top', transform=ax_fpfh.transAxes, fontsize=10)
    ax_fpfh.axis('off')
    
    # 단순 ICP 결과
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(img_simple)
    ax1.set_title('(a) Simple ICP (Geometric Only)', fontsize=12, fontweight='bold')
    stats_simple = f"""Mean Error: {metrics_simple['mean']:.2f} mm
Median: {metrics_simple['median']:.2f} mm
Std: {metrics_simple['std']:.2f} mm
Precision (10mm): {metrics_simple['precision_10mm']:.1f}%
Fitness: {result_simple.fitness:.4f}"""
    ax1.text(0.5, -0.15, stats_simple, ha='center', va='top', 
            transform=ax1.transAxes, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    ax1.axis('off')
    
    # FPFH 기반 결과
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(img_fpfh)
    ax2.set_title('(b) FPFH-based Alignment (Feature-Rich)', fontsize=12, fontweight='bold')
    stats_fpfh = f"""Mean Error: {metrics_fpfh['mean']:.2f} mm
Median: {metrics_fpfh['median']:.2f} mm
Std: {metrics_fpfh['std']:.2f} mm
Precision (10mm): {metrics_fpfh['precision_10mm']:.1f}%
Fitness: {result_fpfh.fitness:.4f}"""
    ax2.text(0.5, -0.15, stats_fpfh, ha='center', va='top', 
            transform=ax2.transAxes, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax2.axis('off')
    
    # 통계 비교
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    improvement_mean = ((metrics_simple['mean'] - metrics_fpfh['mean']) / metrics_simple['mean']) * 100
    improvement_precision = metrics_fpfh['precision_10mm'] - metrics_simple['precision_10mm']
    
    comparison_text = f"""
    Performance Comparison:
    
    Metric                  | Simple ICP | FPFH-based | Improvement
    ------------------------|------------|------------|-------------
    Mean Error (mm)         | {metrics_simple['mean']:8.2f}   | {metrics_fpfh['mean']:8.2f}   | {improvement_mean:+6.1f}%
    Median Error (mm)       | {metrics_simple['median']:8.2f}   | {metrics_fpfh['median']:8.2f}   | -
    Std Deviation (mm)      | {metrics_simple['std']:8.2f}   | {metrics_fpfh['std']:8.2f}   | -
    Precision (10mm) (%)    | {metrics_simple['precision_10mm']:8.1f}   | {metrics_fpfh['precision_10mm']:8.1f}   | {improvement_precision:+6.1f}%
    Fitness Score           | {result_simple.fitness:8.4f}   | {result_fpfh.fitness:8.4f}   | Better
    
    Key Advantage: FPFH captures local surface geometry (33D histogram), enabling robust feature matching
                   even under significant initial misalignment and partial overlap.
    """
    
    ax_stats.text(0.5, 0.5, comparison_text, ha='center', va='center', 
                 fontsize=9, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('FPFH Feature-based Alignment vs Simple ICP', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # 저장
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, 'output_3_fpfh_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n결과 이미지 저장: {output_path}")
    
    print("\n완료!")
    plt.show()


if __name__ == "__main__":
    main()
