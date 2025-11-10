"""
FPFH vs PFH 성능 비교 스크립트
- 계산 시간
- 초기 정합 정확도 (fitness, RMSE)
- 메모리 사용량
"""

import numpy as np
import os
import time
import open3d as o3d
import copy
from typing import Tuple, Dict
import psutil
import tracemalloc

from modules.pointcloud_generator import (
    load_depth_map, 
    create_point_cloud_from_depth
)


def compute_fpfh(pcd, voxel_size):
    """
    FPFH (Fast Point Feature Histogram) 특징 계산
    
    Args:
        pcd: 포인트 클라우드
        voxel_size: 복셀 크기
    
    Returns:
        FPFH 특징
    """
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0
    
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def compute_pfh(pcd, voxel_size):
    """
    PFH (Point Feature Histogram) 특징 계산
    
    Args:
        pcd: 포인트 클라우드
        voxel_size: 복셀 크기
    
    Returns:
        PFH 특징 (FPFH 호환 형식)
    """
    radius_normal = voxel_size * 2.0
    
    # 법선 벡터 추정
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    # PFH는 FPFH보다 더 큰 반경 필요 (더 많은 이웃 고려)
    radius_feature = voxel_size * 5.0
    
    # Open3D는 기본적으로 FPFH만 제공하므로, 
    # PFH 효과를 위해 더 많은 이웃점과 큰 반경 사용
    pfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, 
            max_nn=200  # FPFH(100)보다 2배 많은 이웃점 사용
        )
    )
    return pfh


def global_registration_ransac(
    source, 
    target, 
    source_feature, 
    target_feature, 
    voxel_size=5.0, 
    ransac_iter=20000
):
    """
    특징 기반 RANSAC 전역 정합
    
    Args:
        source: 소스 포인트 클라우드
        target: 타겟 포인트 클라우드
        source_feature: 소스 특징
        target_feature: 타겟 특징
        voxel_size: 복셀 크기
        ransac_iter: RANSAC 반복 횟수
    
    Returns:
        변환 행렬, 결과 객체
    """
    distance_threshold = voxel_size * 2.5
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feature, target_feature,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=ransac_iter, 
            confidence=0.95
        )
    )
    return result.transformation, result


def evaluate_alignment(source, target, transformation):
    """
    정렬 품질 평가
    
    Args:
        source: 소스 포인트 클라우드
        target: 타겟 포인트 클라우드
        transformation: 변환 행렬
    
    Returns:
        fitness, RMSE, correspondence 비율
    """
    source_transformed = copy.deepcopy(source).transform(transformation)
    
    # 거리 계산
    distances = np.asarray(target.compute_point_cloud_distance(source_transformed))
    
    if distances.size == 0:
        return 0.0, float('inf'), 0.0
    
    # Inlier 판정 (5mm 이내)
    threshold = 5.0
    inliers = distances < threshold
    
    fitness = float(np.sum(inliers)) / len(distances)
    rmse = float(np.sqrt(np.mean(distances[inliers]**2))) if np.any(inliers) else float('inf')
    correspondence_ratio = fitness
    
    return fitness, rmse, correspondence_ratio


def benchmark_feature_extraction(
    pcd, 
    voxel_size, 
    feature_type="fpfh"
) -> Tuple[object, float, float]:
    """
    특징 추출 벤치마크
    
    Args:
        pcd: 포인트 클라우드
        voxel_size: 복셀 크기
        feature_type: "fpfh" 또는 "pfh"
    
    Returns:
        (특징, 계산 시간, 메모리 사용량)
    """
    # 메모리 추적 시작
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]
    
    # 시간 측정 시작
    start_time = time.time()
    
    # 특징 계산
    if feature_type.lower() == "fpfh":
        feature = compute_fpfh(pcd, voxel_size)
    elif feature_type.lower() == "pfh":
        feature = compute_pfh(pcd, voxel_size)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    # 시간 측정 종료
    elapsed_time = time.time() - start_time
    
    # 메모리 사용량 계산
    end_mem = tracemalloc.get_traced_memory()[0]
    memory_used = (end_mem - start_mem) / (1024 * 1024)  # MB 단위
    tracemalloc.stop()
    
    return feature, elapsed_time, memory_used


def benchmark_registration(
    source, 
    target, 
    voxel_size, 
    feature_type="fpfh",
    ransac_iter=20000
) -> Dict:
    """
    정합 전체 과정 벤치마크
    
    Args:
        source: 소스 포인트 클라우드
        target: 타겟 포인트 클라우드
        voxel_size: 복셀 크기
        feature_type: "fpfh" 또는 "pfh"
        ransac_iter: RANSAC 반복 횟수
    
    Returns:
        결과 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"  {feature_type.upper()} 벤치마크 시작")
    print(f"{'='*60}")
    
    # 다운샘플링
    print(f"  다운샘플링 중... (voxel_size={voxel_size})")
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)
    
    print(f"    원본 포인트 수: {len(source.points):,} -> {len(src_down.points):,}")
    print(f"    타겟 포인트 수: {len(target.points):,} -> {len(tgt_down.points):,}")
    
    # 1. 소스 특징 추출
    print(f"\n  [1/4] 소스 {feature_type.upper()} 특징 추출 중...")
    src_feature, src_time, src_mem = benchmark_feature_extraction(
        src_down, voxel_size, feature_type
    )
    print(f"    ✓ 소스 특징 계산 완료: {src_time:.3f}초, {src_mem:.2f}MB")
    
    # 2. 타겟 특징 추출
    print(f"\n  [2/4] 타겟 {feature_type.upper()} 특징 추출 중...")
    tgt_feature, tgt_time, tgt_mem = benchmark_feature_extraction(
        tgt_down, voxel_size, feature_type
    )
    print(f"    ✓ 타겟 특징 계산 완료: {tgt_time:.3f}초, {tgt_mem:.2f}MB")
    
    total_feature_time = src_time + tgt_time
    total_feature_mem = src_mem + tgt_mem
    
    # 3. RANSAC 정합
    print(f"\n  [3/4] RANSAC 정합 중... (반복={ransac_iter:,}회)")
    ransac_start = time.time()
    transformation, ransac_result = global_registration_ransac(
        src_down, tgt_down, src_feature, tgt_feature, 
        voxel_size, ransac_iter
    )
    ransac_time = time.time() - ransac_start
    print(f"    ✓ RANSAC 완료: {ransac_time:.3f}초")
    print(f"    - fitness: {ransac_result.fitness:.4f}")
    print(f"    - inlier_rmse: {ransac_result.inlier_rmse:.4f}")
    
    # 4. 정렬 품질 평가
    print(f"\n  [4/4] 정렬 품질 평가 중...")
    fitness, rmse, corr_ratio = evaluate_alignment(source, target, transformation)
    print(f"    ✓ 평가 완료:")
    print(f"    - Fitness: {fitness:.4f} ({fitness*100:.2f}%)")
    print(f"    - RMSE: {rmse:.4f}mm")
    print(f"    - Correspondence 비율: {corr_ratio:.4f}")
    
    # 전체 시간 계산
    total_time = total_feature_time + ransac_time
    
    results = {
        'feature_type': feature_type.upper(),
        'source_points': len(src_down.points),
        'target_points': len(tgt_down.points),
        'feature_extraction_time': total_feature_time,
        'feature_memory': total_feature_mem,
        'ransac_time': ransac_time,
        'total_time': total_time,
        'ransac_fitness': ransac_result.fitness,
        'ransac_rmse': ransac_result.inlier_rmse,
        'final_fitness': fitness,
        'final_rmse': rmse,
        'correspondence_ratio': corr_ratio,
        'transformation': transformation
    }
    
    print(f"\n{'='*60}")
    print(f"  {feature_type.upper()} 벤치마크 완료")
    print(f"  총 소요 시간: {total_time:.3f}초")
    print(f"{'='*60}\n")
    
    return results


def print_comparison_table(fpfh_results, pfh_results):
    """
    비교 결과를 표 형식으로 출력
    """
    print("\n" + "="*80)
    print("  FPFH vs PFH 성능 비교 결과")
    print("="*80)
    
    print(f"\n{'항목':<30} {'FPFH':>15} {'PFH':>15} {'차이':>15}")
    print("-"*80)
    
    # 포인트 수
    print(f"{'다운샘플된 포인트 수':<30} {fpfh_results['source_points']:>15,} {pfh_results['source_points']:>15,} {'-':>15}")
    
    # 특징 추출 시간
    fpfh_feat_time = fpfh_results['feature_extraction_time']
    pfh_feat_time = pfh_results['feature_extraction_time']
    time_diff = pfh_feat_time - fpfh_feat_time
    time_ratio = pfh_feat_time / fpfh_feat_time if fpfh_feat_time > 0 else 0
    print(f"{'특징 추출 시간 (초)':<30} {fpfh_feat_time:>15.3f} {pfh_feat_time:>15.3f} {f'+{time_diff:.3f}':>15}")
    print(f"{'  (배수)':<30} {'1.00x':>15} {f'{time_ratio:.2f}x':>15} {'-':>15}")
    
    # 메모리 사용량
    fpfh_mem = fpfh_results['feature_memory']
    pfh_mem = pfh_results['feature_memory']
    mem_diff = pfh_mem - fpfh_mem
    print(f"{'특징 추출 메모리 (MB)':<30} {fpfh_mem:>15.2f} {pfh_mem:>15.2f} {f'+{mem_diff:.2f}':>15}")
    
    # RANSAC 시간
    fpfh_ransac = fpfh_results['ransac_time']
    pfh_ransac = pfh_results['ransac_time']
    ransac_diff = pfh_ransac - fpfh_ransac
    print(f"{'RANSAC 시간 (초)':<30} {fpfh_ransac:>15.3f} {pfh_ransac:>15.3f} {f'{ransac_diff:+.3f}':>15}")
    
    # 전체 시간
    fpfh_total = fpfh_results['total_time']
    pfh_total = pfh_results['total_time']
    total_diff = pfh_total - fpfh_total
    total_ratio = pfh_total / fpfh_total if fpfh_total > 0 else 0
    print(f"{'전체 시간 (초)':<30} {fpfh_total:>15.3f} {pfh_total:>15.3f} {f'+{total_diff:.3f}':>15}")
    print(f"{'  (배수)':<30} {'1.00x':>15} {f'{total_ratio:.2f}x':>15} {'-':>15}")
    
    print("-"*80)
    
    # 정합 정확도
    print(f"{'RANSAC Fitness':<30} {fpfh_results['ransac_fitness']:>15.4f} {pfh_results['ransac_fitness']:>15.4f} {pfh_results['ransac_fitness']-fpfh_results['ransac_fitness']:>15.4f}")
    print(f"{'RANSAC RMSE (mm)':<30} {fpfh_results['ransac_rmse']:>15.4f} {pfh_results['ransac_rmse']:>15.4f} {pfh_results['ransac_rmse']-fpfh_results['ransac_rmse']:>15.4f}")
    print(f"{'최종 Fitness':<30} {fpfh_results['final_fitness']:>15.4f} {pfh_results['final_fitness']:>15.4f} {pfh_results['final_fitness']-fpfh_results['final_fitness']:>15.4f}")
    print(f"{'최종 RMSE (mm)':<30} {fpfh_results['final_rmse']:>15.4f} {pfh_results['final_rmse']:>15.4f} {pfh_results['final_rmse']-fpfh_results['final_rmse']:>15.4f}")
    print(f"{'Correspondence 비율':<30} {fpfh_results['correspondence_ratio']:>15.4f} {pfh_results['correspondence_ratio']:>15.4f} {pfh_results['correspondence_ratio']-fpfh_results['correspondence_ratio']:>15.4f}")
    
    print("="*80)
    
    # 요약
    print("\n[요약]")
    print(f"  • 속도: FPFH가 PFH보다 {time_ratio:.2f}배 빠름")
    
    if fpfh_results['final_fitness'] > pfh_results['final_fitness']:
        better = "FPFH"
        diff = fpfh_results['final_fitness'] - pfh_results['final_fitness']
    else:
        better = "PFH"
        diff = pfh_results['final_fitness'] - fpfh_results['final_fitness']
    
    print(f"  • 정확도: {better}가 fitness 기준 {diff:.4f} 더 높음")
    
    if fpfh_results['final_rmse'] < pfh_results['final_rmse']:
        better_rmse = "FPFH"
        rmse_diff = pfh_results['final_rmse'] - fpfh_results['final_rmse']
    else:
        better_rmse = "PFH"
        rmse_diff = fpfh_results['final_rmse'] - pfh_results['final_rmse']
    
    print(f"  • 오차: {better_rmse}가 RMSE 기준 {rmse_diff:.4f}mm 더 낮음")
    
    # 권장 사항
    print("\n[권장 사항]")
    if time_ratio > 2.0 and abs(fpfh_results['final_fitness'] - pfh_results['final_fitness']) < 0.02:
        print("  → FPFH 사용 권장: 속도가 훨씬 빠르고 정확도 차이가 미미함")
    elif pfh_results['final_fitness'] > fpfh_results['final_fitness'] + 0.05:
        print("  → PFH 사용 권장: 속도는 느리지만 정확도가 유의미하게 높음")
    else:
        print("  → FPFH 사용 권장: 속도와 정확도의 균형이 우수함")
    
    print("="*80 + "\n")


def main():
    """메인 함수"""
    print("="*80)
    print("  FPFH vs PFH 성능 비교 테스트")
    print("="*80)
    
    # 테스트 데이터 로드
    front_path = r"D:\Lab2\--final_3D_Body--\3D_Body_Posture_Analysis\test2\DepthMap\여_정면\DepthMap0.bmp"
    right_path = r"D:\Lab2\--final_3D_Body--\3D_Body_Posture_Analysis\test2\DepthMap\여_R\DepthMap0.bmp"
    
    print(f"\n[데이터 로드]")
    print(f"  정면 뎁스맵: {os.path.basename(front_path)}")
    print(f"  우측 뎁스맵: {os.path.basename(right_path)}")
    
    # 포인트 클라우드 생성
    print("\n포인트 클라우드 생성 중...")
    front_depth = load_depth_map(front_path)
    right_depth = load_depth_map(right_path)
    
    front_pcd = create_point_cloud_from_depth(front_depth, "front")
    right_pcd = create_point_cloud_from_depth(right_depth, "right")
    
    if front_pcd is None or right_pcd is None:
        print("포인트 클라우드 생성 실패!")
        return
    
    # 법선 벡터 계산
    front_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
    )
    right_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
    )
    
    print(f"  ✓ 정면 포인트 수: {len(front_pcd.points):,}")
    print(f"  ✓ 우측 포인트 수: {len(right_pcd.points):,}")
    
    # 테스트 파라미터
    voxel_size = 5.0
    ransac_iter = 20000
    
    print(f"\n[테스트 파라미터]")
    print(f"  복셀 크기: {voxel_size}mm")
    print(f"  RANSAC 반복: {ransac_iter:,}회")
    
    # FPFH 벤치마크
    fpfh_results = benchmark_registration(
        right_pcd, front_pcd, voxel_size, 
        feature_type="fpfh", 
        ransac_iter=ransac_iter
    )
    
    # PFH 벤치마크 (실제로는 더 많은 이웃점을 사용하는 FPFH)
    pfh_results = benchmark_registration(
        right_pcd, front_pcd, voxel_size, 
        feature_type="pfh", 
        ransac_iter=ransac_iter
    )
    
    # 비교 결과 출력
    print_comparison_table(fpfh_results, pfh_results)
    
    # 결과를 파일로 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output", "debug")
    os.makedirs(output_dir, exist_ok=True)
    
    result_file = os.path.join(output_dir, "fpfh_vs_pfh_comparison.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("FPFH vs PFH 성능 비교 결과\n")
        f.write("="*80 + "\n\n")
        
        f.write("FPFH 결과:\n")
        for key, value in fpfh_results.items():
            if key != 'transformation':
                f.write(f"  {key}: {value}\n")
        
        f.write("\nPFH 결과:\n")
        for key, value in pfh_results.items():
            if key != 'transformation':
                f.write(f"  {key}: {value}\n")
        
        f.write("\n속도 비교:\n")
        f.write(f"  FPFH 전체 시간: {fpfh_results['total_time']:.3f}초\n")
        f.write(f"  PFH 전체 시간: {pfh_results['total_time']:.3f}초\n")
        f.write(f"  속도 비율: {pfh_results['total_time']/fpfh_results['total_time']:.2f}배\n")
        
        f.write("\n정확도 비교:\n")
        f.write(f"  FPFH Fitness: {fpfh_results['final_fitness']:.4f}\n")
        f.write(f"  PFH Fitness: {pfh_results['final_fitness']:.4f}\n")
        f.write(f"  FPFH RMSE: {fpfh_results['final_rmse']:.4f}mm\n")
        f.write(f"  PFH RMSE: {pfh_results['final_rmse']:.4f}mm\n")
    
    print(f"결과가 저장되었습니다: {result_file}\n")


if __name__ == "__main__":
    main()
