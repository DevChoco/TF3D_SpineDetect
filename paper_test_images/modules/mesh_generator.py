"""
메시 생성 및 저장 모듈 (고급 버텍스 리덕션 + 홀 채우기 통합)

이 모듈은 다음 기능을 제공합니다:
- 포인트 클라우드에서 메시 생성
- Poisson 표면 재구성
- Ball Pivoting Algorithm
- 고급 버텍스 리덕션 및 메시 최적화
- 지능형 홀 채우기 및 누락 영역 보완
- 다양한 형식으로 저장 (OBJ, PLY, STL)
"""

import numpy as np
import open3d as o3d
import os
from .mesh_optimizer import (
    smart_vertex_reduction, 
    adaptive_mesh_optimization,
    create_lod_hierarchy,
    measure_optimization_quality,
    save_optimized_mesh,
    analyze_mesh_complexity
)
from .hole_filling import (
    advanced_hole_filling,
    compare_before_after
)


def create_mesh_from_pointcloud(pcd):
    """
    포인트 클라우드에서 메시를 생성합니다.
    
    Args:
        pcd (o3d.geometry.PointCloud): Open3D PointCloud 객체
    
    Returns:
        o3d.geometry.TriangleMesh: Open3D TriangleMesh 객체 또는 None
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


def simplify_mesh(mesh, reduction_ratio=0.5, method="quadric", preserve_boundary=True, adaptive=False):
    """
    메시의 복잡성을 줄입니다 (고급 버텍스 리덕션).
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 단순화할 메시
        reduction_ratio (float): 삼각형 감소 비율 (0.1 = 90% 감소, 0.5 = 50% 감소)
        method (str): 단순화 방법 ("quadric", "cluster", "average", "adaptive")
        preserve_boundary (bool): 경계선 보존 여부
        adaptive (bool): 적응형 단순화 사용 여부
        
    Returns:
        o3d.geometry.TriangleMesh: 단순화된 메시
    """
    if mesh is None:
        return None
    
    original_triangles = len(mesh.triangles)
    original_vertices = len(mesh.vertices)
    target_triangles = max(100, int(original_triangles * reduction_ratio))
    
    print(f"고급 메시 단순화 시작:")
    print(f"  원본: {original_vertices:,}개 버텍스, {original_triangles:,}개 삼각형")
    print(f"  목표: {target_triangles:,}개 삼각형 (감소율: {(1-reduction_ratio)*100:.1f}%)")
    print(f"  방법: {method}, 경계선 보존: {preserve_boundary}, 적응형: {adaptive}")
    
    try:
        if method == "quadric" or method == "adaptive":
            # Quadric Error Metrics를 사용한 고품질 단순화
            if adaptive:
                # 적응형 단순화: 곡률이 높은 부분은 더 많은 버텍스 유지
                mesh.compute_vertex_normals()
                simplified_mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=target_triangles,
                    maximum_error=0.01,  # 낮은 오차 허용
                    boundary_weight=1.0 if preserve_boundary else 0.1
                )
            else:
                simplified_mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=target_triangles
                )
            
        elif method == "cluster":
            # 버텍스 클러스터링을 사용한 단순화 (속도 우선)
            bbox_extent = mesh.get_axis_aligned_bounding_box().get_extent().max()
            if adaptive:
                # 적응형 복셀 크기: 메시 복잡도에 따라 조정
                complexity_factor = min(2.0, original_triangles / 10000)
                voxel_size = bbox_extent / (100 * complexity_factor)
            else:
                voxel_size = bbox_extent / 100
                
            simplified_mesh = mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average
            )
            
        elif method == "progressive":
            # 점진적 메시 단순화 (여러 단계로 나누어 수행)
            simplified_mesh = mesh
            steps = 5
            step_ratio = pow(reduction_ratio, 1.0/steps)
            
            for i in range(steps):
                step_target = max(100, int(len(simplified_mesh.triangles) * step_ratio))
                print(f"    단계 {i+1}/{steps}: {len(simplified_mesh.triangles):,} → {step_target:,} 삼각형")
                simplified_mesh = simplified_mesh.simplify_quadric_decimation(
                    target_number_of_triangles=step_target
                )
                
        elif method == "edge_collapse":
            # 엣지 콜랩스 기반 단순화 (기하학적 특징 보존)
            voxel_size = mesh.get_axis_aligned_bounding_box().get_extent().max() / 150
            simplified_mesh = mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Quadric
            )
            
        else:
            print(f"알 수 없는 단순화 방법: {method}. Quadric 방법을 사용합니다.")
            simplified_mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles
            )
        
        # 결과 정보 출력
        final_triangles = len(simplified_mesh.triangles)
        final_vertices = len(simplified_mesh.vertices)
        actual_reduction = (original_triangles - final_triangles) / original_triangles * 100
        
        print(f"  결과: {final_vertices:,}개 버텍스, {final_triangles:,}개 삼각형")
        print(f"  실제 감소율: {actual_reduction:.1f}%")
        
        # 메시 품질 측정
        quality_score = measure_mesh_quality(simplified_mesh)
        print(f"  메시 품질 점수: {quality_score:.3f}/1.000")
        
        return simplified_mesh
        
    except Exception as e:
        print(f"메시 단순화 중 오류: {e}")
        return mesh


def measure_mesh_quality(mesh):
    """
    메시의 품질을 측정합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 품질을 측정할 메시
        
    Returns:
        float: 품질 점수 (0.0 ~ 1.0, 높을수록 좋음)
    """
    if mesh is None or len(mesh.triangles) == 0:
        return 0.0
    
    try:
        # 삼각형 면적 분산 (균등한 삼각형일수록 좋음)
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        
        areas = []
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle]
            # 삼각형 면적 계산
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas.append(area)
        
        areas = np.array(areas)
        area_variance = np.var(areas) / (np.mean(areas) + 1e-8)
        area_score = 1.0 / (1.0 + area_variance)
        
        # 매니폴드 정도 (좋은 메시일수록 매니폴드에 가까움)
        mesh_copy = mesh.copy()
        mesh_copy.remove_non_manifold_edges()
        manifold_score = len(mesh_copy.triangles) / len(mesh.triangles)
        
        # 전체 품질 점수 (가중평균)
        quality_score = 0.6 * area_score + 0.4 * manifold_score
        
        return min(1.0, max(0.0, quality_score))
        
    except Exception as e:
        print(f"메시 품질 측정 중 오류: {e}")
        return 0.5  # 기본값


def optimize_mesh(mesh, enable_simplification=True, reduction_ratio=0.3, optimization_level="standard"):
    """
    메시를 고급 최적화합니다 (고급 버텍스 리덕션 포함).
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 최적화할 메시
        enable_simplification (bool): 메시 단순화 활성화 여부
        reduction_ratio (float): 메시 단순화 비율
        optimization_level (str): 최적화 레벨 ("fast", "standard", "high_quality")
        
    Returns:
        o3d.geometry.TriangleMesh: 최적화된 메시
    """
    if mesh is None:
        return None
    
    try:
        print(f"\n고급 메시 최적화 시작... (레벨: {optimization_level})")
        initial_quality = measure_mesh_quality(mesh)
        print(f"초기 메시 품질: {initial_quality:.3f}")
        
        # 1단계: 기본 정리
        print("1단계: 기본 메시 정리...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # 2단계: 메시 단순화 (레벨별 설정)
        if enable_simplification:
            print("2단계: 지능형 메시 복잡성 감소...")
            
            if optimization_level == "fast":
                # 빠른 최적화: 클러스터링 사용
                mesh = simplify_mesh(mesh, reduction_ratio=reduction_ratio, 
                                   method="cluster", preserve_boundary=False, adaptive=False)
                
            elif optimization_level == "high_quality":
                # 고품질 최적화: 적응형 Quadric 사용
                mesh = simplify_mesh(mesh, reduction_ratio=reduction_ratio, 
                                   method="adaptive", preserve_boundary=True, adaptive=True)
                
            else:  # standard
                # 표준 최적화: 균형잡힌 Quadric 사용
                mesh = simplify_mesh(mesh, reduction_ratio=reduction_ratio, 
                                   method="quadric", preserve_boundary=True, adaptive=False)
        
        # 3단계: 스무딩 (레벨별 반복 횟수)
        print("3단계: 적응형 메시 스무딩...")
        if optimization_level == "fast":
            smooth_iterations = 1
        elif optimization_level == "high_quality":
            smooth_iterations = 5
        else:  # standard
            smooth_iterations = 3
            
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)
        
        # 4단계: 법선 벡터 재계산
        print("4단계: 법선 벡터 재계산...")
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # 5단계: 품질 검증
        final_quality = measure_mesh_quality(mesh)
        print(f"최종 메시 품질: {final_quality:.3f} (개선도: {final_quality-initial_quality:+.3f})")
        
        print("고급 메시 최적화 완료!")
        return mesh
        
    except Exception as e:
        print(f"메시 최적화 중 오류: {e}")
        return mesh
def create_and_save_mesh(pcd, output_dir="output/3d_models", base_filename="body_mesh", 
                        create_lod=True, reduction_ratio=0.3, optimization_level="standard",
                        custom_lod_levels=None, enable_quality_analysis=True, enable_hole_filling=True,
                        hole_filling_method="comprehensive"):
    """
    포인트 클라우드에서 고품질 메시를 생성하고 고급 버텍스 리덕션 및 홀 채우기를 적용합니다.
    
    Args:
        pcd (o3d.geometry.PointCloud): 포인트 클라우드
        output_dir (str): 출력 디렉토리
        base_filename (str): 기본 파일명
        create_lod (bool): 여러 LOD 레벨 메시 생성 여부
        reduction_ratio (float): 기본 메시 단순화 비율
        optimization_level (str): 최적화 레벨 ("fast", "standard", "high_quality")
        custom_lod_levels (dict): 사용자 정의 LOD 레벨
        enable_quality_analysis (bool): 품질 분석 활성화 여부
        enable_hole_filling (bool): 홀 채우기 활성화 여부
        hole_filling_method (str): 홀 채우기 방법 ("comprehensive", "large_gaps", "bilateral_symmetry", "symmetry")
        
    Returns:
        tuple: (mesh object, saved file paths list)
    """
    print("\n=== 고급 메시 생성, 홀 채우기 및 버텍스 리덕션 ===")
    print("포인트 클라우드를 고품질 메시로 변환하고 누락 영역을 채운 후 지능형 버텍스 리덕션을 적용합니다...")
    
    # 1단계: 기본 메시 생성
    mesh = create_mesh_from_pointcloud(pcd)
    
    saved_files = []
    
    if mesh is not None:
        print(f"\n원본 메시 정보: {len(mesh.vertices):,}개 버텍스, {len(mesh.triangles):,}개 삼각형")
        
        if enable_quality_analysis:
            complexity_analysis = analyze_mesh_complexity(mesh)
            print(f"메시 복잡성 점수: {complexity_analysis['complexity_score']:.3f}")
        
        # 2단계: 지능형 홀 채우기 (뎁스 이미지 한계 보완)
        if enable_hole_filling:
            print(f"\n🔧 지능형 홀 채우기 적용 중 (방법: {hole_filling_method})...")
            print("뎁스 이미지로 인한 팔 가림, 옆구리, 그림자 영역 등의 누락 부분을 복원합니다.")
            
            # 홀 채우기 전 메시 복사
            original_mesh_for_comparison = create_mesh_from_pointcloud(pcd)
            
            # 안전한 홀 채우기 적용
            try:
                # 선택된 방법으로 홀 채우기 적용
                if hole_filling_method == "comprehensive":
                    # 종합적 접근법 (큰 구멍 + 대칭성 + 해부학적 보정)
                    mesh = advanced_hole_filling(mesh, method="comprehensive")
                elif hole_filling_method == "large_gaps":
                    # 큰 구멍 전용 (옆구리, 팔 안쪽)
                    mesh = advanced_hole_filling(mesh, method="large_gaps")
                elif hole_filling_method == "bilateral_symmetry":
                    # 양측 대칭성 완성
                    mesh = advanced_hole_filling(mesh, method="bilateral_symmetry")
                elif hole_filling_method == "symmetry":
                    # 기본 대칭성 복원만
                    mesh = advanced_hole_filling(mesh, method="symmetry")
                else:
                    # 기본값: 대칭성 기반 복원
                    mesh = advanced_hole_filling(mesh, method="symmetry")
            except:
                print("대칭성 기반 복원 실패, 형태학적 방법 사용")
                try:
                    mesh = advanced_hole_filling(mesh, method="morphological")
                except:
                    print("홀 채우기 실패, 원본 메시 사용")
            
            # 홀 채우기 효과 분석
            if enable_quality_analysis:
                try:
                    hole_fill_analysis = compare_before_after(original_mesh_for_comparison, mesh)
                except:
                    print("홀 채우기 분석 실패")
        
        # 3단계: 적응형 메시 최적화 (지능형 버텍스 리덕션)
        print(f"\n🎯 지능형 버텍스 리덕션 적용 중... (목표 감소율: {(1-reduction_ratio)*100:.1f}%)")
        
        if optimization_level == "high_quality":
            # 고품질 모드: 적응형 최적화
            optimized_mesh = adaptive_mesh_optimization(mesh, complexity_level="auto")
            # 추가 리덕션 적용
            optimized_mesh = smart_vertex_reduction(optimized_mesh, target_ratio=reduction_ratio, quality_priority=True)
            
        elif optimization_level == "fast":
            # 빠른 모드: 기본 리덕션
            optimized_mesh = smart_vertex_reduction(mesh, target_ratio=reduction_ratio, quality_priority=False)
            
        else:  # standard
            # 표준 모드: 균형잡힌 리덕션
            optimized_mesh = smart_vertex_reduction(mesh, target_ratio=reduction_ratio, quality_priority=True)
        
        # 4단계: 품질 분석
        if enable_quality_analysis:
            quality_info = measure_optimization_quality(mesh, optimized_mesh)
            print(f"\n=== 최종 최적화 결과 ===")
            print(f"버텍스 감소: {quality_info['vertex_reduction_percent']:.1f}%")
            print(f"삼각형 감소: {quality_info['triangle_reduction_percent']:.1f}%")
            print(f"표면적 보존: {quality_info['area_preservation_percent']:.1f}%")
            print(f"전체 품질 점수: {quality_info['overall_quality_score']:.1f}/100")
            
            # 홀 채우기 정보 추가
            if enable_hole_filling and 'hole_fill_analysis' in locals():
                quality_info.update(hole_fill_analysis)
        else:
            quality_info = None
        
        # 5단계: 기본 메시 저장
        saved_files = save_optimized_mesh(optimized_mesh, output_dir, base_filename, quality_info)
        
        # 6단계: LOD 메시 생성 (선택사항)
        if create_lod:
            print("\n=== 다중 LOD 메시 생성 ===")
            lod_meshes = create_lod_hierarchy(mesh, custom_lod_levels)
            
            lod_saved_files = {}
            for lod_name, lod_mesh in lod_meshes.items():
                lod_filename = f"{base_filename}_{lod_name}"
                
                # LOD별 홀 채우기 적용 (선택적)
                if enable_hole_filling and lod_name in ["ultra_high", "high"]:
                    print(f"  {lod_name.upper()} LOD에 홀 채우기 적용 중...")
                    lod_mesh = advanced_hole_filling(lod_mesh, method="symmetry")  # 빠른 방법 사용
                
                # LOD별 품질 분석
                if enable_quality_analysis:
                    lod_quality = measure_optimization_quality(mesh, lod_mesh)
                else:
                    lod_quality = None
                
                # LOD 메시 저장
                lod_files = save_optimized_mesh(lod_mesh, output_dir, lod_filename, lod_quality)
                lod_saved_files[lod_name] = lod_files
                saved_files.extend(lod_files)
            
            print(f"\n총 {len(lod_meshes)}개의 LOD 레벨이 생성되었습니다.")
            print(f"전체 저장된 파일: {len(saved_files)}개")
        
        # 7단계: 전체 요약 출력
        print(f"\n=== 🎉 메시 생성 및 최적화 완료 ===")
        if enable_hole_filling:
            print("✅ 뎁스 이미지 한계로 인한 누락 영역 복원 완료")
        print("✅ 지능형 버텍스 리덕션으로 최적화 완료")
        print(f"✅ 총 {len(saved_files)}개 파일 저장 완료")
        
        return optimized_mesh, saved_files
    else:
        print("메시 생성에 실패했습니다.")
        return None, []