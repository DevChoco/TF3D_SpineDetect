"""
고급 메시 최적화 및 버텍스 리덕션 모듈

이 모듈은 다음 기능을 제공합니다:
- 지능형 버텍스 리덕션 (Vertex Reduction)
- 적응형 메시 단순화 (Adaptive Mesh Simplification)
- 메시 품질 분석 및 최적화
- 다중 LOD (Level of Detail) 생성
- 고급 메시 후처리
"""

import numpy as np
import open3d as o3d
import os
import copy


def analyze_mesh_complexity(mesh):
    """
    메시의 복잡성을 분석합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 분석할 메시
        
    Returns:
        dict: 복잡성 분석 결과
    """
    if mesh is None:
        return {}
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    analysis = {
        'vertex_count': len(vertices),
        'triangle_count': len(triangles),
        'edge_count': len(mesh.get_non_manifold_edges()),
        'surface_area': mesh.get_surface_area(),
        'volume': mesh.get_volume() if mesh.is_watertight() else 0.0,
        'is_watertight': mesh.is_watertight(),
        'is_manifold': len(mesh.get_non_manifold_edges()) == 0
    }
    
    # 복잡성 점수 계산 (0.0 ~ 1.0)
    vertex_complexity = min(1.0, len(vertices) / 100000)  # 10만 버텍스를 기준으로
    triangle_complexity = min(1.0, len(triangles) / 200000)  # 20만 삼각형을 기준으로
    analysis['complexity_score'] = (vertex_complexity + triangle_complexity) / 2
    
    return analysis


def smart_vertex_reduction(mesh, target_ratio=0.5, quality_priority=True):
    """
    지능형 버텍스 리덕션을 수행합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        target_ratio (float): 목표 버텍스 비율 (0.0 ~ 1.0)
        quality_priority (bool): 품질 우선 모드 여부
        
    Returns:
        o3d.geometry.TriangleMesh: 최적화된 메시
    """
    if mesh is None:
        return None
    
    print(f"\n=== 지능형 버텍스 리덕션 ===")
    
    original_vertices = len(mesh.vertices)
    original_triangles = len(mesh.triangles)
    target_triangles = max(100, int(original_triangles * target_ratio))
    
    print(f"원본: {original_vertices:,}개 버텍스, {original_triangles:,}개 삼각형")
    print(f"목표: {target_triangles:,}개 삼각형 ({(1-target_ratio)*100:.1f}% 감소)")
    print(f"모드: {'품질 우선' if quality_priority else '속도 우선'}")
    
    try:
        # 메시 전처리
        mesh_clean = copy.deepcopy(mesh)
        mesh_clean.remove_degenerate_triangles()
        mesh_clean.remove_duplicated_triangles()
        mesh_clean.remove_duplicated_vertices()
        mesh_clean.remove_non_manifold_edges()
        
        # 법선 벡터 계산 (Quadric 알고리즘에 필요)
        mesh_clean.compute_vertex_normals()
        mesh_clean.compute_triangle_normals()
        
        if quality_priority:
            # 품질 우선: Quadric Error Metrics 사용
            print("Quadric Error Decimation 적용 중...")
            simplified_mesh = mesh_clean.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles,
                maximum_error=0.01,  # 낮은 오차 허용
                boundary_weight=1.0  # 경계 보존
            )
        else:
            # 속도 우선: 점진적 단순화
            print("점진적 단순화 적용 중...")
            simplified_mesh = progressive_simplification(mesh_clean, target_ratio)
        
        # 결과 분석
        final_vertices = len(simplified_mesh.vertices)
        final_triangles = len(simplified_mesh.triangles)
        
        vertex_reduction = (original_vertices - final_vertices) / original_vertices * 100
        triangle_reduction = (original_triangles - final_triangles) / original_triangles * 100
        
        print(f"결과: {final_vertices:,}개 버텍스, {final_triangles:,}개 삼각형")
        print(f"실제 감소율: 버텍스 {vertex_reduction:.1f}%, 삼각형 {triangle_reduction:.1f}%")
        
        return simplified_mesh
        
    except Exception as e:
        print(f"버텍스 리덕션 중 오류: {e}")
        return mesh


def progressive_simplification(mesh, target_ratio, steps=5):
    """
    점진적 메시 단순화를 수행합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        target_ratio (float): 최종 목표 비율
        steps (int): 단순화 단계 수
        
    Returns:
        o3d.geometry.TriangleMesh: 단순화된 메시
    """
    current_mesh = copy.deepcopy(mesh)
    step_ratio = pow(target_ratio, 1.0 / steps)
    
    for i in range(steps):
        current_triangles = len(current_mesh.triangles)
        step_target = max(100, int(current_triangles * step_ratio))
        
        print(f"  단계 {i+1}/{steps}: {current_triangles:,} → {step_target:,} 삼각형")
        
        current_mesh = current_mesh.simplify_quadric_decimation(
            target_number_of_triangles=step_target
        )
        
        # 중간 정리
        current_mesh.remove_degenerate_triangles()
        current_mesh.remove_duplicated_vertices()
    
    return current_mesh


def adaptive_mesh_optimization(mesh, complexity_level="auto"):
    """
    메시 복잡성에 따른 적응형 최적화를 수행합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        complexity_level (str): 복잡성 레벨 ("low", "medium", "high", "auto")
        
    Returns:
        o3d.geometry.TriangleMesh: 최적화된 메시
    """
    if mesh is None:
        return None
    
    analysis = analyze_mesh_complexity(mesh)
    
    if complexity_level == "auto":
        # 자동 복잡성 레벨 결정
        complexity_score = analysis['complexity_score']
        if complexity_score > 0.7:
            complexity_level = "high"
        elif complexity_score > 0.3:
            complexity_level = "medium"
        else:
            complexity_level = "low"
    
    print(f"\n=== 적응형 메시 최적화 ===")
    print(f"복잡성 레벨: {complexity_level}")
    print(f"복잡성 점수: {analysis['complexity_score']:.3f}")
    
    # 복잡성 레벨에 따른 최적화 전략
    if complexity_level == "high":
        # 고복잡성: 강력한 리덕션
        optimized_mesh = smart_vertex_reduction(mesh, target_ratio=0.3, quality_priority=True)
        optimized_mesh = optimized_mesh.filter_smooth_simple(number_of_iterations=3)
        
    elif complexity_level == "medium":
        # 중복잡성: 균형잡힌 리덕션
        optimized_mesh = smart_vertex_reduction(mesh, target_ratio=0.5, quality_priority=True)
        optimized_mesh = optimized_mesh.filter_smooth_simple(number_of_iterations=2)
        
    else:  # low
        # 저복잡성: 최소 리덕션
        optimized_mesh = smart_vertex_reduction(mesh, target_ratio=0.8, quality_priority=False)
        optimized_mesh = optimized_mesh.filter_smooth_simple(number_of_iterations=1)
    
    # 최종 정리
    optimized_mesh.remove_degenerate_triangles()
    optimized_mesh.remove_duplicated_triangles()
    optimized_mesh.remove_duplicated_vertices()
    optimized_mesh.compute_vertex_normals()
    optimized_mesh.compute_triangle_normals()
    
    return optimized_mesh


def create_lod_hierarchy(mesh, lod_levels=None):
    """
    계층적 LOD (Level of Detail) 메시를 생성합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 원본 메시
        lod_levels (dict): LOD 레벨 정의 (None이면 자동 생성)
        
    Returns:
        dict: LOD 레벨별 메시 딕셔너리
    """
    if mesh is None:
        return {}
    
    analysis = analyze_mesh_complexity(mesh)
    
    if lod_levels is None:
        # 메시 복잡성에 따른 자동 LOD 레벨 생성
        if analysis['triangle_count'] > 100000:
            lod_levels = {
                "ultra_high": 1.0,
                "high": 0.6,
                "medium": 0.3,
                "low": 0.1,
                "ultra_low": 0.05
            }
        elif analysis['triangle_count'] > 50000:
            lod_levels = {
                "high": 1.0,
                "medium": 0.5,
                "low": 0.2,
                "ultra_low": 0.08
            }
        else:
            lod_levels = {
                "high": 1.0,
                "medium": 0.6,
                "low": 0.3
            }
    
    print(f"\n=== LOD 계층 생성 ===")
    print(f"원본 메시: {analysis['triangle_count']:,}개 삼각형")
    
    lod_meshes = {}
    
    for lod_name, ratio in lod_levels.items():
        print(f"\n{lod_name.upper()} LOD 생성 중... (비율: {ratio*100:.0f}%)")
        
        if ratio >= 0.95:
            # 거의 원본: 기본 정리만
            lod_mesh = copy.deepcopy(mesh)
            lod_mesh.remove_degenerate_triangles()
            lod_mesh.remove_duplicated_vertices()
        else:
            # 단순화 적용
            lod_mesh = smart_vertex_reduction(mesh, target_ratio=ratio, 
                                            quality_priority=(ratio > 0.3))
        
        # 법선 벡터 재계산
        lod_mesh.compute_vertex_normals()
        lod_mesh.compute_triangle_normals()
        
        lod_meshes[lod_name] = lod_mesh
        
        print(f"  생성 완료: {len(lod_mesh.vertices):,}개 버텍스, {len(lod_mesh.triangles):,}개 삼각형")
    
    return lod_meshes


def measure_optimization_quality(original_mesh, optimized_mesh):
    """
    최적화 품질을 측정합니다.
    
    Args:
        original_mesh (o3d.geometry.TriangleMesh): 원본 메시
        optimized_mesh (o3d.geometry.TriangleMesh): 최적화된 메시
        
    Returns:
        dict: 품질 측정 결과
    """
    if original_mesh is None or optimized_mesh is None:
        return {}
    
    # 기본 통계
    orig_vertices = len(original_mesh.vertices)
    opt_vertices = len(optimized_mesh.vertices)
    orig_triangles = len(original_mesh.triangles)
    opt_triangles = len(optimized_mesh.triangles)
    
    vertex_reduction = (orig_vertices - opt_vertices) / orig_vertices * 100
    triangle_reduction = (orig_triangles - opt_triangles) / orig_triangles * 100
    
    # 표면적 보존 정도
    orig_area = original_mesh.get_surface_area()
    opt_area = optimized_mesh.get_surface_area()
    area_preservation = min(opt_area / orig_area, orig_area / opt_area) * 100
    
    # 부피 보존 정도 (watertight 메시인 경우)
    volume_preservation = 100.0
    if original_mesh.is_watertight() and optimized_mesh.is_watertight():
        orig_volume = original_mesh.get_volume()
        opt_volume = optimized_mesh.get_volume()
        if orig_volume > 0:
            volume_preservation = min(opt_volume / orig_volume, orig_volume / opt_volume) * 100
    
    # 전체 품질 점수
    quality_score = (area_preservation + volume_preservation) / 2
    
    return {
        'vertex_reduction_percent': vertex_reduction,
        'triangle_reduction_percent': triangle_reduction,
        'area_preservation_percent': area_preservation,
        'volume_preservation_percent': volume_preservation,
        'overall_quality_score': quality_score,
        'compression_ratio': orig_triangles / max(1, opt_triangles)
    }


def save_optimized_mesh(mesh, output_dir, filename, quality_info=None):
    """
    최적화된 메시를 저장합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 저장할 메시
        output_dir (str): 출력 디렉토리
        filename (str): 파일명 (확장자 제외)
        quality_info (dict): 품질 정보 (선택사항)
        
    Returns:
        list: 저장된 파일 경로 리스트
    """
    if mesh is None:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # 다양한 형식으로 저장
    formats = [
        ('.obj', '범용 3D 모델 형식'),
        ('.ply', '포인트 클라우드 형식'),
        ('.stl', '3D 프린팅 형식')
    ]
    
    for ext, description in formats:
        filepath = os.path.join(output_dir, f"{filename}{ext}")
        try:
            success = o3d.io.write_triangle_mesh(filepath, mesh)
            if success:
                saved_files.append(filepath)
                print(f"저장 완료 ({description}): {filepath}")
            else:
                print(f"저장 실패: {filepath}")
        except Exception as e:
            print(f"저장 중 오류 ({ext}): {e}")
    
    # 품질 정보 저장
    if quality_info and saved_files:
        info_path = os.path.join(output_dir, f"{filename}_optimization_info.txt")
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("=== 메시 최적화 정보 ===\n\n")
                f.write(f"버텍스 감소율: {quality_info.get('vertex_reduction_percent', 0):.1f}%\n")
                f.write(f"삼각형 감소율: {quality_info.get('triangle_reduction_percent', 0):.1f}%\n")
                f.write(f"표면적 보존도: {quality_info.get('area_preservation_percent', 0):.1f}%\n")
                f.write(f"부피 보존도: {quality_info.get('volume_preservation_percent', 0):.1f}%\n")
                f.write(f"전체 품질 점수: {quality_info.get('overall_quality_score', 0):.1f}/100\n")
                f.write(f"압축 비율: {quality_info.get('compression_ratio', 1):.2f}:1\n")
            print(f"최적화 정보 저장: {info_path}")
        except Exception as e:
            print(f"정보 파일 저장 중 오류: {e}")
    
    return saved_files