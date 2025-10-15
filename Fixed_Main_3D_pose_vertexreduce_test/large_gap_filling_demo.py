#!/usr/bin/env python3
"""
큰 구멍(옆구리, 팔 안쪽) 채우기 데모

이 스크립트는 인체 측면 부분의 큰 구멍들을 지능적으로 채우는 
새로운 알고리즘을 테스트합니다.
"""

import os
import sys
import numpy as np
import open3d as o3d

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.hole_filling import (
    fill_large_gaps_intelligently,
    bilateral_symmetry_completion,
    advanced_hole_filling,
    compare_before_after
)


def load_test_mesh():
    """
    테스트용 메시를 로드합니다.
    """
    # 가장 최근에 생성된 메시 파일 찾기
    mesh_dir = "output/3d_models"
    if not os.path.exists(mesh_dir):
        print(f"메시 디렉토리가 없습니다: {mesh_dir}")
        return None
    
    # PLY 파일 중에서 가장 최근 파일 선택
    ply_files = [f for f in os.listdir(mesh_dir) if f.endswith('.ply')]
    
    if not ply_files:
        print(f"PLY 파일이 없습니다: {mesh_dir}")
        return None
    
    # 파일 크기가 가장 큰 것을 선택 (보통 가장 상세한 메시)
    largest_file = max(ply_files, key=lambda f: os.path.getsize(os.path.join(mesh_dir, f)))
    mesh_path = os.path.join(mesh_dir, largest_file)
    
    print(f"테스트 메시 로딩: {mesh_path}")
    
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        if len(mesh.vertices) == 0:
            print("빈 메시입니다.")
            return None
        
        print(f"로드된 메시: {len(mesh.vertices):,}개 버텍스, {len(mesh.triangles):,}개 삼각형")
        return mesh
        
    except Exception as e:
        print(f"메시 로딩 실패: {e}")
        return None


def create_artificial_large_gaps(mesh):
    """
    테스트를 위해 인공적으로 큰 구멍을 만듭니다 (옆구리, 팔 부분).
    """
    if mesh is None:
        return None
    
    print("\n=== 인공 큰 구멍 생성 (테스트용) ===")
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(vertices) == 0:
        return mesh
    
    # 바운딩 박스 계산
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    
    print(f"메시 중심: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"메시 크기: ({extent[0]:.1f}, {extent[1]:.1f}, {extent[2]:.1f})")
    
    # 제거할 영역 정의
    gaps_to_create = []
    
    # 1. 왼쪽 옆구리 구멍
    left_side_gap = {
        'name': '왼쪽 옆구리',
        'x_range': (center[0] - extent[0] * 0.45, center[0] - extent[0] * 0.25),
        'y_range': (center[1] - extent[1] * 0.1, center[1] + extent[1] * 0.3),
        'z_range': (center[2] - extent[2] * 0.2, center[2] + extent[2] * 0.2)
    }
    gaps_to_create.append(left_side_gap)
    
    # 2. 오른쪽 옆구리 구멍
    right_side_gap = {
        'name': '오른쪽 옆구리',
        'x_range': (center[0] + extent[0] * 0.25, center[0] + extent[0] * 0.45),
        'y_range': (center[1] - extent[1] * 0.1, center[1] + extent[1] * 0.3),
        'z_range': (center[2] - extent[2] * 0.2, center[2] + extent[2] * 0.2)
    }
    gaps_to_create.append(right_side_gap)
    
    # 3. 왼쪽 팔 안쪽 구멍
    left_arm_gap = {
        'name': '왼쪽 팔 안쪽',
        'x_range': (center[0] - extent[0] * 0.2, center[0] - extent[0] * 0.05),
        'y_range': (center[1] + extent[1] * 0.2, center[1] + extent[1] * 0.4),
        'z_range': (center[2] - extent[2] * 0.1, center[2] + extent[2] * 0.3)
    }
    gaps_to_create.append(left_arm_gap)
    
    # 4. 오른쪽 팔 안쪽 구멍
    right_arm_gap = {
        'name': '오른쪽 팔 안쪽',
        'x_range': (center[0] + extent[0] * 0.05, center[0] + extent[0] * 0.2),
        'y_range': (center[1] + extent[1] * 0.2, center[1] + extent[1] * 0.4),
        'z_range': (center[2] - extent[2] * 0.1, center[2] + extent[2] * 0.3)
    }
    gaps_to_create.append(right_arm_gap)
    
    # 구멍 생성
    vertices_to_keep = np.ones(len(vertices), dtype=bool)
    removed_count = 0
    
    for gap in gaps_to_create:
        # 해당 영역의 버텍스 찾기
        in_gap = ((vertices[:, 0] >= gap['x_range'][0]) & (vertices[:, 0] <= gap['x_range'][1]) &
                  (vertices[:, 1] >= gap['y_range'][0]) & (vertices[:, 1] <= gap['y_range'][1]) &
                  (vertices[:, 2] >= gap['z_range'][0]) & (vertices[:, 2] <= gap['z_range'][1]))
        
        gap_vertex_count = np.sum(in_gap)
        vertices_to_keep &= ~in_gap
        removed_count += gap_vertex_count
        
        print(f"  {gap['name']}: {gap_vertex_count}개 버텍스 제거")
    
    # 새로운 메시 생성
    if removed_count == 0:
        print("제거할 버텍스가 없습니다.")
        return mesh
    
    # 유지할 버텍스들
    new_vertices = vertices[vertices_to_keep]
    
    # 버텍스 인덱스 매핑
    old_to_new_index = {}
    new_index = 0
    for old_index in range(len(vertices)):
        if vertices_to_keep[old_index]:
            old_to_new_index[old_index] = new_index
            new_index += 1
    
    # 유효한 삼각형만 유지
    new_triangles = []
    for triangle in triangles:
        if all(idx in old_to_new_index for idx in triangle):
            new_triangle = [old_to_new_index[idx] for idx in triangle]
            new_triangles.append(new_triangle)
    
    # 새 메시 생성
    gappy_mesh = o3d.geometry.TriangleMesh()
    gappy_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    gappy_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # 법선 벡터 계산
    gappy_mesh.compute_vertex_normals()
    gappy_mesh.compute_triangle_normals()
    
    print(f"구멍난 메시 생성 완료: {removed_count:,}개 버텍스 제거")
    print(f"결과: {len(new_vertices):,}개 버텍스, {len(new_triangles):,}개 삼각형")
    
    return gappy_mesh


def demonstrate_large_gap_filling():
    """
    큰 구멍 채우기 알고리즘을 시연합니다.
    """
    print("="*80)
    print("     큰 구멍(옆구리, 팔 안쪽) 채우기 데모")
    print("="*80)
    
    # 1. 테스트 메시 로드
    original_mesh = load_test_mesh()
    if original_mesh is None:
        print("테스트 메시를 로드할 수 없습니다.")
        return
    
    # 2. 인공 구멍 생성
    gappy_mesh = create_artificial_large_gaps(original_mesh)
    if gappy_mesh is None:
        print("구멍 생성에 실패했습니다.")
        return
    
    # 3. 결과 저장 디렉토리 생성
    output_dir = "output/large_gap_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. 원본 메시 저장
    original_path = os.path.join(output_dir, "original_mesh.ply")
    o3d.io.write_triangle_mesh(original_path, original_mesh)
    print(f"\n원본 메시 저장: {original_path}")
    
    # 5. 구멍난 메시 저장
    gappy_path = os.path.join(output_dir, "gappy_mesh.ply")
    o3d.io.write_triangle_mesh(gappy_path, gappy_mesh)
    print(f"구멍난 메시 저장: {gappy_path}")
    
    # 6. 각 방법별로 테스트
    methods_to_test = [
        ("large_gaps", "큰 구멍 전용 알고리즘"),
        ("bilateral_symmetry", "양측 대칭성 완성"),
        ("comprehensive", "종합적 접근법")
    ]
    
    results = {}
    
    for method, description in methods_to_test:
        print(f"\n{'='*60}")
        print(f"테스트 중: {description}")
        print(f"{'='*60}")
        
        try:
            # 홀 채우기 실행
            filled_mesh = advanced_hole_filling(gappy_mesh, method=method)
            
            if filled_mesh is not None:
                # 결과 저장
                result_path = os.path.join(output_dir, f"filled_{method}.ply")
                o3d.io.write_triangle_mesh(result_path, filled_mesh)
                print(f"결과 저장: {result_path}")
                
                # 성능 분석
                analysis = compare_before_after(gappy_mesh, filled_mesh)
                results[method] = {
                    'mesh': filled_mesh,
                    'analysis': analysis,
                    'description': description
                }
                
        except Exception as e:
            print(f"{method} 테스트 중 오류: {e}")
    
    # 7. 결과 비교 및 요약
    print(f"\n{'='*80}")
    print("     결과 비교 요약")
    print(f"{'='*80}")
    
    print(f"{'방법':<20} {'버텍스 증가':<15} {'삼각형 증가':<15} {'표면적 증가':<15}")
    print("-" * 70)
    
    for method, result in results.items():
        analysis = result['analysis']
        if analysis:
            vertex_inc = analysis.get('vertex_increase_percent', 0)
            triangle_inc = analysis.get('triangle_increase_percent', 0)
            surface_inc = analysis.get('surface_area_increase_percent', 0)
            
            print(f"{method:<20} {vertex_inc:>+8.1f}%      {triangle_inc:>+8.1f}%      {surface_inc:>+8.1f}%")
    
    # 8. 최적 결과 추천
    if results:
        print(f"\n{'='*60}")
        print("추천 방법")
        print(f"{'='*60}")
        
        # 버텍스 증가율이 적당하면서 표면적 증가가 좋은 방법 찾기
        best_method = None
        best_score = -1
        
        for method, result in results.items():
            analysis = result['analysis']
            if analysis:
                vertex_inc = analysis.get('vertex_increase_percent', 0)
                surface_inc = analysis.get('surface_area_increase_percent', 0)
                
                # 점수 계산: 표면적 증가는 좋고, 과도한 버텍스 증가는 펜널티
                score = surface_inc - max(0, vertex_inc - 200) * 0.5  # 200% 이상 증가시 펜널티
                
                if score > best_score:
                    best_score = score
                    best_method = method
        
        if best_method:
            print(f"최적 방법: {best_method} ({results[best_method]['description']})")
            print(f"점수: {best_score:.1f}")
            
            # 최적 결과를 기본 이름으로 저장
            best_mesh = results[best_method]['mesh']
            best_path = os.path.join(output_dir, "best_filled_mesh.ply")
            o3d.io.write_triangle_mesh(best_path, best_mesh)
            print(f"최적 결과 저장: {best_path}")
    
    print(f"\n{'='*80}")
    print("데모 완료! 결과 파일들이 저장되었습니다.")
    print(f"저장 위치: {output_dir}")
    print(f"{'='*80}")


def visualize_comparison():
    """
    처리 전후 비교 시각화
    """
    print("\n=== 결과 시각화 ===")
    
    output_dir = "output/large_gap_demo"
    
    # 파일 경로들
    files_to_visualize = [
        ("original_mesh.ply", "원본 메시"),
        ("gappy_mesh.ply", "구멍난 메시"),
        ("best_filled_mesh.ply", "최적 복원 메시")
    ]
    
    geometries = []
    
    for filename, description in files_to_visualize:
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            try:
                mesh = o3d.io.read_triangle_mesh(filepath)
                
                if len(mesh.vertices) > 0:
                    # 각 메시를 다른 위치에 배치
                    offset = len(geometries) * 200  # X축으로 200 단위씩 이동
                    mesh.translate([offset, 0, 0])
                    
                    # 색상 설정
                    if "original" in filename:
                        mesh.paint_uniform_color([0.7, 0.7, 0.9])  # 연한 파란색
                    elif "gappy" in filename:
                        mesh.paint_uniform_color([0.9, 0.7, 0.7])  # 연한 빨간색
                    else:
                        mesh.paint_uniform_color([0.7, 0.9, 0.7])  # 연한 초록색
                    
                    geometries.append(mesh)
                    print(f"{description} 로드 완료: {len(mesh.vertices):,}개 버텍스")
                
            except Exception as e:
                print(f"{filename} 로드 실패: {e}")
    
    if geometries:
        print(f"\n{len(geometries)}개 메시를 시각화합니다...")
        print("좌측부터: 원본 → 구멍난 메시 → 복원된 메시")
        
        # 시각화
        o3d.visualization.draw_geometries(
            geometries,
            window_name="큰 구멍 채우기 결과 비교",
            width=1200,
            height=800
        )
    else:
        print("시각화할 메시가 없습니다.")


if __name__ == "__main__":
    try:
        # 큰 구멍 채우기 데모 실행
        demonstrate_large_gap_filling()
        
        # 사용자에게 시각화 여부 물어보기
        print("\n결과를 시각화하시겠습니까? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice in ['y', 'yes', '예', 'ㅇ']:
            visualize_comparison()
        
    except KeyboardInterrupt:
        print("\n\n데모가 중단되었습니다.")
    except Exception as e:
        print(f"\n데모 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()