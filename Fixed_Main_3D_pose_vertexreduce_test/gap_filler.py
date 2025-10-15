#!/usr/bin/env python3
"""
실시간 큰 구멍 채우기 도구

이 스크립트는 기존 메시에서 옆구리, 팔 안쪽 등의 큰 구멍을 
실시간으로 감지하고 채우는 도구입니다.
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.hole_filling import (
    fill_large_gaps_intelligently,
    bilateral_symmetry_completion,
    advanced_hole_filling,
    compare_before_after,
    detect_mesh_holes
)


def load_mesh_file(file_path):
    """
    메시 파일을 로드합니다.
    """
    print(f"메시 로딩: {file_path}")
    
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        
        if len(mesh.vertices) == 0:
            print("❌ 빈 메시입니다.")
            return None
        
        print(f"✅ 메시 로드 성공: {len(mesh.vertices):,}개 버텍스, {len(mesh.triangles):,}개 삼각형")
        return mesh
        
    except Exception as e:
        print(f"❌ 메시 로딩 실패: {e}")
        return None


def analyze_gaps(mesh):
    """
    메시의 구멍을 분석합니다.
    """
    print("\n=== 구멍 분석 ===")
    
    if mesh is None:
        return
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # 기본 정보
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    
    print(f"메시 중심: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"메시 크기: 너비={extent[0]:.1f}, 높이={extent[1]:.1f}, 깊이={extent[2]:.1f}")
    
    # 매니폴드 여부 확인
    if mesh.is_watertight():
        print("✅ 매니폴드 메시 (물이 새지 않음)")
    else:
        print("⚠️  비매니폴드 메시 (구멍 있음)")
    
    # 홀 감지
    holes = detect_mesh_holes(mesh, hole_size_threshold=20)
    if holes:
        print(f"🔍 감지된 홀: {len(holes)}개")
        for i, hole in enumerate(holes):
            print(f"  홀 {i+1}: 경계 길이 {hole['boundary_length']}")
    else:
        print("🔍 감지된 홀: 없음")
    
    # 밀도 분석 (간단한 3D 그리드)
    print("\n=== 밀도 분석 ===")
    
    # 좌우 분할
    left_vertices = vertices[vertices[:, 0] < center[0]]
    right_vertices = vertices[vertices[:, 0] > center[0]]
    
    left_ratio = len(left_vertices) / len(vertices)
    right_ratio = len(right_vertices) / len(vertices)
    
    print(f"좌측 밀도: {len(left_vertices):,}개 ({left_ratio:.1%})")
    print(f"우측 밀도: {len(right_vertices):,}개 ({right_ratio:.1%})")
    
    # 불균형 정도
    imbalance = abs(left_ratio - right_ratio)
    if imbalance > 0.15:
        print(f"⚠️  좌우 불균형 감지: {imbalance:.1%} 차이")
        if len(left_vertices) > len(right_vertices):
            print("   → 우측에 데이터 부족")
        else:
            print("   → 좌측에 데이터 부족")
    else:
        print("✅ 좌우 균형 양호")
    
    # 영역별 밀도 (상/중/하)
    upper_y = center[1] + extent[1] * 0.3
    lower_y = center[1] - extent[1] * 0.3
    
    upper_vertices = vertices[vertices[:, 1] > upper_y]
    middle_vertices = vertices[(vertices[:, 1] >= lower_y) & (vertices[:, 1] <= upper_y)]
    lower_vertices = vertices[vertices[:, 1] < lower_y]
    
    print(f"\n상체 밀도: {len(upper_vertices):,}개")
    print(f"몸통 밀도: {len(middle_vertices):,}개") 
    print(f"하체 밀도: {len(lower_vertices):,}개")
    
    # 옆구리 영역 밀도 확인
    left_side = vertices[(vertices[:, 0] < center[0] - extent[0] * 0.3) & 
                       (vertices[:, 1] > center[1] - extent[1] * 0.2) &
                       (vertices[:, 1] < center[1] + extent[1] * 0.3)]
    
    right_side = vertices[(vertices[:, 0] > center[0] + extent[0] * 0.3) & 
                         (vertices[:, 1] > center[1] - extent[1] * 0.2) &
                         (vertices[:, 1] < center[1] + extent[1] * 0.3)]
    
    print(f"\n옆구리 분석:")
    print(f"좌측 옆구리: {len(left_side):,}개")
    print(f"우측 옆구리: {len(right_side):,}개")
    
    if len(left_side) < 100 or len(right_side) < 100:
        print("⚠️  옆구리 영역에 데이터 부족 (< 100 버텍스)")
    
    return {
        'holes': holes,
        'left_right_imbalance': imbalance,
        'side_data_sufficient': len(left_side) >= 100 and len(right_side) >= 100
    }


def interactive_gap_filling(mesh):
    """
    대화형 구멍 채우기
    """
    print(f"\n{'='*60}")
    print("대화형 구멍 채우기")
    print(f"{'='*60}")
    
    # 분석 결과
    analysis = analyze_gaps(mesh)
    
    # 추천 방법 결정
    recommendations = []
    
    if analysis['left_right_imbalance'] > 0.15:
        recommendations.append("bilateral_symmetry (좌우 불균형 보정)")
    
    if not analysis['side_data_sufficient']:
        recommendations.append("large_gaps (옆구리 데이터 부족)")
    
    if analysis['holes']:
        recommendations.append("comprehensive (구멍 감지됨)")
    
    if not recommendations:
        recommendations.append("symmetry (일반적 대칭 보정)")
    
    print(f"\n추천 방법:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # 사용 가능한 방법들
    methods = {
        '1': ('large_gaps', '큰 구멍 전용 (옆구리, 팔 안쪽)'),
        '2': ('bilateral_symmetry', '양측 대칭성 완성'),
        '3': ('comprehensive', '종합적 접근법'),
        '4': ('symmetry', '기본 대칭성 복원'),
        '5': ('anatomical', '해부학적 보정'),
        '6': ('morphological', '형태학적 스무딩')
    }
    
    print(f"\n사용 가능한 방법:")
    for key, (method, desc) in methods.items():
        print(f"  {key}. {desc}")
    
    # 사용자 선택
    while True:
        choice = input(f"\n방법 선택 (1-6, 또는 'q' 종료): ").strip()
        
        if choice.lower() == 'q':
            return None, None
        
        if choice in methods:
            method, description = methods[choice]
            print(f"\n선택된 방법: {description}")
            break
        else:
            print("올바른 번호를 입력하세요.")
    
    # 홀 채우기 실행
    print(f"\n🔧 홀 채우기 실행 중...")
    
    try:
        filled_mesh = advanced_hole_filling(mesh, method=method)
        
        if filled_mesh is not None:
            # 결과 분석
            analysis_result = compare_before_after(mesh, filled_mesh)
            
            print(f"\n✅ 홀 채우기 완료!")
            if analysis_result:
                print(f"버텍스 변화: {analysis_result['vertex_increase_percent']:+.1f}%")
                print(f"표면적 변화: {analysis_result['surface_area_increase_percent']:+.1f}%")
            
            return filled_mesh, method
        else:
            print("❌ 홀 채우기 실패")
            return None, None
            
    except Exception as e:
        print(f"❌ 홀 채우기 중 오류: {e}")
        return None, None


def save_result(mesh, method, output_dir="output/gap_filled"):
    """
    결과 저장
    """
    if mesh is None:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gap_filled_{method}_{timestamp}.ply"
    filepath = os.path.join(output_dir, filename)
    
    try:
        success = o3d.io.write_triangle_mesh(filepath, mesh)
        if success:
            print(f"✅ 결과 저장: {filepath}")
            return filepath
        else:
            print(f"❌ 저장 실패: {filepath}")
            return None
    except Exception as e:
        print(f"❌ 저장 중 오류: {e}")
        return None


def visualize_comparison(original_mesh, filled_mesh):
    """
    원본과 채워진 메시 비교 시각화
    """
    if original_mesh is None or filled_mesh is None:
        print("시각화할 메시가 없습니다.")
        return
    
    print("\n시각화 준비 중...")
    
    # 메시 복사 및 위치 조정
    original_copy = original_mesh.__copy__()
    filled_copy = filled_mesh.__copy__()
    
    # 바운딩 박스 계산
    bbox = original_mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    
    # 좌우로 배치
    original_copy.translate([-extent[0] * 0.6, 0, 0])
    filled_copy.translate([extent[0] * 0.6, 0, 0])
    
    # 색상 설정
    original_copy.paint_uniform_color([0.8, 0.6, 0.6])  # 연한 빨간색
    filled_copy.paint_uniform_color([0.6, 0.8, 0.6])    # 연한 녹색
    
    # 시각화
    print("좌측: 원본 메시 (빨간색)")
    print("우측: 채워진 메시 (녹색)")
    
    o3d.visualization.draw_geometries(
        [original_copy, filled_copy],
        window_name="구멍 채우기 결과 비교",
        width=1200,
        height=800
    )


def main():
    parser = argparse.ArgumentParser(description="실시간 큰 구멍 채우기 도구")
    parser.add_argument("input_file", nargs="?", help="입력 메시 파일 경로")
    parser.add_argument("--method", choices=['large_gaps', 'bilateral_symmetry', 'comprehensive', 'symmetry', 'anatomical', 'morphological'], 
                       help="자동 실행할 방법")
    parser.add_argument("--output", help="출력 디렉토리")
    parser.add_argument("--visualize", action="store_true", help="결과 시각화")
    
    args = parser.parse_args()
    
    # 입력 파일 결정
    input_file = args.input_file
    
    if not input_file:
        # 최신 메시 파일 자동 찾기
        mesh_dirs = ["output/3d_models", "output/large_gap_demo"]
        
        for mesh_dir in mesh_dirs:
            if os.path.exists(mesh_dir):
                ply_files = [f for f in os.listdir(mesh_dir) if f.endswith('.ply')]
                if ply_files:
                    latest_file = max(ply_files, key=lambda f: os.path.getmtime(os.path.join(mesh_dir, f)))
                    input_file = os.path.join(mesh_dir, latest_file)
                    print(f"자동 선택된 파일: {input_file}")
                    break
    
    if not input_file or not os.path.exists(input_file):
        print("❌ 입력 파일을 찾을 수 없습니다.")
        print("사용법: python gap_filler.py [메시파일.ply]")
        return
    
    # 메시 로드
    mesh = load_mesh_file(input_file)
    if mesh is None:
        return
    
    # 자동 모드 vs 대화형 모드
    if args.method:
        # 자동 모드
        print(f"\n자동 모드: {args.method}")
        filled_mesh = advanced_hole_filling(mesh, method=args.method)
        method = args.method
    else:
        # 대화형 모드
        filled_mesh, method = interactive_gap_filling(mesh)
    
    if filled_mesh is None:
        print("홀 채우기가 완료되지 않았습니다.")
        return
    
    # 결과 저장
    output_dir = args.output or "output/gap_filled"
    saved_path = save_result(filled_mesh, method, output_dir)
    
    # 시각화
    if args.visualize or (not args.method and input("\n결과를 시각화하시겠습니까? (y/n): ").strip().lower() in ['y', 'yes', '예']):
        visualize_comparison(mesh, filled_mesh)
    
    print(f"\n{'='*60}")
    print("구멍 채우기 완료!")
    if saved_path:
        print(f"저장된 파일: {saved_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n작업이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()