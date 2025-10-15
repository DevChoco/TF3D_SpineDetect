#!/usr/bin/env python3
"""
구멍 채우기 결과 분석 리포트 생성기

여러 구멍 채우기 방법의 결과를 비교 분석하고 
상세한 리포트를 생성합니다.
"""

import os
import sys
import numpy as np
import open3d as o3d
import datetime
import json

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.hole_filling import compare_before_after


def load_all_results():
    """
    생성된 모든 결과 파일들을 로드합니다.
    """
    print("=== 결과 파일 수집 ===")
    
    # 검색할 디렉토리들
    search_dirs = [
        "output/gap_filled",
        "output/large_gap_demo",
        "output/3d_models"
    ]
    
    results = {}
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        print(f"\n{search_dir} 검색 중...")
        
        for filename in os.listdir(search_dir):
            if filename.endswith('.ply'):
                filepath = os.path.join(search_dir, filename)
                
                try:
                    mesh = o3d.io.read_triangle_mesh(filepath)
                    if len(mesh.vertices) > 0:
                        # 파일명에서 방법 추출
                        method = "unknown"
                        if "comprehensive" in filename:
                            method = "comprehensive"
                        elif "large_gaps" in filename:
                            method = "large_gaps"
                        elif "bilateral_symmetry" in filename:
                            method = "bilateral_symmetry"
                        elif "gappy" in filename:
                            method = "gappy_original"
                        elif "original" in filename:
                            method = "original"
                        elif "best" in filename:
                            method = "best_filled"
                        elif "ultra_high" in filename:
                            method = "original_ultra_high"
                        elif "ultra_low" in filename:
                            method = "original_ultra_low"
                        
                        results[method] = {
                            'mesh': mesh,
                            'filepath': filepath,
                            'filename': filename,
                            'vertices': len(mesh.vertices),
                            'triangles': len(mesh.triangles),
                            'surface_area': mesh.get_surface_area(),
                            'is_watertight': mesh.is_watertight()
                        }
                        
                        print(f"  ✅ {filename}: {len(mesh.vertices):,}개 버텍스")
                
                except Exception as e:
                    print(f"  ❌ {filename}: 로드 실패 ({e})")
    
    print(f"\n수집된 결과: {len(results)}개")
    return results


def analyze_coverage_improvement(original_mesh, filled_mesh):
    """
    구멍 채우기로 인한 커버리지 개선을 분석합니다.
    """
    if original_mesh is None or filled_mesh is None:
        return {}
    
    original_vertices = np.asarray(original_mesh.vertices)
    filled_vertices = np.asarray(filled_mesh.vertices)
    
    # 바운딩 박스 분석
    original_bbox = original_mesh.get_axis_aligned_bounding_box()
    filled_bbox = filled_mesh.get_axis_aligned_bounding_box()
    
    original_extent = original_bbox.get_extent()
    filled_extent = filled_bbox.get_extent()
    
    # 밀도 분석
    original_density = len(original_vertices) / (original_extent[0] * original_extent[1] * original_extent[2])
    filled_density = len(filled_vertices) / (filled_extent[0] * filled_extent[1] * filled_extent[2])
    
    # 좌우 균형 분석
    original_center = original_bbox.get_center()
    filled_center = filled_bbox.get_center()
    
    original_left = np.sum(original_vertices[:, 0] < original_center[0])
    original_right = np.sum(original_vertices[:, 0] > original_center[0])
    
    filled_left = np.sum(filled_vertices[:, 0] < filled_center[0])
    filled_right = np.sum(filled_vertices[:, 0] > filled_center[0])
    
    original_balance = abs(original_left - original_right) / len(original_vertices)
    filled_balance = abs(filled_left - filled_right) / len(filled_vertices)
    
    return {
        'density_improvement': (filled_density - original_density) / original_density * 100,
        'balance_improvement': (original_balance - filled_balance) * 100,
        'volume_coverage': {
            'original': original_extent[0] * original_extent[1] * original_extent[2],
            'filled': filled_extent[0] * filled_extent[1] * filled_extent[2]
        }
    }


def generate_comparison_report(results):
    """
    비교 분석 리포트를 생성합니다.
    """
    print("\n=== 비교 분석 리포트 생성 ===")
    
    # 기준 메시 찾기 (원본)
    reference_methods = ['original_ultra_low', 'original_ultra_high', 'gappy_original', 'original']
    reference_mesh = None
    reference_name = None
    
    for method in reference_methods:
        if method in results:
            reference_mesh = results[method]['mesh']
            reference_name = method
            break
    
    if reference_mesh is None:
        print("❌ 기준 메시를 찾을 수 없습니다.")
        return None
    
    print(f"📊 기준 메시: {reference_name}")
    
    # 분석할 방법들
    analysis_methods = ['comprehensive', 'large_gaps', 'bilateral_symmetry', 'best_filled']
    
    report = {
        'analysis_time': datetime.datetime.now().isoformat(),
        'reference_mesh': reference_name,
        'reference_stats': results[reference_name],
        'comparisons': {}
    }
    
    print(f"\n{'방법':<20} {'버텍스':<12} {'증가율':<10} {'표면적':<12} {'증가율':<10} {'품질':<8}")
    print("-" * 80)
    
    ref_vertices = results[reference_name]['vertices']
    ref_surface = results[reference_name]['surface_area']
    
    print(f"{reference_name:<20} {ref_vertices:<12,} {'기준':<10} {ref_surface:<12.1f} {'기준':<10} {'기준':<8}")
    
    for method in analysis_methods:
        if method not in results:
            continue
            
        mesh = results[method]['mesh']
        stats = results[method]
        
        # 기본 통계
        vertex_increase = ((stats['vertices'] - ref_vertices) / ref_vertices) * 100
        surface_increase = ((stats['surface_area'] - ref_surface) / ref_surface) * 100
        
        # 품질 점수 계산
        quality_score = 100
        if vertex_increase > 500:  # 500% 이상 증가는 과도함
            quality_score -= (vertex_increase - 500) * 0.1
        if surface_increase < 0:  # 표면적 감소는 좋지 않음
            quality_score += surface_increase  # 음수이므로 빼는 효과
        if stats['is_watertight']:
            quality_score += 10  # 물이 새지 않으면 보너스
        
        quality_score = max(0, min(100, quality_score))
        
        print(f"{method:<20} {stats['vertices']:<12,} {vertex_increase:>+8.1f}% {stats['surface_area']:<12.1f} {surface_increase:>+8.1f}% {quality_score:>6.1f}")
        
        # 상세 분석
        comparison_analysis = compare_before_after(reference_mesh, mesh)
        coverage_analysis = analyze_coverage_improvement(reference_mesh, mesh)
        
        report['comparisons'][method] = {
            'basic_stats': stats,
            'comparison_analysis': comparison_analysis,
            'coverage_analysis': coverage_analysis,
            'quality_score': quality_score
        }
    
    return report


def save_report(report, output_path="output/gap_filling_analysis_report.json"):
    """
    리포트를 JSON 파일로 저장합니다.
    """
    if report is None:
        return
    
    # 메시 객체는 직렬화할 수 없으므로 제거
    clean_report = {}
    for key, value in report.items():
        if key == 'comparisons':
            clean_report[key] = {}
            for method, data in value.items():
                clean_report[key][method] = {}
                for sub_key, sub_value in data.items():
                    if sub_key != 'basic_stats' or 'mesh' not in str(sub_value):
                        if sub_key == 'basic_stats':
                            # 메시 객체 제외한 기본 통계만 포함
                            clean_stats = {k: v for k, v in sub_value.items() if k != 'mesh'}
                            clean_report[key][method][sub_key] = clean_stats
                        else:
                            clean_report[key][method][sub_key] = sub_value
        elif key != 'reference_stats' or 'mesh' not in str(value):
            if key == 'reference_stats':
                clean_report[key] = {k: v for k, v in value.items() if k != 'mesh'}
            else:
                clean_report[key] = value
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_report, f, indent=2, ensure_ascii=False)
        print(f"\n📋 리포트 저장: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ 리포트 저장 실패: {e}")
        return None


def create_summary_visualization(results):
    """
    결과 요약 시각화
    """
    print("\n=== 결과 시각화 ===")
    
    # 시각화할 메시들 선택
    methods_to_show = ['original_ultra_low', 'comprehensive', 'large_gaps']
    colors = [
        [0.8, 0.6, 0.6],  # 원본: 연한 빨간색
        [0.6, 0.8, 0.6],  # 종합적: 연한 녹색  
        [0.6, 0.6, 0.8]   # 큰 구멍: 연한 파란색
    ]
    
    geometries = []
    labels = []
    
    for i, method in enumerate(methods_to_show):
        if method in results:
            mesh = results[method]['mesh'].__copy__()
            
            # 위치 조정 (X축으로 이동)
            offset = i * 200
            mesh.translate([offset, 0, 0])
            
            # 색상 설정
            mesh.paint_uniform_color(colors[i])
            
            geometries.append(mesh)
            
            # 라벨 생성
            method_names = {
                'original_ultra_low': '원본',
                'comprehensive': '종합적 채우기',
                'large_gaps': '큰 구멍 채우기'
            }
            labels.append(f"{method_names.get(method, method)}: {results[method]['vertices']:,}개")
    
    if geometries:
        print("시각화할 메시:")
        for label in labels:
            print(f"  - {label}")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="구멍 채우기 방법별 비교",
            width=1400,
            height=800
        )
    else:
        print("시각화할 메시가 없습니다.")


def main():
    """
    메인 실행 함수
    """
    print("="*80)
    print("     구멍 채우기 결과 분석 리포트")
    print("="*80)
    
    # 1. 모든 결과 로드
    results = load_all_results()
    
    if not results:
        print("❌ 분석할 결과가 없습니다.")
        return
    
    # 2. 비교 분석 리포트 생성
    report = generate_comparison_report(results)
    
    # 3. 리포트 저장
    if report:
        save_report(report)
    
    # 4. 요약 출력
    print(f"\n{'='*80}")
    print("분석 요약")
    print(f"{'='*80}")
    
    if report and report['comparisons']:
        best_method = None
        best_score = -1
        
        for method, data in report['comparisons'].items():
            score = data['quality_score']
            if score > best_score:
                best_score = score
                best_method = method
        
        if best_method:
            print(f"🏆 최적 방법: {best_method} (품질 점수: {best_score:.1f})")
            
            best_data = report['comparisons'][best_method]
            if 'comparison_analysis' in best_data:
                analysis = best_data['comparison_analysis']
                print(f"   버텍스 증가: {analysis.get('vertex_increase_percent', 0):.1f}%")
                print(f"   표면적 증가: {analysis.get('surface_area_increase_percent', 0):.1f}%")
    
    # 5. 시각화 여부 묻기
    choice = input("\n결과를 시각화하시겠습니까? (y/n): ").strip().lower()
    if choice in ['y', 'yes', '예']:
        create_summary_visualization(results)
    
    print(f"\n{'='*80}")
    print("분석 완료!")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n분석이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()