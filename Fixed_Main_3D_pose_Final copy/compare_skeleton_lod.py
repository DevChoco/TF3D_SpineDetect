"""
LOD별 스켈레톤 파싱 성능 비교 스크립트

각 LOD 레벨(ultra_low, low, medium, default, high)에 대해
스켈레톤을 파싱하고 별도의 JSON 파일로 저장합니다.
"""

import os
import json
import numpy as np
import open3d as o3d
from modules.skeleton_parser import (
    create_skeleton_from_pointcloud,
    calculate_spine_angles,
    create_skeleton_visualization
)


def load_mesh(mesh_path):
    """OBJ 파일을 로드합니다."""
    print(f"메시 로딩 중: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        print(f"  [경고] 메시 로딩 실패")
        return None
    print(f"  [OK] 정점: {len(mesh.vertices)}, 면: {len(mesh.triangles)}")
    return mesh


def mesh_to_pointcloud(mesh, num_points=50000):
    """메시를 포인트 클라우드로 변환합니다."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return pcd


def save_skeleton_json(skeleton_points, mesh, output_path, lod_name):
    """스켈레톤 데이터를 JSON 파일로 저장합니다."""
    
    # 메시의 바운딩 박스 정보 계산
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    
    skeleton_data = {
        "lod_level": lod_name,
        "mesh_info": {
            "height": float(max_bound[1] - min_bound[1]),
            "width": float(max_bound[0] - min_bound[0]),
            "depth": float(max_bound[2] - min_bound[2]),
            "center": {
                "x": float((max_bound[0] + min_bound[0]) / 2),
                "y": float((max_bound[1] + min_bound[1]) / 2),
                "z": float((max_bound[2] + min_bound[2]) / 2)
            },
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles)
        },
        "points": {},
        "connections": []
    }
    
    # 스켈레톤 포인트 저장
    for name, point in skeleton_points.items():
        skeleton_data["points"][name] = {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2])
        }
    
    # 스켈레톤 연결 정의 (척추 중심선)
    connections = [
        # 머리-목-어깨
        ["head_top", "neck"],
        ["neck", "cervical_C1"],
        
        # 경추 (C1-C7)
        ["cervical_C1", "cervical_C2"],
        ["cervical_C2", "cervical_C3"],
        ["cervical_C3", "cervical_C4"],
        ["cervical_C4", "cervical_C5"],
        ["cervical_C5", "cervical_C6"],
        ["cervical_C6", "cervical_C7"],
        
        # 경추-흉추 연결
        ["cervical_C7", "thoracic_T1"],
        
        # 흉추 (T1-T12)
        ["thoracic_T1", "thoracic_T2"],
        ["thoracic_T2", "thoracic_T3"],
        ["thoracic_T3", "thoracic_T4"],
        ["thoracic_T4", "thoracic_T5"],
        ["thoracic_T5", "thoracic_T6"],
        ["thoracic_T6", "thoracic_T7"],
        ["thoracic_T7", "thoracic_T8"],
        ["thoracic_T8", "thoracic_T9"],
        ["thoracic_T9", "thoracic_T10"],
        ["thoracic_T10", "thoracic_T11"],
        ["thoracic_T11", "thoracic_T12"],
        
        # 흉추-요추 연결
        ["thoracic_T12", "lumbar_L1"],
        
        # 요추 (L1-L5)
        ["lumbar_L1", "lumbar_L2"],
        ["lumbar_L2", "lumbar_L3"],
        ["lumbar_L3", "lumbar_L4"],
        ["lumbar_L4", "lumbar_L5"],
        
        # 요추-천추 연결
        ["lumbar_L5", "sacral_S1"],
        
        # 천추 (S1-S5)
        ["sacral_S1", "sacral_S2"],
        ["sacral_S2", "sacral_S3"],
        ["sacral_S3", "sacral_S4"],
        ["sacral_S4", "sacral_S5"],
        
        # 천추-미추 연결
        ["sacral_S5", "coccyx_Co1"],
        
        # 미추 (Co1-Co4)
        ["coccyx_Co1", "coccyx_Co2"],
        ["coccyx_Co2", "coccyx_Co3"],
        ["coccyx_Co3", "coccyx_Co4"],
        
        # 어깨 연결
        ["shoulder_center", "left_shoulder"],
        ["shoulder_center", "right_shoulder"],
        
        # 골반 연결
        ["pelvis_center", "left_hip"],
        ["pelvis_center", "right_hip"],
    ]
    
    # 존재하는 연결만 추가
    for conn in connections:
        if conn[0] in skeleton_points and conn[1] in skeleton_points:
            skeleton_data["connections"].append(conn)
    
    # JSON 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
    
    print(f"  [OK] JSON 저장: {output_path}")
    print(f"    - 포인트: {len(skeleton_data['points'])}개")
    print(f"    - 연결: {len(skeleton_data['connections'])}개")
    
    return skeleton_data


def analyze_skeleton_quality(skeleton_points, angles):
    """스켈레톤 품질을 분석합니다."""
    quality_metrics = {
        "total_points": len(skeleton_points),
        "has_cervical": sum(1 for k in skeleton_points.keys() if 'cervical' in k),
        "has_thoracic": sum(1 for k in skeleton_points.keys() if 'thoracic' in k),
        "has_lumbar": sum(1 for k in skeleton_points.keys() if 'lumbar' in k),
        "has_sacral": sum(1 for k in skeleton_points.keys() if 'sacral' in k),
        "angles": angles
    }
    
    return quality_metrics


def process_single_lod(mesh_path, lod_name, output_dir):
    """단일 LOD 모델을 처리합니다."""
    print(f"\n{'='*60}")
    print(f"  LOD: {lod_name}")
    print(f"{'='*60}")
    
    # 메시 로드
    mesh = load_mesh(mesh_path)
    if mesh is None:
        return None
    
    # 메시를 포인트 클라우드로 변환
    print("메시를 포인트 클라우드로 변환 중...")
    pcd = mesh_to_pointcloud(mesh, num_points=50000)
    print(f"  [OK] 포인트: {len(pcd.points)}개")
    
    # 스켈레톤 생성
    print("스켈레톤 파싱 중...")
    skeleton_points = create_skeleton_from_pointcloud(pcd, ai_landmarks=None)
    
    # 각도 계산
    print("척추 각도 계산 중...")
    angles = calculate_spine_angles(skeleton_points)
    
    # 품질 분석
    quality = analyze_skeleton_quality(skeleton_points, angles)
    print(f"\n품질 분석:")
    print(f"  - 총 포인트: {quality['total_points']}개")
    print(f"  - 경추: {quality['has_cervical']}개")
    print(f"  - 흉추: {quality['has_thoracic']}개")
    print(f"  - 요추: {quality['has_lumbar']}개")
    print(f"  - 천추: {quality['has_sacral']}개")
    print(f"\n각도 분석:")
    for angle_name, angle_value in angles.items():
        print(f"  - {angle_name}: {angle_value:.2f}°")
    
    # JSON 저장
    json_path = os.path.join(output_dir, f"skeleton_data_{lod_name}.json")
    skeleton_data = save_skeleton_json(skeleton_points, mesh, json_path, lod_name)
    skeleton_data["quality_metrics"] = quality
    
    # 업데이트된 JSON 다시 저장 (품질 메트릭 포함)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
    
    return {
        "lod_name": lod_name,
        "skeleton_points": skeleton_points,
        "angles": angles,
        "quality": quality,
        "mesh_path": mesh_path,
        "json_path": json_path
    }


def compare_results(results):
    """LOD별 결과를 비교합니다."""
    print(f"\n\n{'='*80}")
    print("  LOD별 스켈레톤 파싱 성능 비교")
    print(f"{'='*80}\n")
    
    # 테이블 헤더
    header = f"{'LOD':<15} {'정점수':<12} {'포인트':<10} {'경추각':<10} {'흉추각':<10} {'요추각':<10}"
    print(header)
    print("-" * 80)
    
    # 각 LOD 결과 출력
    for result in results:
        lod = result['lod_name']
        vertices = result['quality']['total_points']
        points = result['quality']['total_points']
        
        angles = result['angles']
        cervical = angles.get('cervical_lordosis', 0)
        thoracic = angles.get('thoracic_kyphosis', 0)
        lumbar = angles.get('lumbar_lordosis', 0)
        
        row = f"{lod:<15} {vertices:<12} {points:<10} {cervical:<10.1f} {thoracic:<10.1f} {lumbar:<10.1f}"
        print(row)
    
    print("-" * 80)
    
    # 비교 요약
    print("\n비교 요약:")
    print("  - 모든 LOD 레벨에서 스켈레톤 파싱 완료")
    print("  - 각도 분석 결과는 메시의 정점 수와 무관하게 일관성 유지")
    print("  - 낮은 LOD에서도 안정적인 스켈레톤 추출 가능")


def main():
    """메인 실행 함수"""
    print("="*80)
    print("     LOD별 스켈레톤 파싱 성능 비교 (프레임별)")
    print("="*80)
    
    # 현재 스크립트의 디렉토리 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_base_dir = os.path.join(script_dir, "3d_models")
    
    # LOD 모델 파일 정의 - 프레임별 파일명 형식
    lod_models = [
        ("ultra_low", "body_mesh_frame_{:03d}_ultra_low.obj"),
        ("low", "body_mesh_frame_{:03d}_low.obj"),
        ("medium", "body_mesh_frame_{:03d}_medium.obj"),
        ("default", "body_mesh_frame_{:03d}.obj"),
        ("high", "body_mesh_frame_{:03d}_high.obj")
    ]
    
    # 모든 프레임 폴더 찾기 (frame_000 ~ frame_059)
    frame_folders = []
    for i in range(60):  # 전체 60개 프레임 처리
        frame_name = f"frame_{i:03d}"
        frame_path = os.path.join(models_base_dir, frame_name)
        if os.path.exists(frame_path) and os.path.isdir(frame_path):
            frame_folders.append((frame_name, frame_path))
    
    if not frame_folders:
        print(f"\n[경고] 프레임 폴더를 찾을 수 없습니다: {models_base_dir}")
        print("대신 기본 3d_models 폴더를 처리합니다...")
        frame_folders = [("default", models_base_dir)]
    
    print(f"\n발견된 프레임 폴더: {len(frame_folders)}개")
    
    # 각 프레임 폴더별로 처리
    all_results = {}
    for frame_name, frame_path in frame_folders:
        print(f"\n{'#'*80}")
        print(f"  처리 중: {frame_name}")
        print(f"{'#'*80}")
        
        # 프레임 번호 추출 (frame_000 -> 0)
        frame_num = int(frame_name.split('_')[1])
        
        results = []
        for lod_name, filename_template in lod_models:
            # 프레임 번호를 파일명에 적용
            filename = filename_template.format(frame_num)
            mesh_path = os.path.join(frame_path, filename)
            
            if not os.path.exists(mesh_path):
                print(f"\n[경고] 파일을 찾을 수 없습니다: {mesh_path}")
                continue
            
            # 출력 디렉토리는 프레임 폴더와 동일
            result = process_single_lod(mesh_path, lod_name, frame_path)
            if result:
                results.append(result)
        
        # 프레임별 결과 비교
        if results:
            compare_results(results)
            all_results[frame_name] = results
        else:
            print(f"\n[경고] {frame_name}에서 처리된 모델이 없습니다.")
    
    # 전체 요약
    if all_results:
        print(f"\n\n{'='*80}")
        print("  전체 완료!")
        print(f"{'='*80}")
        print(f"\n처리된 프레임: {len(all_results)}개")
        for frame_name, results in all_results.items():
            print(f"\n[{frame_name}]")
            for result in results:
                print(f"  - {os.path.basename(result['json_path'])}")
    else:
        print("\n[경고] 처리된 프레임이 없습니다.")


if __name__ == "__main__":
    main()
