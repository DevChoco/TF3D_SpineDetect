"""
Ultra High LOD 메시만 생성하는 스크립트

기존에 생성된 포인트 클라우드나 메시에서 Ultra High 품질의 메시만 별도로 생성합니다.
"""

import numpy as np
import os
import open3d as o3d
import json

from modules.pointcloud_generator import (
    load_depth_map, 
    create_point_cloud_from_depth
)
from modules.fpfh_alignment import align_point_clouds_fpfh
from modules.skeleton_parser import (
    detect_landmarks_with_ai,
    create_skeleton_from_pointcloud,
    calculate_spine_angles,
    create_skeleton_visualization
)
from modules.mesh_generator import create_and_save_mesh


def save_skeleton_data(skeleton_points, skeleton_pcd, skeleton_cylinders, output_dir, merged_cloud=None, lod_level="ultra_high", angles=None):
    """스켈레톤 데이터를 JSON 파일로 저장"""
    if skeleton_points is None:
        return
    
    try:
        skeleton_data = {
            "points": {},
            "connections": [],
            "mesh_info": {},
            "lod_level": lod_level,
            "quality_metrics": {
                "angles": angles if angles else {}
            }
        }
        
        # 메시/포인트클라우드 크기 정보 저장
        if merged_cloud is not None and len(merged_cloud.points) > 0:
            points = np.asarray(merged_cloud.points)
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            
            skeleton_data["mesh_info"] = {
                "height": float(max_bound[1] - min_bound[1]),
                "width": float(max_bound[0] - min_bound[0]),
                "depth": float(max_bound[2] - min_bound[2]),
                "min_bound": {
                    "x": float(min_bound[0]),
                    "y": float(min_bound[1]),
                    "z": float(min_bound[2])
                },
                "max_bound": {
                    "x": float(max_bound[0]),
                    "y": float(max_bound[1]),
                    "z": float(max_bound[2])
                }
            }
        
        # 포인트 데이터 저장
        for name, point in skeleton_points.items():
            if point is not None:
                skeleton_data["points"][name] = {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2])
                }
        
        # 연결선 데이터 저장
        connections = [
            ["HEAD", "NECK"],
            ["NECK", "SPINE_UPPER"],
            ["SPINE_UPPER", "SPINE_MID"],
            ["SPINE_MID", "SPINE_LOWER"],
            ["SPINE_LOWER", "PELVIS"],
            ["NECK", "SHOULDER_LEFT"],
            ["NECK", "SHOULDER_RIGHT"],
            ["SHOULDER_LEFT", "ELBOW_LEFT"],
            ["SHOULDER_RIGHT", "ELBOW_RIGHT"],
            ["ELBOW_LEFT", "WRIST_LEFT"],
            ["ELBOW_RIGHT", "WRIST_RIGHT"],
            ["PELVIS", "HIP_LEFT"],
            ["PELVIS", "HIP_RIGHT"],
            ["HIP_LEFT", "KNEE_LEFT"],
            ["HIP_RIGHT", "KNEE_RIGHT"],
            ["KNEE_LEFT", "ANKLE_LEFT"],
            ["KNEE_RIGHT", "ANKLE_RIGHT"]
        ]
        
        for connection in connections:
            if connection[0] in skeleton_data["points"] and connection[1] in skeleton_data["points"]:
                skeleton_data["connections"].append(connection)
        
        # JSON 파일로 저장
        json_path = os.path.join(output_dir, f"skeleton_data_{lod_level}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n스켈레톤 데이터 저장 ({lod_level}): {json_path}")
        
    except Exception as e:
        print(f"스켈레톤 데이터 저장 중 오류: {e}")


def process_depth_maps(views_dict):
    """뎁스맵 처리"""
    point_clouds = {}
    
    for view_name, file_path in views_dict.items():
        print(f"\n{view_name} 뷰 처리 중...")
        depth_map = load_depth_map(file_path)
        
        if depth_map is not None:
            pcd = create_point_cloud_from_depth(depth_map, view_name)
            if pcd is not None:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
                point_clouds[view_name] = pcd
    
    return point_clouds


def align_point_clouds(point_clouds):
    """포인트 클라우드 정렬"""
    print(f"\n=== FPFH 기반 포인트 클라우드 정렬 ===")
    
    aligned_clouds = [point_clouds["front"]]
    front_target = point_clouds["front"]
    
    left_aligned = None
    right_aligned = None
    
    if "left" in point_clouds:
        print("\n좌측 뷰를 정면과 정렬...")
        params_align = {
            'voxel_coarse': 5.0,
            'voxel_list': [20.0, 10.0, 5.0],
            'ransac_iter': 20000,
            'use_cpd': False,
            'fitness_threshold_accept': 0.02,
            'allow_rotation': False,
            'allow_small_rotation': True,
            'use_point_to_plane_icp': True
        }
        left_aligned = align_point_clouds_fpfh(point_clouds["left"], front_target, params=params_align)
        aligned_clouds.append(left_aligned)
    
    if "right" in point_clouds:
        print("\n우측 뷰를 정면과 정렬...")
        params_align = {
            'voxel_coarse': 3.0,
            'voxel_list': [10.0, 5.0, 2.5, 1.0],
            'ransac_iter': 100000,
            'use_cpd': False,
            'fitness_threshold_accept': 0.02,
            'allow_rotation': True,
            'allow_small_rotation': True,
            'use_point_to_plane_icp': True
        }
        right_aligned = align_point_clouds_fpfh(point_clouds["right"], front_target, params=params_align)
        aligned_clouds.append(right_aligned)
    
    if "back" in point_clouds:
        print("\n후면 뷰를 좌/우 누적 클라우드에 정렬...")
        side_target = o3d.geometry.PointCloud()
        st_points = []
        st_colors = []

        if left_aligned is not None and len(left_aligned.points) > 0:
            st_points.extend(np.asarray(left_aligned.points))
            if left_aligned.has_colors():
                st_colors.extend(np.asarray(left_aligned.colors))

        if right_aligned is not None and len(right_aligned.points) > 0:
            st_points.extend(np.asarray(right_aligned.points))
            if right_aligned.has_colors():
                st_colors.extend(np.asarray(right_aligned.colors))

        if len(st_points) > 0:
            side_target.points = o3d.utility.Vector3dVector(np.array(st_points))
            if len(st_colors) == len(st_points) and len(st_colors) > 0:
                side_target.colors = o3d.utility.Vector3dVector(np.array(st_colors))

            params_align = {
                'voxel_coarse': 5.0,
                'voxel_list': [25.0, 12.0, 6.0],
                'ransac_iter': 30000,
                'use_cpd': False,
                'fitness_threshold_accept': 0.02,
                'allow_rotation': False,
                'allow_small_rotation': True,
                'use_point_to_plane_icp': True
            }
            back_aligned = align_point_clouds_fpfh(point_clouds["back"], side_target, params=params_align)
            aligned_clouds.append(back_aligned)
    
    return aligned_clouds


def merge_and_clean_pointclouds(aligned_clouds):
    """포인트 클라우드 병합 및 정리"""
    print(f"\n=== 최종 병합 및 이상치 제거 ===")
    
    merged_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for pcd in aligned_clouds:
        points.extend(np.asarray(pcd.points))
        colors.extend(np.asarray(pcd.colors))
    merged_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    merged_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    print(f"병합된 포인트 수: {len(merged_cloud.points)}")
    
    # 이상치 제거
    cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.5)
    merged_cloud = cl
    print(f"이상치 제거 후: {len(merged_cloud.points)} 포인트")
    
    # 법선 벡터 재계산
    merged_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    return merged_cloud


def create_ultra_high_mesh_only(merged_cloud, output_dir, base_filename="body_mesh_fpfh"):
    """
    Ultra High LOD 메시만 생성
    
    Args:
        merged_cloud: 병합된 포인트 클라우드
        output_dir: 출력 디렉토리
        base_filename: 기본 파일명
    """
    print("\n=== Ultra High LOD 메시 생성 ===")
    print("최고 품질의 메시를 생성합니다...")
    
    # Ultra High 설정: 버텍스 감소 없음 (또는 최소한만)
    mesh, saved_files = create_and_save_mesh(
        merged_cloud, 
        output_dir, 
        base_filename,
        create_lod=False,  # 다른 LOD는 생성하지 않음
        reduction_ratio=0.95,  # 5%만 감소 (거의 원본 유지)
        optimization_level="ultra_high_quality",  # 최고 품질
        enable_quality_analysis=True,
        enable_hole_filling=True,
        hole_filling_method="comprehensive"
    )
    
    if mesh is not None:
        # Ultra High 전용 파일명으로 저장
        ultra_high_files = []
        for fmt in ['obj', 'ply', 'stl']:
            ultra_high_path = os.path.join(output_dir, f"{base_filename}_ultra_high.{fmt}")
            
            # 파일이 이미 존재하는지 확인
            if os.path.exists(ultra_high_path):
                print(f"  기존 파일 덮어쓰기: {ultra_high_path}")
            
            # 파일 저장
            if fmt == 'obj':
                o3d.io.write_triangle_mesh(ultra_high_path, mesh, write_ascii=True)
            elif fmt == 'ply':
                o3d.io.write_triangle_mesh(ultra_high_path, mesh, write_ascii=False)
            elif fmt == 'stl':
                o3d.io.write_triangle_mesh(ultra_high_path, mesh, write_ascii=False)
            
            ultra_high_files.append(ultra_high_path)
            print(f"  저장 완료: {ultra_high_path}")
        
        # 메시 정보 출력
        print(f"\nUltra High 메시 정보:")
        print(f"  정점 수: {len(mesh.vertices):,}")
        print(f"  면 수: {len(mesh.triangles):,}")
        
        return mesh, ultra_high_files
    
    return None, []


def process_single_frame(frame_num, base_depth_path, output_base_dir):
    """단일 프레임 처리"""
    print("\n" + "="*60)
    print(f"     프레임 {frame_num} 처리 중...")
    print("="*60)
    
    # 프레임별 입력 경로
    views = {
        "front": os.path.join(base_depth_path, "여_정면", f"DepthMap{frame_num}.bmp"),
        "right": os.path.join(base_depth_path, "여_R", f"DepthMap{frame_num}.bmp"),
        "left": os.path.join(base_depth_path, "여_L", f"DepthMap{frame_num}.bmp"),
        "back": os.path.join(base_depth_path, "여_후면", f"DepthMap{frame_num}.bmp")
    }
    
    # 파일 존재 확인
    for view_name, path in views.items():
        if not os.path.exists(path):
            print(f"경고: {view_name} 파일이 없습니다: {path}")
            return False
    
    # 프레임별 출력 디렉토리
    frame_str = str(frame_num).zfill(3)
    output_dir = os.path.join(output_base_dir, f"frame_{frame_str}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[출력 경로]: {output_dir}")
    
    try:
        # 1. 뎁스맵 처리
        point_clouds = process_depth_maps(views)
        
        if not point_clouds:
            print(f"프레임 {frame_num}: 포인트 클라우드 생성 실패!")
            return False
        
        # 2. 포인트 클라우드 정렬
        aligned_clouds = align_point_clouds(point_clouds)
        
        # 3. 병합 및 정리
        merged_cloud = merge_and_clean_pointclouds(aligned_clouds)
        
        # 4. Ultra High 메시만 생성
        mesh, saved_files = create_ultra_high_mesh_only(
            merged_cloud, 
            output_dir, 
            base_filename=f"body_mesh_frame_{frame_str}"
        )
        
        if mesh is None:
            print(f"프레임 {frame_num}: Ultra High 메시 생성 실패!")
            return False
        
        print(f"\n프레임 {frame_num} 생성 완료:")
        for file_path in saved_files:
            print(f"  {file_path}")
        
        # 5. 스켈레톤 생성 및 저장
        front_landmarks = detect_landmarks_with_ai(views["front"])
        skeleton_points = create_skeleton_from_pointcloud(merged_cloud, front_landmarks)
        angles = calculate_spine_angles(skeleton_points)
        skeleton_pcd, skeleton_cylinders = create_skeleton_visualization(skeleton_points)
        
        # Ultra High용 스켈레톤 데이터 저장
        save_skeleton_data(skeleton_points, skeleton_pcd, skeleton_cylinders, 
                         output_dir, merged_cloud, lod_level="ultra_high", angles=angles)
        
        return True
        
    except Exception as e:
        print(f"프레임 {frame_num} 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 실행 함수 - 모든 여성 프레임 처리"""
    print("="*60)
    print("     Ultra High LOD 메시 전용 생성기")
    print("     모든 여성 프레임 처리 (0-59)")
    print("="*60)
    
    # 기본 경로 설정
    base_depth_path = r"D:\Lab2\--final_3D_Body--\3D_Body_Posture_Analysis\test2\DepthMap"
    
    try:
        # 출력 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base_dir = os.path.join(script_dir, "output", "3d_models")
        os.makedirs(output_base_dir, exist_ok=True)
        
        print(f"\n[기본 출력 경로]: {output_base_dir}")
        
        # 처리할 프레임 범위 (0-59)
        total_frames = 60
        success_count = 0
        fail_count = 0
        
        print(f"\n총 {total_frames}개 프레임 처리 시작...\n")
        
        # 모든 프레임 반복 처리
        for frame_num in range(total_frames):
            success = process_single_frame(frame_num, base_depth_path, output_base_dir)
            
            if success:
                success_count += 1
                print(f"✓ 프레임 {frame_num} 완료 ({success_count}/{total_frames})")
            else:
                fail_count += 1
                print(f"✗ 프레임 {frame_num} 실패")
            
            # 진행 상황 출력
            if (frame_num + 1) % 10 == 0:
                print(f"\n--- 진행 상황: {frame_num + 1}/{total_frames} 프레임 처리 완료 ---")
                print(f"    성공: {success_count}, 실패: {fail_count}\n")
        
        # 최종 결과 출력
        print("\n" + "="*60)
        print("     Ultra High LOD 메시 생성 완료!")
        print("="*60)
        print(f"\n최종 결과:")
        print(f"  총 프레임: {total_frames}")
        print(f"  성공: {success_count}")
        print(f"  실패: {fail_count}")
        print(f"  성공률: {(success_count/total_frames*100):.1f}%")
        print(f"\n출력 위치: {output_base_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
