import numpy as np
import os
import cv2
import open3d as o3d
import copy

from modules.pointcloud_generator import (
    load_depth_map, 
    create_point_cloud_from_depth,
    remove_noise_from_pointcloud
)
from modules.fpfh_alignment import align_point_clouds_fpfh
from modules.skeleton_parser import (
    detect_landmarks_with_ai,
    create_skeleton_from_pointcloud,
    calculate_spine_angles,
    create_skeleton_visualization,
    print_angles
)
from modules.mesh_generator import create_and_save_mesh


def process_depth_maps(views_dict, debug_save=True): # 뎁스 파싱
    point_clouds = {}
    
    for view_name, file_path in views_dict.items():
        print(f"\n{view_name} 뷰 처리 중...")
        depth_map = load_depth_map(file_path)
        
        if depth_map is not None:
            # 디버깅을 위해 마스크 저장
            if debug_save:
                from modules.pointcloud_generator import create_mask_from_depth
                mask = create_mask_from_depth(depth_map, threshold_low=0.2, threshold_high=0.95)
                
                debug_dir = "output/debug"
                os.makedirs(debug_dir, exist_ok=True)
                mask_path = os.path.join(debug_dir, f"{view_name}_mask.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                print(f"마스크 저장됨: {mask_path}")
            
            pcd = create_point_cloud_from_depth(depth_map, view_name)
            if pcd is not None:
                # 법선 벡터 계산
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
                point_clouds[view_name] = pcd
    
    return point_clouds


def align_point_clouds(point_clouds):
    print("\n=== FPFH 기반 포인트 클라우드 정렬 단계 ===")
    
    # 정면을 기준으로 정렬 시작
    aligned_clouds = [point_clouds["front"]]
    front_target = point_clouds["front"]
    
    # 좌측과 우측을 정면과 정렬
    left_aligned = None
    right_aligned = None
    
    if "left" in point_clouds:
        print("\n좌측 뷰를 정면과 정렬...")
        params_align = {
            'voxel_coarse': 5.0,
            'voxel_list': [20.0, 10.0, 5.0],
            'ransac_iter': 20000,
            'use_cpd': False,
            'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
            'fitness_threshold_accept': 0.02,
            'force_cpd': False,
            'allow_rotation': False,
            'allow_small_rotation': True
        }
        left_aligned = align_point_clouds_fpfh(point_clouds["left"], front_target, params=params_align)
        aligned_clouds.append(left_aligned)
    
    if "right" in point_clouds:
        print("\n우측 뷰를 정면과 정렬...")
        params_align = {
            'voxel_coarse': 5.0,
            'voxel_list': [20.0, 10.0, 5.0],
            'ransac_iter': 20000,
            'use_cpd': False,
            'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
            'fitness_threshold_accept': 0.02,
            'force_cpd': False,
            'allow_rotation': False,
            'allow_small_rotation': True
        }
        right_aligned = align_point_clouds_fpfh(point_clouds["right"], front_target, params=params_align)
        aligned_clouds.append(right_aligned)
    
    # 후면은 좌/우 포인트 클라우드(정렬 결과)에만 정렬
    if "back" in point_clouds:
        print("\n후면 뷰를 좌/우 누적 클라우드에 정렬...")
        side_target = o3d.geometry.PointCloud()
        st_points = []
        st_colors = []

        # 좌측 정렬 결과
        if left_aligned is not None and len(left_aligned.points) > 0:
            st_points.extend(np.asarray(left_aligned.points))
            if left_aligned.has_colors():
                st_colors.extend(np.asarray(left_aligned.colors))

        # 우측 정렬 결과
        if right_aligned is not None and len(right_aligned.points) > 0:
            st_points.extend(np.asarray(right_aligned.points))
            if right_aligned.has_colors():
                st_colors.extend(np.asarray(right_aligned.colors))

        if len(st_points) == 0:
            print("  좌/우 타겟이 비어 있어 후면 정렬을 건너뜁니다.")
        else:
            side_target.points = o3d.utility.Vector3dVector(np.array(st_points))
            if len(st_colors) == len(st_points) and len(st_colors) > 0:
                side_target.colors = o3d.utility.Vector3dVector(np.array(st_colors))

            # 후면을 좌/우 타겟에 정렬
            params_align = {
                'voxel_coarse': 5.0,
                'voxel_list': [25.0, 12.0, 6.0],
                'ransac_iter': 30000,
                'use_cpd': False,
                'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
                'fitness_threshold_accept': 0.02,
                'force_cpd': False,
                'allow_rotation': False,
                'allow_small_rotation': True
            }
            back_aligned = align_point_clouds_fpfh(point_clouds["back"], side_target, params=params_align)
            aligned_clouds.append(back_aligned)
    
    return aligned_clouds


def merge_and_clean_pointclouds(aligned_clouds):
    print(f"\n=== 최종 병합 및 이상치 제거 ===")
    
    # 모든 포인트 클라우드를 하나로 합치기
    merged_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for pcd in aligned_clouds:
        points.extend(np.asarray(pcd.points))
        colors.extend(np.asarray(pcd.colors))
    merged_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    merged_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    print(f"병합된 포인트 수: {len(merged_cloud.points)}")
    
    # 극단적인 이상치만 제거 (다운샘플링 최소화)
    cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.5)  # 매우 관대한 기준
    merged_cloud = cl
    print(f"극단적 이상치 제거 후: {len(merged_cloud.points)} 포인트")
    
    # 법선 벡터 재계산
    merged_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    return merged_cloud


def analyze_posture(merged_cloud, front_image_path):
    """
    포인트 클라우드에서 스켈레톤을 생성하고 자세를 분석합니다.
    
    Args:
        merged_cloud (o3d.geometry.PointCloud): 병합된 포인트 클라우드
        front_image_path (str): 정면 이미지 경로 (AI 랜드마크 검출용)
        
    Returns:
        tuple: (skeleton_points, angles, skeleton_pcd, skeleton_cylinders)
    """
    print("\n=== 스켈레톤 생성 및 자세 분석 ===")
    
    # AI 기반 랜드마크 검출
    front_landmarks = detect_landmarks_with_ai(front_image_path)
    
    if front_landmarks:
        print("AI 랜드마크 검출 성공! 개인별 신체 특징을 반영한 정확한 스켈레톤을 생성합니다.")
        for name, landmark in front_landmarks.items():
            print(f"  {name}: x={landmark['x']:.1f}, y={landmark['y']:.1f}, visibility={landmark['visibility']:.3f}")
    else:
        print("AI 랜드마크 검출 실패, 기본 해부학적 비율을 사용합니다.")
    
    # 스켈레톤 생성 및 각도 분석
    skeleton_points = create_skeleton_from_pointcloud(merged_cloud, front_landmarks)
    angles = calculate_spine_angles(skeleton_points)
    skeleton_pcd, skeleton_cylinders = create_skeleton_visualization(skeleton_points)
    
    # 각도 분석 결과 출력
    print_angles(angles)
    
    return skeleton_points, angles, skeleton_pcd, skeleton_cylinders


def visualize_results(merged_cloud, mesh, skeleton_pcd, skeleton_cylinders):
    """
    결과를 시각화합니다.
    
    Args:
        merged_cloud (o3d.geometry.PointCloud): 병합된 포인트 클라우드
        mesh (o3d.geometry.TriangleMesh): 생성된 메시
        skeleton_pcd (o3d.geometry.PointCloud): 스켈레톤 포인트 클라우드
        skeleton_cylinders (list): 스켈레톤 연결선 실린더 리스트
    """
    print("\n=== 3D 시각화 ===")
    
    # 시각화 창 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Pose Analysis with FPFH Alignment", width=1024, height=768)
    
    # 포인트 클라우드 추가 (더 투명하게)
    merged_cloud_small = merged_cloud.voxel_down_sample(voxel_size=5.0)  # 더 많이 다운샘플링
    merged_cloud_small.paint_uniform_color([0.3, 0.3, 0.3])  # 더 어두운 회색으로 반투명 효과
    vis.add_geometry(merged_cloud_small)
    
    # 메시 추가 (있는 경우)
    if mesh is not None:
        mesh.paint_uniform_color([0.8, 0.8, 0.9])  # 연한 회색
        vis.add_geometry(mesh)
    
    # 스켈레톤 추가
    vis.add_geometry(skeleton_pcd)
    for cylinder in skeleton_cylinders:
        vis.add_geometry(cylinder)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.point_size = 8.0  # 스켈레톤 포인트 크기를 크게 설정
    opt.background_color = np.asarray([0, 0, 0])  # 검은색 배경
    opt.mesh_show_wireframe = True  # 와이어프레임 표시
    opt.mesh_show_back_face = True  # 메시 뒷면도 표시
    
    # 카메라 위치 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_up([0, -1, 0])
    
    # 시각화 실행
    vis.run()
    vis.destroy_window()


def main():
    """메인 실행 함수"""
    print("="*60)
    print("     모듈화된 3D 자세 분석 시스템")
    print("     FPFH 정렬 + 스켈레톤 파싱")
    print("="*60)
    
    # 입력 이미지 경로 설정
    views = {
        "front": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_정면.bmp",
        "right": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_오른쪽.bmp",
        "left": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_왼쪽.bmp",
        "back": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_후면.bmp"
    }
    
    # 다른 테스트 데이터 (남성)
    # views = {
    #     "front": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\남성\남_정면.bmp",
    #     "right": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\남성\남_오른쪽.bmp",
    #     "left": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\남성\남_왼쪽.bmp",
    #     "back": r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\남성\남_후면.bmp"
    # }
    
    try:
        # 1단계: 깊이맵 처리 및 포인트 클라우드 생성
        point_clouds = process_depth_maps(views, debug_save=True)
        
        if not point_clouds:
            print("포인트 클라우드 생성 실패. 프로그램을 종료합니다.")
            return
        
        # 2단계: FPFH 기반 포인트 클라우드 정렬
        aligned_clouds = align_point_clouds(point_clouds)
        
        # 3단계: 포인트 클라우드 병합 및 정리
        merged_cloud = merge_and_clean_pointclouds(aligned_clouds)
        
        # 4단계: 메시 생성 및 저장
        mesh, saved_files = create_and_save_mesh(merged_cloud, "output/3d_models", "body_mesh_fpfh")
        
        if saved_files:
            print(f"\n메시 파일이 저장되었습니다:")
            for file_path in saved_files:
                print(f"  {file_path}")
        
        # 5단계: 스켈레톤 파싱 및 자세 분석
        skeleton_points, angles, skeleton_pcd, skeleton_cylinders = analyze_posture(
            merged_cloud, views["front"]
        )
        
        # 6단계: 결과 시각화
        visualize_results(merged_cloud, mesh, skeleton_pcd, skeleton_cylinders)
        
        print("\n="*60)
        print("     3D 자세 분석이 완료되었습니다!")
        print("="*60)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()