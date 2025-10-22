"""
논문용 테스트 이미지 생성 #1: 마스크 적용 전후 포인트 클라우드 비교

이 스크립트는 마스크 적용 전후의 포인트 클라우드 품질 차이를 시각적으로 보여줍니다.
- 좌측: 마스크 없이 생성한 포인트 클라우드 (노이즈 포함)
- 우측: 마스크 적용 후 포인트 클라우드 (깨끗한 결과)
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import sys

# 상위 디렉토리의 modules를 import하기 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Fixed_Main_3D_pose'))

from modules.pointcloud_generator import load_depth_map, create_mask_from_depth


def create_pointcloud_no_mask(depth_map, view="front"):
    """마스크 없이 포인트 클라우드 생성 (노이즈 포함)"""
    size = depth_map.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    
    x = x - size/2
    y = y - size/2
    scale = 100
    
    # 뷰에 따른 좌표 변환 (Y축 반전 - 발이 아래로)
    if view == "front":
        points = np.stack([x, -y, depth_map * scale * 1.1], axis=-1)
    else:
        points = np.stack([x, -y, depth_map * scale], axis=-1)
    
    # 최소한의 임계값만 적용 (노이즈 많이 남음)
    valid_mask = depth_map > 0.05
    points = points[valid_mask]
    
    # 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7, 0.3, 0.3])  # 빨간색
    
    return pcd


def create_pointcloud_with_mask(depth_map, view="front"):
    """마스크 적용하여 포인트 클라우드 생성 (깨끗한 결과)"""
    size = depth_map.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    
    # 마스크 생성 (형태학적 연산 + 연결 요소 분석)
    mask = create_mask_from_depth(depth_map, threshold_low=0.2, threshold_high=0.95)
    
    x = x - size/2
    y = y - size/2
    scale = 100
    
    # 뷰에 따른 좌표 변환 (Y축 반전 - 발이 아래로)
    if view == "front":
        points = np.stack([x, -y, depth_map * scale * 1.1], axis=-1)
    else:
        points = np.stack([x, -y, depth_map * scale], axis=-1)
    
    # 마스크 적용
    points = points[mask]
    
    # 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.3, 0.7, 0.3])  # 초록색
    
    return pcd


def capture_pointcloud_image(pcd, title=""):
    """포인트 클라우드를 이미지로 캡처"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=600)
    vis.add_geometry(pcd)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # 흰색 배경
    opt.point_size = 3.0
    
    # 카메라 설정 - 사용자 지정 Extrinsic Matrix
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    
    # Extrinsic Matrix 설정
    extrinsic = np.array([
        [ 2.17081021e-01, -8.23744991e-03,  9.76118832e-01, -4.65365706e+01],
        [-7.14191300e-02, -9.97418450e-01,  7.46585000e-03, -1.35329047e+00],
        [ 9.73537436e-01, -7.13342512e-02, -2.17108927e-01,  2.74588478e+02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ])
    
    cam_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    
    # 이미지 캡처
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    
    return np.asarray(image)


def visualize_interactive(pcd_no_mask, pcd_with_mask):
    """대화형 3D 뷰어로 두 포인트 클라우드를 별도 창에서 비교"""
    print("\n=== 대화형 3D 뷰어 ===")
    print("조작 방법:")
    print("  - 마우스 왼쪽 버튼 드래그: 회전")
    print("  - 마우스 휠: 확대/축소")
    print("  - 마우스 오른쪽 버튼 드래그: 이동")
    print("  - 'Q' 또는 창 닫기: 다음 창으로 이동")
    print(f"\n두 모델을 별도 창에서 동일한 시점으로 표시합니다:")
    print(f"  - 첫 번째 창(빨간색): 마스크 없음 ({len(pcd_no_mask.points):,}개 포인트)")
    print(f"  - 두 번째 창(초록색): 마스크 적용 ({len(pcd_with_mask.points):,}개 포인트)")
    print("\n카메라 파라미터가 5초마다 자동으로 출력됩니다...")
    
    # 공통 카메라 파라미터
    def setup_camera(vis):
        """공통 카메라 설정 - 사용자 지정 Extrinsic Matrix"""
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # 검은색 배경
        opt.point_size = 3.0
        
        ctr = vis.get_view_control()
        
        # 사용자가 제공한 Extrinsic Matrix 적용
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        
        # Extrinsic Matrix 설정
        extrinsic = np.array([
            [ 2.17081021e-01, -8.23744991e-03,  9.76118832e-01, -4.65365706e+01],
            [-7.14191300e-02, -9.97418450e-01,  7.46585000e-03, -1.35329047e+00],
            [ 9.73537436e-01, -7.13342512e-02, -2.17108927e-01,  2.74588478e+02],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ])
        
        cam_params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    
    def print_camera_params(vis, title=""):
        """현재 카메라 파라미터 출력"""
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        
        print(f"\n{title} 카메라 파라미터:")
        print(f"  Field of View: {ctr.get_field_of_view():.2f}°")
        
        # Extrinsic matrix에서 카메라 정보 추출
        extrinsic = cam_params.extrinsic
        
        # 카메라 위치 (Translation vector)
        camera_pos = extrinsic[:3, 3]
        print(f"  Camera Position: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}]")
        
        # 회전 행렬에서 방향 벡터 추출
        rotation = extrinsic[:3, :3]
        front_vector = -rotation[:, 2]  # Z축의 음수 방향이 카메라가 보는 방향
        up_vector = -rotation[:, 1]     # Y축의 음수 방향이 위쪽
        
        print(f"  Front Vector: [{front_vector[0]:.4f}, {front_vector[1]:.4f}, {front_vector[2]:.4f}]")
        print(f"  Up Vector: [{up_vector[0]:.4f}, {up_vector[1]:.4f}, {up_vector[2]:.4f}]")
        
        # Intrinsic 파라미터
        intrinsic = cam_params.intrinsic
        print(f"  Intrinsic Matrix (focal length & principal point):")
        print(f"    fx={intrinsic.intrinsic_matrix[0,0]:.2f}, fy={intrinsic.intrinsic_matrix[1,1]:.2f}")
        print(f"    cx={intrinsic.intrinsic_matrix[0,2]:.2f}, cy={intrinsic.intrinsic_matrix[1,2]:.2f}")
        
        print(f"  Extrinsic Matrix (4x4):")
        for i in range(4):
            print(f"    {extrinsic[i]}")
    
    import time
    
    def run_with_periodic_output(vis, title):
        """5초마다 카메라 파라미터를 출력하면서 뷰어 실행"""
        last_print_time = time.time()
        
        def update_callback(vis):
            nonlocal last_print_time
            current_time = time.time()
            
            # 5초마다 카메라 파라미터 출력
            if current_time - last_print_time >= 5.0:
                print_camera_params(vis, f"[{time.strftime('%H:%M:%S')}] {title}")
                last_print_time = current_time
            
            return False
        
        vis.register_animation_callback(update_callback)
        vis.run()
    
    # 1. 마스크 없는 포인트 클라우드 (좌측 창)
    print("\n[1/2] 마스크 없음 창 표시 중... (빨간색)")
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(
        window_name="(1) Without Mask - Raw Depth Data",
        width=1024,
        height=768,
        left=50,
        top=50
    )
    vis1.add_geometry(pcd_no_mask)
    setup_camera(vis1)
    
    # 초기 카메라 파라미터 출력
    print_camera_params(vis1, "[초기] Without Mask")
    
    # 5초마다 카메라 파라미터 출력하면서 실행
    run_with_periodic_output(vis1, "Without Mask")
    
    # 최종 카메라 파라미터 출력
    print_camera_params(vis1, "[최종] Without Mask")
    
    vis1.destroy_window()
    
    # 2. 마스크 적용 포인트 클라우드 (우측 창)
    print("\n[2/2] 마스크 적용 창 표시 중... (초록색)")
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(
        window_name="(2) With Mask - Morphological Operations",
        width=1024,
        height=768,
        left=1074,  # 첫 번째 창 오른쪽에 배치
        top=50
    )
    vis2.add_geometry(pcd_with_mask)
    setup_camera(vis2)
    
    # 초기 카메라 파라미터 출력
    print_camera_params(vis2, "[초기] With Mask")
    
    # 5초마다 카메라 파라미터 출력하면서 실행
    run_with_periodic_output(vis2, "With Mask")
    
    # 최종 카메라 파라미터 출력
    print_camera_params(vis2, "[최종] With Mask")
    
    vis2.destroy_window()


def main():
    # 테스트 이미지 경로
    depth_map_path = r"D:\Lab2\--final_3D_Body--\3D_Body_Posture_Analysis\test2\DepthMap\남_정면\DepthMap0.bmp"
    
    print("="*60)
    print("논문용 이미지 생성 #1: 마스크 적용 전후 비교")
    print("="*60)
    
    # 깊이맵 로드
    print("\n깊이맵 로딩 중...")
    depth_map = load_depth_map(depth_map_path)
    
    if depth_map is None:
        print("깊이맵 로드 실패!")
        return
    
    # 1. 마스크 없는 포인트 클라우드 생성
    print("마스크 없이 포인트 클라우드 생성 중...")
    pcd_no_mask = create_pointcloud_no_mask(depth_map, "front")
    print(f"  포인트 수: {len(pcd_no_mask.points):,}개 (노이즈 포함)")
    
    # 2. 마스크 적용 포인트 클라우드 생성
    print("\n마스크 적용하여 포인트 클라우드 생성 중...")
    pcd_with_mask = create_pointcloud_with_mask(depth_map, "front")
    print(f"  포인트 수: {len(pcd_with_mask.points):,}개 (깨끗한 결과)")
    
    # 통계 정보
    reduction_ratio = (len(pcd_no_mask.points) - len(pcd_with_mask.points)) / len(pcd_no_mask.points) * 100
    print(f"\n노이즈 제거율: {reduction_ratio:.1f}%")
    
    # 3. 대화형 3D 뷰어로 시각화
    visualize_interactive(pcd_no_mask, pcd_with_mask)
    
    # 4. 이미지 캡처 (논문용)
    print("\n논문용 이미지 캡처 중...")
    img_no_mask = capture_pointcloud_image(pcd_no_mask, "Without Mask")
    img_with_mask = capture_pointcloud_image(pcd_with_mask, "With Mask")
    
    # 5. 비교 이미지 생성 (간단하게 2개만)
    print("비교 이미지 생성 중...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 마스크 없는 결과
    axes[0].imshow(img_no_mask)
    axes[0].set_title('(a) Without Mask\n(Raw Depth Data)', fontsize=14, fontweight='bold')
    axes[0].text(0.5, -0.08, f'Points: {len(pcd_no_mask.points):,}\nIncludes noise and background', 
                 ha='center', va='top', transform=axes[0].transAxes, fontsize=11, 
                 color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].axis('off')
    
    # 마스크 적용 결과
    axes[1].imshow(img_with_mask)
    axes[1].set_title('(b) With Mask\n(Morphological Operations)', fontsize=14, fontweight='bold')
    axes[1].text(0.5, -0.08, f'Points: {len(pcd_with_mask.points):,}\nClean body region only\nNoise reduction: {reduction_ratio:.1f}%', 
                 ha='center', va='top', transform=axes[1].transAxes, fontsize=11, 
                 color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].axis('off')
    
    plt.suptitle('Point Cloud Quality Comparison: Mask Application Effect', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, 'output_1_mask_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n결과 이미지 저장: {output_path}")
    
    print("\n완료!")
    plt.show()


if __name__ == "__main__":
    main()
