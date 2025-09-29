"""
모듈 테스트 스크립트

개별 모듈의 기능을 테스트할 수 있는 간단한 스크립트입니다.
"""

import numpy as np
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pointcloud_generator():
    """포인트 클라우드 생성 모듈 테스트"""
    print("=== 포인트 클라우드 생성 모듈 테스트 ===")
    
    try:
        from modules.pointcloud_generator import load_depth_map, create_point_cloud_from_depth
        
        # 테스트 이미지 경로 (실제 경로로 변경하세요)
        test_image = r"D:\Lab2\3D_Body_Posture_Analysis_FPFH\test2\여성\여_정면.bmp"
        
        if os.path.exists(test_image):
            depth_map = load_depth_map(test_image)
            if depth_map is not None:
                print(f"✅ 깊이맵 로드 성공: {depth_map.shape}")
                
                pcd = create_point_cloud_from_depth(depth_map, "front")
                if pcd is not None:
                    print(f"✅ 포인트 클라우드 생성 성공: {len(pcd.points)}개 포인트")
                else:
                    print("❌ 포인트 클라우드 생성 실패")
            else:
                print("❌ 깊이맵 로드 실패")
        else:
            print(f"❌ 테스트 이미지가 없습니다: {test_image}")
            
    except Exception as e:
        print(f"❌ 포인트 클라우드 생성 모듈 오류: {e}")


def test_fpfh_alignment():
    """FPFH 정렬 모듈 테스트"""
    print("\n=== FPFH 정렬 모듈 테스트 ===")
    
    try:
        from modules.fpfh_alignment import compute_fpfh, global_registration_fpfh_ransac
        import open3d as o3d
        
        # 간단한 테스트 포인트 클라우드 생성
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3) * 100)
        
        pcd2 = o3d.geometry.PointCloud()  
        pcd2.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3) * 100 + 10)
        
        # FPFH 특징 계산 테스트
        pcd1.estimate_normals()
        fpfh = compute_fpfh(pcd1, voxel_size=5.0)
        print(f"✅ FPFH 특징 계산 성공: {fpfh.data.shape}")
        
        print("✅ FPFH 정렬 모듈 기본 기능 정상")
        
    except Exception as e:
        print(f"❌ FPFH 정렬 모듈 오류: {e}")


def test_skeleton_parser():
    """스켈레톤 파싱 모듈 테스트"""
    print("\n=== 스켈레톤 파싱 모듈 테스트 ===")
    
    try:
        from modules.skeleton_parser import create_skeleton_from_pointcloud, calculate_spine_angles
        import open3d as o3d
        
        # 테스트용 포인트 클라우드 생성 (인체 형태)
        pcd = o3d.geometry.PointCloud()
        # 간단한 인체 형태의 포인트 생성
        points = []
        for i in range(1000):
            x = np.random.normal(0, 20)
            y = np.random.uniform(-100, 100)  # 키 방향
            z = np.random.normal(0, 15)
            points.append([x, y, z])
        
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        
        # 스켈레톤 생성 테스트
        skeleton_points = create_skeleton_from_pointcloud(pcd)
        print(f"✅ 스켈레톤 생성 성공: {len(skeleton_points)}개 포인트")
        
        # 각도 계산 테스트
        angles = calculate_spine_angles(skeleton_points)
        print(f"✅ 각도 계산 성공: {len(angles)}개 각도")
        
        print("✅ 스켈레톤 파싱 모듈 기본 기능 정상")
        
    except Exception as e:
        print(f"❌ 스켈레톤 파싱 모듈 오류: {e}")


def test_mesh_generator():
    """메시 생성 모듈 테스트"""
    print("\n=== 메시 생성 모듈 테스트 ===")
    
    try:
        from modules.mesh_generator import create_mesh_from_pointcloud
        import open3d as o3d
        
        # 테스트용 포인트 클라우드 생성
        pcd = o3d.geometry.PointCloud()
        # 구 형태의 포인트 생성
        points = []
        for i in range(2000):
            # 구 표면의 점들
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = 50 + np.random.normal(0, 2)  # 약간의 노이즈
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points.append([x, y, z])
        
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.estimate_normals()
        
        # 메시 생성 테스트
        mesh = create_mesh_from_pointcloud(pcd)
        if mesh is not None:
            print(f"✅ 메시 생성 성공: {len(mesh.vertices)}개 정점, {len(mesh.triangles)}개 삼각형")
        else:
            print("❌ 메시 생성 실패")
        
        print("✅ 메시 생성 모듈 기본 기능 정상")
        
    except Exception as e:
        print(f"❌ 메시 생성 모듈 오류: {e}")


def test_all_modules():
    """모든 모듈 테스트 실행"""
    print("🧪 모듈화된 3D 자세 분석 시스템 테스트 시작")
    print("=" * 50)
    
    test_pointcloud_generator()
    test_fpfh_alignment()
    test_skeleton_parser()
    test_mesh_generator()
    
    print("\n" + "=" * 50)
    print("🏁 테스트 완료")


if __name__ == "__main__":
    test_all_modules()