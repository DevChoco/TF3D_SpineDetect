"""
메시 생성 및 저장 모듈

이 모듈은 다음 기능을 제공합니다:
- 포인트 클라우드에서 메시 생성
- Poisson 표면 재구성
- Ball Pivoting Algorithm
- 메시 최적화 및 후처리
- 다양한 형식으로 저장 (OBJ, PLY, STL)
"""

import numpy as np
import open3d as o3d
import os


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


def optimize_mesh(mesh):
    """
    메시를 최적화합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 최적화할 메시
        
    Returns:
        o3d.geometry.TriangleMesh: 최적화된 메시
    """
    if mesh is None:
        return None
    
    try:
        # 메시 단순화 (삼각형 수 감소)
        mesh_optimized = mesh.simplify_quadric_decimation(target_number_of_triangles=len(mesh.triangles))
        
        # 불필요한 요소 제거
        mesh_optimized.remove_degenerate_triangles()
        mesh_optimized.remove_duplicated_triangles()
        mesh_optimized.remove_duplicated_vertices()
        mesh_optimized.remove_non_manifold_edges()
        
        # 법선 벡터 재계산
        mesh_optimized.compute_vertex_normals()
        mesh_optimized.compute_triangle_normals()
        
        return mesh_optimized
    except Exception as e:
        print(f"메시 최적화 중 오류: {e}")
        return mesh


def save_mesh(mesh, output_dir, base_filename="body_mesh"):
    """
    메시를 다양한 형식으로 저장합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 저장할 메시
        output_dir (str): 출력 디렉토리
        base_filename (str): 기본 파일명
        
    Returns:
        list: 저장된 파일 경로 리스트
    """
    if mesh is None:
        print("저장할 메시가 없습니다.")
        return []
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # OBJ 형식으로 저장
    try:
        obj_path = os.path.join(output_dir, f"{base_filename}.obj")
        success = o3d.io.write_triangle_mesh(obj_path, mesh)
        if success:
            print(f"메시가 저장되었습니다 (OBJ): {obj_path}")
            saved_files.append(obj_path)
        else:
            print(f"OBJ 저장 실패: {obj_path}")
    except Exception as e:
        print(f"OBJ 저장 중 오류: {e}")
    
    # PLY 형식으로 저장
    try:
        ply_path = os.path.join(output_dir, f"{base_filename}.ply")
        success = o3d.io.write_triangle_mesh(ply_path, mesh)
        if success:
            print(f"메시가 저장되었습니다 (PLY): {ply_path}")
            saved_files.append(ply_path)
        else:
            print(f"PLY 저장 실패: {ply_path}")
    except Exception as e:
        print(f"PLY 저장 중 오류: {e}")
    
    # STL 형식으로 저장 (3D 프린팅 및 CAD용)
    try:
        stl_path = os.path.join(output_dir, f"{base_filename}.stl")
        
        # STL 저장을 위한 메시 최적화
        mesh_for_stl = optimize_mesh(mesh)
        
        if mesh_for_stl is not None:
            success = o3d.io.write_triangle_mesh(stl_path, mesh_for_stl)
            if success:
                print(f"메시가 저장되었습니다 (STL): {stl_path}")
                print(f"  STL 메시 정보: {len(mesh_for_stl.vertices)}개 정점, {len(mesh_for_stl.triangles)}개 삼각형")
                saved_files.append(stl_path)
            else:
                print(f"STL 저장 실패: {stl_path}")
        else:
            # 최적화 없이 원본 메시로 다시 시도
            success = o3d.io.write_triangle_mesh(stl_path, mesh)
            if success:
                print(f"STL 저장 성공 (최적화 없음): {stl_path}")
                saved_files.append(stl_path)
            else:
                print(f"STL 저장 완전 실패: {stl_path}")
    except Exception as e:
        print(f"STL 저장 중 오류: {e}")
    
    return saved_files


def create_and_save_mesh(pcd, output_dir="output/3d_models", base_filename="body_mesh"):
    """
    포인트 클라우드에서 메시를 생성하고 저장합니다.
    
    Args:
        pcd (o3d.geometry.PointCloud): 포인트 클라우드
        output_dir (str): 출력 디렉토리
        base_filename (str): 기본 파일명
        
    Returns:
        tuple: (mesh object, saved file paths list)
    """
    print("\n포인트 클라우드를 메시로 변환 중...")
    mesh = create_mesh_from_pointcloud(pcd)
    
    saved_files = []
    if mesh is not None:
        saved_files = save_mesh(mesh, output_dir, base_filename)
    
    return mesh, saved_files