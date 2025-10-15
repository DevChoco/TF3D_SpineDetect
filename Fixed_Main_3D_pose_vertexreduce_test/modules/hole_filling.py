"""
메시 홀 채우기 및 누락 영역 보완 모듈

뎁스 이미지의 한계로 인한 누락 영역을 지능적으로 채우는 기능을 제공합니다:
- 메시 홀 자동 감지 및 채우기
- 대칭성 기반 누락 영역 복원
- 포아송 재구성 기반 표면 보간
- 형태학적 연산을 통한 스무딩
- 인체 해부학적 지식 기반 보정
"""

import numpy as np
import open3d as o3d
import copy
import cv2


def detect_mesh_holes(mesh, hole_size_threshold=50):
    """
    메시에서 홀(구멍)을 감지합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        hole_size_threshold (int): 홀로 간주할 최소 크기
        
    Returns:
        list: 감지된 홀들의 경계 정보
    """
    print("\n=== 메시 홀 감지 ===")
    
    if mesh is None or len(mesh.triangles) == 0:
        return []
    
    try:
        # 경계 엣지 찾기
        mesh_copy = copy.deepcopy(mesh)
        boundary_edges = mesh_copy.get_non_manifold_edges()
        
        if len(boundary_edges) == 0:
            print("매니폴드 메시입니다. 홀이 감지되지 않았습니다.")
            return []
        
        print(f"매니폴드가 아닌 엣지 {len(boundary_edges)}개 감지")
        
        # 홀 크기별 분류
        holes_info = []
        for i, edge in enumerate(boundary_edges):
            if len(edge) >= hole_size_threshold:
                holes_info.append({
                    'id': i,
                    'boundary_length': len(edge),
                    'vertices': edge
                })
        
        print(f"크기가 충분한 홀 {len(holes_info)}개 감지")
        return holes_info
        
    except Exception as e:
        print(f"홀 감지 중 오류: {e}")
        return []


def fill_holes_poisson(mesh, hole_info_list):
    """
    포아송 재구성을 사용하여 홀을 채웁니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        hole_info_list (list): 홀 정보 리스트
        
    Returns:
        o3d.geometry.TriangleMesh: 홀이 채워진 메시
    """
    if not hole_info_list:
        return mesh
    
    print(f"\n=== 포아송 홀 채우기 ({len(hole_info_list)}개 홀) ===")
    
    try:
        # 메시를 포인트 클라우드로 변환
        vertices = np.asarray(mesh.vertices)
        
        # 샘플링을 통해 포인트 클라우드 생성
        pcd = mesh.sample_points_uniformly(number_of_points=len(vertices) * 2)
        
        # 법선 벡터 계산
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # 포아송 재구성으로 홀 채우기
        filled_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=10,  # 높은 해상도로 재구성
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # 밀도가 낮은 부분 제거
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.05)  # 더 보수적
        filled_mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"홀 채우기 완료: {len(filled_mesh.vertices)}개 버텍스, {len(filled_mesh.triangles)}개 삼각형")
        
        return filled_mesh
        
    except Exception as e:
        print(f"포아송 홀 채우기 중 오류: {e}")
        return mesh


def fill_holes_morphological(mesh, iterations=3):
    """
    형태학적 연산을 사용하여 작은 홀들을 채웁니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        iterations (int): 반복 횟수
        
    Returns:
        o3d.geometry.TriangleMesh: 홀이 채워진 메시
    """
    print(f"\n=== 형태학적 홀 채우기 (반복: {iterations}회) ===")
    
    try:
        filled_mesh = copy.deepcopy(mesh)
        
        for i in range(iterations):
            # 라플라시안 스무딩으로 표면 보간
            filled_mesh = filled_mesh.filter_smooth_laplacian(number_of_iterations=1)
            
            # 중복 버텍스 및 삼각형 제거
            filled_mesh.remove_duplicated_vertices()
            filled_mesh.remove_duplicated_triangles()
            filled_mesh.remove_degenerate_triangles()
            
            print(f"  반복 {i+1}: {len(filled_mesh.vertices)}개 버텍스")
        
        # 법선 벡터 재계산
        filled_mesh.compute_vertex_normals()
        filled_mesh.compute_triangle_normals()
        
        print("형태학적 홀 채우기 완료")
        return filled_mesh
        
    except Exception as e:
        print(f"형태학적 홀 채우기 중 오류: {e}")
        return mesh


def detect_symmetry_axis(pcd):
    """
    포인트 클라우드에서 대칭축을 감지합니다.
    
    Args:
        pcd (o3d.geometry.PointCloud): 포인트 클라우드
        
    Returns:
        np.array: 대칭축 방향 벡터 (없으면 None)
    """
    try:
        points = np.asarray(pcd.points)
        
        # 유효한 포인트 확인
        if len(points) == 0:
            return np.array([1, 0, 0])  # 기본값
        
        # NaN 또는 무한대 값 제거
        valid_mask = np.isfinite(points).all(axis=1)
        if not np.any(valid_mask):
            return np.array([1, 0, 0])  # 기본값
        
        points = points[valid_mask]
        
        # 중심점 계산
        center = np.mean(points, axis=0)
        
        # NaN 체크
        if not np.isfinite(center).all():
            return np.array([1, 0, 0])  # 기본값
        
        # PCA를 통한 주축 찾기
        centered_points = points - center
        
        try:
            covariance_matrix = np.cov(centered_points.T)
            
            # 공분산 행렬이 유효한지 확인
            if not np.isfinite(covariance_matrix).all():
                return np.array([1, 0, 0])  # 기본값
            
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            
            # 고유값과 고유벡터가 유효한지 확인
            if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
                return np.array([1, 0, 0])  # 기본값
            
            # 가장 작은 고유값에 해당하는 벡터가 대칭축
            symmetry_axis = eigenvectors[:, np.argmin(eigenvalues)]
            
            # 인체는 보통 X축 대칭 (좌우 대칭)
            return np.array([1, 0, 0])
            
        except np.linalg.LinAlgError:
            return np.array([1, 0, 0])  # 기본값
            
    except Exception as e:
        print(f"대칭축 감지 중 오류: {e}")
        return np.array([1, 0, 0])  # 기본값


def mirror_fill_missing_regions(mesh, symmetry_axis=None):
    """
    대칭성을 이용하여 누락된 영역을 채웁니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        symmetry_axis (np.array): 대칭축 (None이면 자동 감지)
        
    Returns:
        o3d.geometry.TriangleMesh: 대칭 복원된 메시
    """
    print("\n=== 대칭성 기반 누락 영역 복원 ===")
    
    try:
        if mesh is None or len(mesh.vertices) == 0:
            return mesh
        
        vertices = np.asarray(mesh.vertices)
        
        # NaN 또는 무한대 값 체크 및 제거
        valid_mask = np.isfinite(vertices).all(axis=1)
        if not np.any(valid_mask):
            print("유효하지 않은 버텍스 데이터입니다.")
            return mesh
        
        if not valid_mask.all():
            print(f"무효한 버텍스 {np.sum(~valid_mask)}개 제거")
            vertices = vertices[valid_mask]
            
            # 메시 재구성
            valid_indices = np.where(valid_mask)[0]
            triangles = np.asarray(mesh.triangles)
            
            # 유효한 버텍스만 포함하는 삼각형 필터링
            valid_triangles = []
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
            
            for triangle in triangles:
                if all(idx in index_map for idx in triangle):
                    new_triangle = [index_map[idx] for idx in triangle]
                    valid_triangles.append(new_triangle)
            
            # 새 메시 생성
            clean_mesh = o3d.geometry.TriangleMesh()
            clean_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            clean_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
            mesh = clean_mesh
        
        # 메시 중심점 계산
        center = np.mean(vertices, axis=0)
        
        # NaN 체크
        if not np.isfinite(center).all():
            print("중심점 계산 실패, 대칭 복원을 건너뜁니다.")
            return mesh
        
        print(f"메시 중심점: {center}")
        
        # 대칭축 설정 (기본값: X축 대칭)
        if symmetry_axis is None:
            symmetry_axis = np.array([1, 0, 0])  # 항상 X축 대칭 (좌우 대칭)
        
        print(f"감지된 대칭축: {symmetry_axis}")
        
        # X축 대칭 (좌우 대칭) 적용
        print("X축 대칭 (좌우 대칭) 적용")
        
        # 중심을 기준으로 좌측과 우측 분리
        left_mask = vertices[:, 0] < center[0]
        right_mask = vertices[:, 0] > center[0]
        
        left_vertices = vertices[left_mask]
        right_vertices = vertices[right_mask]
        
        print(f"좌측 버텍스: {len(left_vertices)}개, 우측 버텍스: {len(right_vertices)}개")
        
        # 좌우 차이가 심하지 않으면 그대로 반환
        total_vertices = len(vertices)
        left_ratio = len(left_vertices) / total_vertices
        right_ratio = len(right_vertices) / total_vertices
        
        if abs(left_ratio - right_ratio) < 0.1:  # 10% 이하 차이
            print("좌우 균형이 적절하여 대칭 복원을 건너뜁니다.")
            return mesh
        
        # 데이터가 충분한 쪽을 기준으로 미러링
        if len(left_vertices) > len(right_vertices) * 1.2:  # 좌측이 20% 이상 많음
            print("좌측을 기준으로 우측 복원")
            source_vertices = left_vertices
        elif len(right_vertices) > len(left_vertices) * 1.2:  # 우측이 20% 이상 많음
            print("우측을 기준으로 좌측 복원")
            source_vertices = right_vertices
        else:
            print("좌우 차이가 크지 않아 대칭 복원을 건너뜁니다.")
            return mesh
        
        # 미러링할 포인트 수를 제한 (과도한 증가 방지)
        max_mirror_points = min(len(source_vertices), total_vertices // 2)
        if len(source_vertices) > max_mirror_points:
            # 랜덤 샘플링으로 포인트 수 제한
            indices = np.random.choice(len(source_vertices), max_mirror_points, replace=False)
            source_vertices = source_vertices[indices]
        
        # 미러링 변환 (X축 기준 반사)
        mirrored_vertices = source_vertices - center
        mirrored_vertices[:, 0] = -mirrored_vertices[:, 0]  # X좌표만 반전
        mirrored_vertices = mirrored_vertices + center
        
        # 기존 버텍스와 미러링된 버텍스 합치기
        all_vertices = np.vstack([vertices, mirrored_vertices])
        
        print(f"미러링 후 총 버텍스: {len(all_vertices)}개")
        
        # 새로운 포인트 클라우드 생성
        enhanced_pcd = o3d.geometry.PointCloud()
        enhanced_pcd.points = o3d.utility.Vector3dVector(all_vertices)
        
        # 중복 포인트 제거 
        enhanced_pcd = enhanced_pcd.remove_duplicated_points()
        
        # 법선 벡터 계산
        enhanced_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        enhanced_pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # 포아송 재구성으로 새로운 메시 생성
        enhanced_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            enhanced_pcd, 
            depth=8,  # 깊이를 낮춰서 안정성 향상
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # 밀도 필터링
        densities = np.asarray(densities)
        if len(densities) > 0:
            vertices_to_remove = densities < np.quantile(densities, 0.15)  # 더 보수적
            enhanced_mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"대칭 복원 완료: {len(enhanced_mesh.vertices)}개 버텍스, {len(enhanced_mesh.triangles)}개 삼각형")
        
        return enhanced_mesh
        
    except Exception as e:
        print(f"대칭 복원 중 오류: {e}")
        return mesh


def anatomical_hole_filling(mesh, body_region="torso"):
    """
    인체 해부학적 지식을 기반으로 누락 영역을 채웁니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        body_region (str): 신체 부위 ("torso", "limbs", "full")
        
    Returns:
        o3d.geometry.TriangleMesh: 해부학적으로 보정된 메시
    """
    print(f"\n=== 해부학적 홀 채우기 ({body_region}) ===")
    
    try:
        if mesh is None or len(mesh.vertices) == 0:
            return mesh
        
        vertices = np.asarray(mesh.vertices)
        
        # NaN 값 체크
        valid_mask = np.isfinite(vertices).all(axis=1)
        if not valid_mask.all():
            print(f"무효한 버텍스 {np.sum(~valid_mask)}개 제거")
            vertices = vertices[valid_mask]
        
        if len(vertices) == 0:
            print("유효한 버텍스가 없습니다.")
            return mesh
        
        # 인체 비율에 따른 보정
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        height = extent[1]  # Y축 높이
        width = extent[0]   # X축 너비
        depth = extent[2]   # Z축 깊이
        
        print(f"메시 크기 - 높이: {height:.1f}, 너비: {width:.1f}, 깊이: {depth:.1f}")
        
        # 현재 밀도 계산
        current_density = len(vertices)
        
        # 인체 비율 기반 목표 밀도 조정
        if body_region == "torso":
            target_density = max(current_density, int(current_density * 1.2))
        elif body_region == "limbs":
            target_density = max(current_density, int(current_density * 1.1))
        else:  # full
            target_density = max(current_density, int(current_density * 1.15))
        
        # 목표 밀도가 현재보다 너무 크면 제한
        max_density = current_density * 1.5
        target_density = min(target_density, max_density)
        
        # 포인트 클라우드로 변환 후 균등 샘플링
        try:
            pcd = mesh.sample_points_uniformly(number_of_points=int(target_density))
        except:
            # 샘플링 실패 시 현재 버텍스 사용
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
        
        points = np.asarray(pcd.points)
        
        # 해부학적 제약 조건 적용
        if len(points) == 0:
            return mesh
        
        # 인체 중심선 기준으로 대칭성 강화
        center_x = np.mean(points[:, 0])
        
        # NaN 체크
        if not np.isfinite(center_x):
            center_x = 0
        
        # 좌우 불균형 보정
        left_points = points[points[:, 0] < center_x]
        right_points = points[points[:, 0] > center_x]
        
        print(f"좌우 분포 - 좌측: {len(left_points)}, 우측: {len(right_points)}")
        
        # 좌우 차이가 크지 않으면 보정하지 않음
        total_points = len(points)
        if total_points == 0:
            return mesh
        
        left_ratio = len(left_points) / total_points
        right_ratio = len(right_points) / total_points
        
        if abs(left_ratio - right_ratio) < 0.15:  # 15% 이하 차이
            print("좌우 균형이 적절합니다.")
            enhanced_pcd = pcd
        else:
            print(f"좌우 불균형 감지 - 좌측: {len(left_points)}, 우측: {len(right_points)}")
            
            # 부족한 쪽에 적당한 수의 대칭 포인트 추가
            max_additional = min(abs(len(left_points) - len(right_points)), total_points // 4)
            
            if len(left_points) < len(right_points) and max_additional > 0:
                # 우측을 좌측으로 미러링
                sample_count = min(max_additional, len(right_points))
                sample_indices = np.random.choice(len(right_points), sample_count, replace=False)
                sample_points = right_points[sample_indices]
                
                # X좌표만 반전
                mirrored_points = sample_points.copy()
                mirrored_points[:, 0] = 2 * center_x - sample_points[:, 0]
                
                points = np.vstack([points, mirrored_points])
                
            elif len(right_points) < len(left_points) and max_additional > 0:
                # 좌측을 우측으로 미러링
                sample_count = min(max_additional, len(left_points))
                sample_indices = np.random.choice(len(left_points), sample_count, replace=False)
                sample_points = left_points[sample_indices]
                
                # X좌표만 반전
                mirrored_points = sample_points.copy()
                mirrored_points[:, 0] = 2 * center_x - sample_points[:, 0]
                
                points = np.vstack([points, mirrored_points])
            
            # 보정된 포인트 클라우드 생성
            enhanced_pcd = o3d.geometry.PointCloud()
            enhanced_pcd.points = o3d.utility.Vector3dVector(points)
        
        # 중복 포인트 제거
        enhanced_pcd = enhanced_pcd.remove_duplicated_points()
        
        # 법선 벡터 계산
        enhanced_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        enhanced_pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # 포아송 재구성
        anatomical_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            enhanced_pcd, 
            depth=8,  # 안정성을 위해 깊이 감소
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # 밀도 필터링
        densities = np.asarray(densities)
        if len(densities) > 0:
            vertices_to_remove = densities < np.quantile(densities, 0.12)  # 더 관대한 필터링
            anatomical_mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"해부학적 보정 완료: {len(anatomical_mesh.vertices)}개 버텍스")
        
        return anatomical_mesh
        
    except Exception as e:
        print(f"해부학적 홀 채우기 중 오류: {e}")
        return mesh


def advanced_hole_filling(mesh, method="comprehensive"):
    """
    고급 홀 채우기 파이프라인을 실행합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        method (str): 채우기 방법 ("poisson", "morphological", "symmetry", "anatomical", 
                     "comprehensive", "large_gaps", "bilateral_symmetry")
        
    Returns:
        o3d.geometry.TriangleMesh: 홀이 채워진 메시
    """
    print(f"\n=== 고급 홀 채우기 파이프라인 (방법: {method}) ===")
    
    if mesh is None:
        return None
    
    original_vertices = len(mesh.vertices)
    original_triangles = len(mesh.triangles)
    
    print(f"원본 메시: {original_vertices:,}개 버텍스, {original_triangles:,}개 삼각형")
    
    try:
        if method == "comprehensive":
            # 종합적 접근: 여러 방법을 순차적으로 적용
            
            # 1단계: 홀 감지
            holes = detect_mesh_holes(mesh, hole_size_threshold=30)
            
            # 2단계: 큰 구멍 지능적 채우기 (NEW - 옆구리, 팔 안쪽 등)
            print("\n🔄 1단계: 큰 구멍 지능적 채우기 (옆구리, 팔 안쪽)")
            enhanced_mesh = fill_large_gaps_intelligently(mesh, gap_threshold=40)
            
            # 3단계: 양측 대칭성 완성 (NEW - 좌우 불균형 보정)
            print("\n🔄 2단계: 양측 대칭성 기반 완성")
            enhanced_mesh = bilateral_symmetry_completion(enhanced_mesh)
            
            # 4단계: 대칭성 기반 복원 (기존)
            print("\n🔄 3단계: 대칭성 기반 일반 복원")
            enhanced_mesh = mirror_fill_missing_regions(enhanced_mesh)
            
            # 5단계: 해부학적 보정
            print("\n🔄 4단계: 해부학적 지식 기반 보정")
            enhanced_mesh = anatomical_hole_filling(enhanced_mesh, body_region="full")
            
            # 6단계: 잔여 홀 포아송 채우기
            if holes:
                print("\n🔄 5단계: 잔여 홀 포아송 채우기")
                enhanced_mesh = fill_holes_poisson(enhanced_mesh, holes)
            
            # 7단계: 형태학적 스무딩
            print("\n🔄 6단계: 형태학적 스무딩")
            enhanced_mesh = fill_holes_morphological(enhanced_mesh, iterations=2)
            
        elif method == "large_gaps":
            # 큰 구멍 전용 처리
            enhanced_mesh = fill_large_gaps_intelligently(mesh)
            enhanced_mesh = bilateral_symmetry_completion(enhanced_mesh)
            
        elif method == "bilateral_symmetry":
            # 대칭성 전용 처리
            enhanced_mesh = bilateral_symmetry_completion(mesh)
            
        elif method == "poisson":
            holes = detect_mesh_holes(mesh)
            enhanced_mesh = fill_holes_poisson(mesh, holes)
            
        elif method == "morphological":
            enhanced_mesh = fill_holes_morphological(mesh)
            
        elif method == "symmetry":
            enhanced_mesh = mirror_fill_missing_regions(mesh)
            
        elif method == "anatomical":
            enhanced_mesh = anatomical_hole_filling(mesh)
            
        else:
            print(f"알 수 없는 방법: {method}")
            return mesh
        
        # 최종 정리
        enhanced_mesh.remove_degenerate_triangles()
        enhanced_mesh.remove_duplicated_triangles()
        enhanced_mesh.remove_duplicated_vertices()
        enhanced_mesh.compute_vertex_normals()
        enhanced_mesh.compute_triangle_normals()
        
        final_vertices = len(enhanced_mesh.vertices)
        final_triangles = len(enhanced_mesh.triangles)
        
        vertex_increase = ((final_vertices - original_vertices) / original_vertices) * 100
        triangle_increase = ((final_triangles - original_triangles) / original_triangles) * 100
        
        print(f"\n✅ 홀 채우기 완료:")
        print(f"  최종 메시: {final_vertices:,}개 버텍스, {final_triangles:,}개 삼각형")
        print(f"  증가율: 버텍스 {vertex_increase:+.1f}%, 삼각형 {triangle_increase:+.1f}%")
        
        return enhanced_mesh
        
    except Exception as e:
        print(f"고급 홀 채우기 중 오류: {e}")
        return mesh


def compare_before_after(original_mesh, filled_mesh):
    """
    홀 채우기 전후를 비교 분석합니다.
    
    Args:
        original_mesh (o3d.geometry.TriangleMesh): 원본 메시
        filled_mesh (o3d.geometry.TriangleMesh): 채워진 메시
        
    Returns:
        dict: 비교 분석 결과
    """
    if original_mesh is None or filled_mesh is None:
        return {}
    
    try:
        analysis = {
            'original_vertices': len(original_mesh.vertices),
            'filled_vertices': len(filled_mesh.vertices),
            'original_triangles': len(original_mesh.triangles),
            'filled_triangles': len(filled_mesh.triangles),
            'original_surface_area': original_mesh.get_surface_area(),
            'filled_surface_area': filled_mesh.get_surface_area(),
        }
        
        analysis['vertex_increase_percent'] = ((analysis['filled_vertices'] - analysis['original_vertices']) / analysis['original_vertices']) * 100
        analysis['triangle_increase_percent'] = ((analysis['filled_triangles'] - analysis['original_triangles']) / analysis['original_triangles']) * 100
        analysis['surface_area_increase_percent'] = ((analysis['filled_surface_area'] - analysis['original_surface_area']) / analysis['original_surface_area']) * 100
        
        # 부피 비교 (watertight 메시인 경우)
        if original_mesh.is_watertight() and filled_mesh.is_watertight():
            analysis['original_volume'] = original_mesh.get_volume()
            analysis['filled_volume'] = filled_mesh.get_volume()
            analysis['volume_increase_percent'] = ((analysis['filled_volume'] - analysis['original_volume']) / analysis['original_volume']) * 100
        else:
            analysis['original_volume'] = 0
            analysis['filled_volume'] = 0
            analysis['volume_increase_percent'] = 0
        
        print(f"\n=== 홀 채우기 효과 분석 ===")
        print(f"버텍스 증가: {analysis['vertex_increase_percent']:+.1f}% ({analysis['original_vertices']:,} → {analysis['filled_vertices']:,})")
        print(f"삼각형 증가: {analysis['triangle_increase_percent']:+.1f}% ({analysis['original_triangles']:,} → {analysis['filled_triangles']:,})")
        print(f"표면적 증가: {analysis['surface_area_increase_percent']:+.1f}%")
        if analysis['volume_increase_percent'] != 0:
            print(f"부피 증가: {analysis['volume_increase_percent']:+.1f}%")
        
        return analysis
        
    except Exception as e:
        print(f"비교 분석 중 오류: {e}")
        return {}


def fill_large_gaps_intelligently(mesh, gap_threshold=50, interpolation_method="cubic"):
    """
    큰 구멍들(옆구리, 팔 안쪽 등)을 지능적으로 채우는 특별화된 함수
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        gap_threshold (float): 큰 구멍으로 간주할 크기 임계값
        interpolation_method (str): 보간 방법 ("cubic", "rbf", "laplacian")
        
    Returns:
        o3d.geometry.TriangleMesh: 큰 구멍이 채워진 메시
    """
    print(f"\n=== 큰 구멍 지능적 채우기 (임계값: {gap_threshold}) ===")
    
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    
    try:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # NaN 값 체크 및 제거
        valid_mask = np.isfinite(vertices).all(axis=1)
        if not valid_mask.all():
            print(f"무효한 버텍스 {np.sum(~valid_mask)}개 제거")
            vertices = vertices[valid_mask]
        
        if len(vertices) == 0:
            return mesh
        
        # 1. 바운딩 박스 분석으로 인체 영역 식별
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        
        print(f"인체 바운딩 박스 - 높이: {extent[1]:.1f}, 너비: {extent[0]:.1f}, 깊이: {extent[2]:.1f}")
        
        # 2. 옆구리 영역 식별 (X축 양쪽 끝 영역)
        left_boundary = center[0] - extent[0] * 0.35   # 왼쪽 옆구리
        right_boundary = center[0] + extent[0] * 0.35  # 오른쪽 옆구리
        
        # 3. 팔 안쪽 영역 식별 (Y축 상반부, X축 중간 영역)
        torso_top = center[1] + extent[1] * 0.15      # 가슴-어깨 높이
        torso_bottom = center[1] - extent[1] * 0.15   # 허리 높이
        
        # 4. 구멍 밀도 분석을 위한 3D 그리드 생성
        grid_resolution = 20
        x_bins = np.linspace(bbox.min_bound[0], bbox.max_bound[0], grid_resolution)
        y_bins = np.linspace(bbox.min_bound[1], bbox.max_bound[1], grid_resolution)
        z_bins = np.linspace(bbox.min_bound[2], bbox.max_bound[2], grid_resolution)
        
        # 5. 각 그리드 셀의 버텍스 밀도 계산
        density_grid = np.zeros((grid_resolution-1, grid_resolution-1, grid_resolution-1))
        
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                for k in range(len(z_bins)-1):
                    # 현재 셀 내의 버텍스 개수 계산
                    in_cell = ((vertices[:, 0] >= x_bins[i]) & (vertices[:, 0] < x_bins[i+1]) &
                              (vertices[:, 1] >= y_bins[j]) & (vertices[:, 1] < y_bins[j+1]) &
                              (vertices[:, 2] >= z_bins[k]) & (vertices[:, 2] < z_bins[k+1]))
                    density_grid[i, j, k] = np.sum(in_cell)
        
        # 6. 낮은 밀도 영역을 구멍으로 식별
        mean_density = np.mean(density_grid[density_grid > 0])
        low_density_threshold = mean_density * 0.2  # 평균의 20% 이하를 구멍으로 간주
        
        print(f"평균 밀도: {mean_density:.1f}, 구멍 임계값: {low_density_threshold:.1f}")
        
        # 7. 구멍 영역에 보간 포인트 생성
        fill_points = []
        
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                for k in range(len(z_bins)-1):
                    if density_grid[i, j, k] < low_density_threshold:
                        # 셀 중심점 계산
                        cell_center = np.array([
                            (x_bins[i] + x_bins[i+1]) / 2,
                            (y_bins[j] + y_bins[j+1]) / 2,
                            (z_bins[k] + z_bins[k+1]) / 2
                        ])
                        
                        # 인체 형태에 맞는 영역인지 확인
                        x, y, z = cell_center
                        
                        # 옆구리 영역 또는 팔 안쪽 영역인지 확인
                        is_side_area = (x <= left_boundary or x >= right_boundary) and \
                                      (y >= torso_bottom and y <= torso_top)
                        
                        is_arm_inner = (x > left_boundary and x < right_boundary) and \
                                      (y >= torso_bottom and y <= torso_top) and \
                                      (z >= center[2] - extent[2] * 0.3)  # 앞쪽 영역
                        
                        if is_side_area or is_arm_inner:
                            # 주변 버텍스로부터 보간하여 적절한 위치 계산
                            nearby_vertices = vertices[
                                np.linalg.norm(vertices - cell_center, axis=1) < extent[0] * 0.15
                            ]
                            
                            if len(nearby_vertices) > 3:  # 충분한 참조점이 있는 경우
                                # 거리 기반 가중 평균으로 표면 위치 추정
                                distances = np.linalg.norm(nearby_vertices - cell_center, axis=1)
                                weights = 1 / (distances + 1e-6)  # 가까울수록 높은 가중치
                                weights /= np.sum(weights)
                                
                                interpolated_point = np.average(nearby_vertices, axis=0, weights=weights)
                                
                                # 인체 표면에 가까운 위치로 조정
                                direction_to_center = center - interpolated_point
                                direction_to_center /= (np.linalg.norm(direction_to_center) + 1e-6)
                                
                                # 표면으로부터 약간 안쪽으로 위치 조정
                                surface_point = interpolated_point + direction_to_center * (extent[0] * 0.05)
                                fill_points.append(surface_point)
        
        if len(fill_points) == 0:
            print("채울 구멍이 감지되지 않았습니다.")
            return mesh
        
        fill_points = np.array(fill_points)
        print(f"생성된 보간 포인트: {len(fill_points)}개")
        
        # 8. 기존 버텍스와 보간 포인트 결합
        enhanced_vertices = np.vstack([vertices, fill_points])
        
        # 9. 새로운 포인트 클라우드 생성
        enhanced_pcd = o3d.geometry.PointCloud()
        enhanced_pcd.points = o3d.utility.Vector3dVector(enhanced_vertices)
        
        # 10. 중복 포인트 제거
        enhanced_pcd = enhanced_pcd.remove_duplicated_points()
        
        # 11. 법선 벡터 계산 (더 세밀하게)
        enhanced_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=50)
        )
        enhanced_pcd.orient_normals_consistent_tangent_plane(k=20)
        
        # 12. 고해상도 포아송 재구성
        enhanced_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            enhanced_pcd, 
            depth=10,  # 높은 해상도
            width=0,
            scale=1.05,
            linear_fit=False
        )
        
        # 13. 밀도 기반 필터링 (더 관대하게)
        densities = np.asarray(densities)
        if len(densities) > 0:
            density_threshold = np.quantile(densities, 0.05)  # 하위 5%만 제거
            vertices_to_remove = densities < density_threshold
            enhanced_mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # 14. 최종 정리
        enhanced_mesh.remove_degenerate_triangles()
        enhanced_mesh.remove_duplicated_triangles()
        enhanced_mesh.remove_duplicated_vertices()
        enhanced_mesh.compute_vertex_normals()
        enhanced_mesh.compute_triangle_normals()
        
        # 15. 결과 출력
        final_vertices = len(enhanced_mesh.vertices)
        final_triangles = len(enhanced_mesh.triangles)
        vertex_increase = ((final_vertices - len(vertices)) / len(vertices)) * 100
        
        print(f"큰 구멍 채우기 완료: {final_vertices:,}개 버텍스 (+{vertex_increase:.1f}%)")
        
        return enhanced_mesh
        
    except Exception as e:
        print(f"큰 구멍 채우기 중 오류: {e}")
        return mesh


def bilateral_symmetry_completion(mesh, symmetry_plane="yz"):
    """
    양측 대칭성을 이용한 고급 완성 (옆구리, 팔 등 누락 부분 보완)
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 대상 메시
        symmetry_plane (str): 대칭면 ("yz", "xz", "xy")
        
    Returns:
        o3d.geometry.TriangleMesh: 대칭 완성된 메시
    """
    print(f"\n=== 양측 대칭성 기반 고급 완성 (평면: {symmetry_plane}) ===")
    
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    
    try:
        vertices = np.asarray(mesh.vertices)
        
        # NaN 값 체크
        valid_mask = np.isfinite(vertices).all(axis=1)
        if not valid_mask.all():
            vertices = vertices[valid_mask]
        
        if len(vertices) == 0:
            return mesh
        
        # 바운딩 박스와 중심점 계산
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        
        print(f"메시 중심: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        
        # 좌우 분할 (X축 기준)
        left_mask = vertices[:, 0] < center[0] - extent[0] * 0.05  # 약간의 여유
        right_mask = vertices[:, 0] > center[0] + extent[0] * 0.05
        center_mask = ~(left_mask | right_mask)  # 중앙 영역
        
        left_vertices = vertices[left_mask]
        right_vertices = vertices[right_mask]
        center_vertices = vertices[center_mask]
        
        print(f"좌측: {len(left_vertices)}개, 우측: {len(right_vertices)}개, 중앙: {len(center_vertices)}개")
        
        # 좌우 데이터 불균형 확인
        total_vertices = len(vertices)
        left_ratio = len(left_vertices) / total_vertices
        right_ratio = len(right_vertices) / total_vertices
        
        print(f"좌우 비율 - 좌측: {left_ratio:.1%}, 우측: {right_ratio:.1%}")
        
        # 심각한 불균형인 경우만 보완
        if abs(left_ratio - right_ratio) < 0.15:  # 15% 이하 차이
            print("좌우 균형이 양호하여 대칭 보완을 건너뜁니다.")
            return mesh
        
        # 데이터가 더 많은 쪽을 기준으로 부족한 쪽 보완
        if len(left_vertices) > len(right_vertices) * 1.3:  # 좌측이 30% 이상 많음
            print("좌측 데이터를 기반으로 우측 보완")
            source_vertices = left_vertices
            target_side = "right"
            flip_axis = 0  # X축 반전
            
        elif len(right_vertices) > len(left_vertices) * 1.3:  # 우측이 30% 이상 많음
            print("우측 데이터를 기반으로 좌측 보완")
            source_vertices = right_vertices
            target_side = "left"
            flip_axis = 0  # X축 반전
            
        else:
            print("좌우 차이가 보완 기준에 미달하여 건너뜁니다.")
            return mesh
        
        # 보완할 영역 식별 (옆구리, 팔 안쪽 등)
        # 상체 부분 (Y축 상위 50%)
        upper_body_mask = source_vertices[:, 1] > center[1] - extent[1] * 0.2
        upper_source = source_vertices[upper_body_mask]
        
        # 과도한 미러링 방지 - 최대 원본의 50%만 추가
        max_mirror_points = min(len(upper_source), total_vertices // 2)
        if len(upper_source) > max_mirror_points:
            # 옆구리/팔 영역 우선 선택
            distances_from_center = np.abs(upper_source[:, 0] - center[0])
            priority_indices = np.argsort(distances_from_center)[-max_mirror_points:]
            upper_source = upper_source[priority_indices]
        
        # 대칭 변환 적용
        mirrored_vertices = upper_source.copy()
        mirrored_vertices[:, flip_axis] = 2 * center[flip_axis] - mirrored_vertices[:, flip_axis]
        
        # 기존 영역과 겹치지 않는 포인트만 추가
        if target_side == "right":
            # 우측 영역에 추가
            valid_mirror_mask = mirrored_vertices[:, 0] > center[0]
        else:
            # 좌측 영역에 추가
            valid_mirror_mask = mirrored_vertices[:, 0] < center[0]
        
        valid_mirrored = mirrored_vertices[valid_mirror_mask]
        
        if len(valid_mirrored) == 0:
            print("유효한 미러링 포인트가 없습니다.")
            return mesh
        
        # 기존 버텍스와 결합
        all_vertices = np.vstack([vertices, valid_mirrored])
        
        print(f"미러링 포인트 {len(valid_mirrored)}개 추가 (총 {len(all_vertices)}개)")
        
        # 포인트 클라우드 생성
        enhanced_pcd = o3d.geometry.PointCloud()
        enhanced_pcd.points = o3d.utility.Vector3dVector(all_vertices)
        
        # 중복 제거 (더 엄격하게)
        enhanced_pcd = enhanced_pcd.remove_duplicated_points()
        
        # 통계적 이상치 제거 (너무 멀리 떨어진 포인트)
        enhanced_pcd, _ = enhanced_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 법선 벡터 계산
        enhanced_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=6, max_nn=30)
        )
        enhanced_pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # 포아송 재구성
        enhanced_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            enhanced_pcd, 
            depth=9,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # 밀도 필터링
        densities = np.asarray(densities)
        if len(densities) > 0:
            density_threshold = np.quantile(densities, 0.08)
            vertices_to_remove = densities < density_threshold
            enhanced_mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # 최종 정리
        enhanced_mesh.remove_degenerate_triangles()
        enhanced_mesh.remove_duplicated_triangles()
        enhanced_mesh.remove_duplicated_vertices()
        enhanced_mesh.compute_vertex_normals()
        enhanced_mesh.compute_triangle_normals()
        
        final_vertices = len(enhanced_mesh.vertices)
        vertex_increase = ((final_vertices - total_vertices) / total_vertices) * 100
        
        print(f"대칭 완성 결과: {final_vertices:,}개 버텍스 (+{vertex_increase:.1f}%)")
        
        return enhanced_mesh
        
    except Exception as e:
        print(f"대칭 완성 중 오류: {e}")
        return mesh