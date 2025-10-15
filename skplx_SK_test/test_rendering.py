"""
간단한 렌더링 테스트 및 수정
"""
import trimesh
import numpy as np

# 메쉬 로드
mesh = trimesh.load('3d_file/body_mesh_fpfh.obj', force='mesh')
print(f"메쉬 로드: {len(mesh.vertices)} vertices")

# 메쉬 정규화
mesh_centered = mesh.copy()
mesh_centered.vertices -= mesh_centered.centroid
scale = np.max(np.abs(mesh_centered.vertices))
mesh_centered.vertices /= scale

print(f"정규화 스케일: {scale}")
print(f"정규화 범위: {mesh_centered.bounds}")

# PyRender 테스트
try:
    import pyrender
    print("\nPyRender 설치됨 - 렌더링 테스트...")
    
    # PyRender 메쉬 생성
    mesh_pr = pyrender.Mesh.from_trimesh(mesh_centered)
    
    # 씬 생성
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
    scene.add(mesh_pr)
    
    # 조명
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=np.eye(4))
    
    # 카메라
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ])
    scene.add(camera, pose=camera_pose)
    
    # 렌더링
    renderer = pyrender.OffscreenRenderer(512, 512)
    color, depth = renderer.render(scene)
    renderer.delete()
    
    print(f"✓ PyRender 성공! 이미지 형상: {color.shape}")
    print(f"  픽셀 범위: {color.min()} ~ {color.max()}")
    
    # 저장
    import cv2
    cv2.imwrite('render_test.png', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    print("  저장: render_test.png")
    
except ImportError as e:
    print(f"\n✗ PyRender 없음: {e}")
    print("  설치: pip install pyrender pyopengl")
except Exception as e:
    print(f"\n✗ PyRender 오류: {e}")
