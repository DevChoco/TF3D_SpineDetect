# 모듈화된 3D 자세 분석 시스템

이 프로젝트는 4개의 깊이맵 이미지(정면, 좌측, 우측, 후면)를 사용하여 3D 인체 모델을 생성하고 자세를 분석하는 시스템입니다. FPFH(Fast Point Feature Histograms) 기반 포인트 클라우드 정렬과 AI 기반 스켈레톤 파싱을 사용합니다.

## 주요 기능

### 1. FPFH 기반 포인트 클라우드 정렬
- Fast Point Feature Histograms를 사용한 특징 추출
- RANSAC 기반 전역 초기 정합
- 다중 스케일 ICP 정밀 정렬
- 제한된 회전 모드 및 번역 전용 모드 지원
- 비강직 정합(CPD) 옵션 지원

### 2. 스켈레톤 파싱 및 자세 분석
- MediaPipe를 활용한 AI 기반 랜드마크 검출
- 의학적 기준에 따른 정확한 척추 구조 생성
- 경추, 흉추, 요추 각도 분석
- 어깨 수평도 및 골반 기울기 측정
- 전체 척추 정렬도 평가

### 3. 3D 메시 생성
- Poisson 표면 재구성
- Ball Pivoting Algorithm (대안)
- 메시 최적화 및 후처리
- 다양한 형식 지원 (OBJ, PLY, STL)

## 모듈 구조

```
modular_3d_pose/
├── modules/
│   ├── __init__.py
│   ├── pointcloud_generator.py    # 포인트 클라우드 생성 및 전처리
│   ├── fpfh_alignment.py         # FPFH 기반 정렬
│   ├── skeleton_parser.py        # 스켈레톤 파싱 및 자세 분석
│   └── mesh_generator.py         # 메시 생성 및 저장
├── output/
│   ├── 3d_models/               # 생성된 메시 파일
│   └── debug/                   # 디버그 이미지
├── main.py                      # 메인 실행 파일
├── requirements.txt             # 필요 라이브러리
└── README.md                    # 프로젝트 설명서
```

## 설치 및 실행

### 1. 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. 입력 이미지 준비
4개의 깊이맵 이미지를 준비합니다:
- 정면 (front)
- 좌측 (left)  
- 우측 (right)
- 후면 (back)

### 3. 실행
```bash
python main.py
```

## 사용법

### 기본 사용법
`main.py`의 `views` 딕셔너리에서 입력 이미지 경로를 설정하고 실행합니다.

```python
views = {
    "front": "path/to/front_depth.bmp",
    "right": "path/to/right_depth.bmp", 
    "left": "path/to/left_depth.bmp",
    "back": "path/to/back_depth.bmp"
}
```

### 모듈별 사용법

#### 1. 포인트 클라우드 생성
```python
from modules.pointcloud_generator import load_depth_map, create_point_cloud_from_depth

depth_map = load_depth_map("path/to/depth_image.bmp")
pcd = create_point_cloud_from_depth(depth_map, "front")
```

#### 2. FPFH 정렬
```python
from modules.fpfh_alignment import align_point_clouds_fpfh

params = {
    'voxel_coarse': 5.0,
    'voxel_list': [20.0, 10.0, 5.0],
    'ransac_iter': 20000,
    'allow_rotation': False,
    'allow_small_rotation': True
}
aligned_pcd = align_point_clouds_fpfh(source_pcd, target_pcd, params=params)
```

#### 3. 스켈레톤 분석
```python
from modules.skeleton_parser import (
    detect_landmarks_with_ai,
    create_skeleton_from_pointcloud, 
    calculate_spine_angles,
    print_angles
)

# AI 랜드마크 검출
landmarks = detect_landmarks_with_ai("path/to/front_image.bmp")

# 스켈레톤 생성
skeleton_points = create_skeleton_from_pointcloud(pcd, landmarks)

# 각도 분석
angles = calculate_spine_angles(skeleton_points)
print_angles(angles)
```

#### 4. 메시 생성
```python
from modules.mesh_generator import create_and_save_mesh

mesh, saved_files = create_and_save_mesh(pcd, "output/3d_models", "body_mesh")
```

## 출력 결과

### 1. 3D 모델 파일
- `body_mesh.obj`: 일반적인 3D 모델 형식
- `body_mesh.ply`: 포인트 클라우드 연구용
- `body_mesh.stl`: 3D 프린팅용

### 2. 자세 분석 결과
```
           인체 자세 분석 결과
==================================================

척추 각도 분석:
   • 경추 전만각 (Cervical Lordosis): 42.3°
     - 정상 범위: 35-45°

   • 흉추 후만각 (Thoracic Kyphosis): 28.7°
     - 정상 범위: 20-40°

   • 요추 전만각 (Lumbar Lordosis): 45.1°
     - 정상 범위: 40-60°

어깨 및 골반 분석:
   • 어깨 수평도 (Shoulder Level): 2.1°
     - 정상: 0° (완전 수평)

   • 골반 기울기 (Pelvis Tilt): 1.3°
     - 정상: 0° (완전 수평)

전체 척추 정렬:
   • 척추 정렬도 (Spine Alignment): 3.4°
     - 정상: 0° (완전 수직)

자세 평가:
   ✅  전반적으로 양호한 자세입니다!
==================================================
```

### 3. 디버그 이미지
- `front_mask.png`: 정면 뷰 마스크
- `right_mask.png`: 우측 뷰 마스크
- `left_mask.png`: 좌측 뷰 마스크
- `back_mask.png`: 후면 뷰 마스크

## 정렬 매개변수

### FPFH 정렬 매개변수
- `voxel_coarse`: 전역 RANSAC용 복셀 크기 (기본값: 5.0)
- `voxel_list`: 다중 스케일 ICP용 복셀 크기 리스트 (기본값: [20.0, 10.0, 5.0])
- `ransac_iter`: RANSAC 반복 횟수 (기본값: 20000)
- `allow_rotation`: 회전 허용 여부 (기본값: True)
- `allow_small_rotation`: 미세 회전 허용 여부 (기본값: False)
- `use_cpd`: 비강직 정합 사용 여부 (기본값: True)

## 기술적 특징

### 1. 모듈화 설계
- 각 기능별로 독립적인 모듈 구성
- 쉬운 유지보수 및 확장성
- 재사용 가능한 컴포넌트

### 2. 강건한 정렬 알고리즘
- FPFH 특징 기반 전역 정합
- 다중 스케일 ICP로 정밀 정렬
- 회전 제한 모드로 안정성 향상

### 3. AI 기반 랜드마크 검출
- MediaPipe 활용으로 정확한 해부학적 위치 검출
- 개인별 체형 특성 반영
- 검출 실패시 기본값 자동 적용

### 4. 의학적 정확성
- 해부학적 기준에 따른 척추 구조
- 정확한 각도 측정 및 평가
- 임상적으로 유의미한 분석 결과

## 문제 해결

### 1. 정렬 실패시
- 입력 이미지 품질 확인
- 조명 조건 개선
- 마스크 임계값 조정

### 2. 메시 생성 실패시
- 포인트 클라우드 밀도 확인
- Poisson 매개변수 조정
- Ball Pivoting 알고리즘 사용

### 3. 랜드마크 검출 실패시
- 이미지 해상도 확인
- 촬영 각도 조정
- 대비 및 밝기 개선

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 사용할 수 있습니다.

## 연락처

문의사항이 있으시면 GitHub Issues를 통해 연락해 주세요.