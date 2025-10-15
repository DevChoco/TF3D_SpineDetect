# MediaPipe 기반 3D 메쉬 척추 관절 검출 시스템

## 개요

이 시스템은 **일반 3D 메쉬**에서 **척추 관절 라인을 정확하게 예측**하는 고급 파이프라인입니다.

MediaPipe BlazePose의 3D 키포인트 추정 기능과 메쉬 표면 정보를 결합하여, 다양한 각도에서도 안정적으로 척추를 검출합니다.

## 주요 특징

### 1. MediaPipe BlazePose 3D 활용
- **BlazePose GHUM (Google Human Model)** 기반
- **3D heatmap regression**으로 깊이(Z) 방향 추정
- **33개 키포인트** 검출 (Full 모델)
- X, Y, Z 좌표 모두 제공

### 2. 다중 뷰 융합 (Multi-View Fusion)
- 정면, 측면, 후면 등 여러 각도에서 렌더링
- 각 뷰별 독립 검출 후 융합
- **Visibility 가중 평균**으로 정확도 향상
- 가려진 관절 처리 가능

### 3. Visibility 기반 필터링
- 각 랜드마크의 **visibility 값** (0.0~1.0) 활용
- 가려진 관절 자동 제외
- 신뢰도 높은 랜드마크만 사용

### 4. Temporal Smoothing (시간적 보정)
- 연속 프레임의 랜드마크 이력 활용
- 가우시안 평균으로 노이즈 제거
- 움직임 안정화

### 5. 메쉬 표면 정제
- KD-Tree로 메쉬 표면의 가장 가까운 정점 탐색
- 초기 예측을 메쉬 형상에 맞게 조정
- 해부학적 정확도 향상

### 6. 척추 세그먼트 추정
자동으로 다음 척추 부위를 예측합니다:
- **C7** (경추 7번) - 목 아래
- **T1-T12** (흉추 1~12번) - 등 상부/중부
- **L1-L5** (요추 1~5번) - 허리
- **Sacrum** (천골) - 골반

## 설치

### 1. 의존성 설치

```powershell
pip install -r requirements_mediapipe.txt
```

### 2. 추가 설정 (선택)

PyRender를 사용하는 경우 추가 라이브러리가 필요할 수 있습니다:

```powershell
pip install pyopengl pyglet
```

## 사용법

### 기본 사용

```powershell
python mediapipe_spine_detector.py --mesh 3d_file/body_mesh_fpfh.obj --visualize
```

### 고급 옵션

```powershell
# 여러 뷰 사용 + 시각화
python mediapipe_spine_detector.py --mesh 3d_file/body_mesh_fpfh.obj --views front side back top --visualize

# 최소 가시성 임계값 조정 (기본: 0.5)
python mediapipe_spine_detector.py --mesh 3d_file/body_mesh_fpfh.obj --min-visibility 0.6

# 시간적 스무딩 활성화 (연속 이미지용)
python mediapipe_spine_detector.py --mesh 3d_file/body_mesh_fpfh.obj --smooth

# 메쉬 정제 비활성화 (빠른 처리)
python mediapipe_spine_detector.py --mesh 3d_file/body_mesh_fpfh.obj --no-refine

# 출력 디렉토리 지정
python mediapipe_spine_detector.py --mesh 3d_file/body_mesh_fpfh.obj --output-dir custom_output
```

### Python 스크립트에서 사용

```python
from mediapipe_spine_detector import MediaPipeSpineDetector

# 검출기 생성
detector = MediaPipeSpineDetector(
    model_complexity=2,      # 0(Lite), 1(Full), 2(Heavy)
    smooth_landmarks=True    # 랜드마크 스무딩
)

# 척추 검출
result = detector.detect_spine_from_mesh(
    mesh_path='3d_file/body_mesh_fpfh.obj',
    views=['front', 'side', 'back'],
    min_visibility=0.5,
    apply_smoothing=False,
    refine_with_mesh=True
)

# 결과 확인
if result['success']:
    print(f"검출된 척추 키포인트: {len(result['spine_keypoints'])}개")
    print(f"평균 신뢰도: {result['statistics']['avg_confidence']:.3f}")
    
    # 척추 키포인트 접근
    for kp in result['spine_keypoints']:
        print(f"{kp['name']}: ({kp['x']:.3f}, {kp['y']:.3f}, {kp['z']:.3f}), "
              f"confidence={kp['confidence']:.3f}")
    
    # 시각화
    detector.visualize_results(result, 'output_spine.png')
    
    # JSON 저장
    detector.save_results(result, 'output_spine.json')
```

## 출력 형식

### JSON 결과 구조

```json
{
  "success": true,
  "mesh_path": "3d_file/body_mesh_fpfh.obj",
  "views_used": ["front", "side", "back"],
  "spine_keypoints": [
    {
      "name": "C7",
      "level": "cervical",
      "x": 0.012,
      "y": 0.456,
      "z": -0.023,
      "confidence": 0.89,
      "refined": true,
      "mesh_distance": 0.008
    },
    {
      "name": "T1",
      "level": "thoracic_upper",
      ...
    }
  ],
  "statistics": {
    "total_keypoints": 25,
    "refined_keypoints": 23,
    "avg_confidence": 0.82,
    "views_detected": 3
  }
}
```

### 시각화 출력

3개의 플롯이 생성됩니다:
1. **3D 척추 라인** - 3차원 공간에서의 척추 형태
2. **측면 투영 (Z-Y)** - 척추 곡률 확인
3. **정면 투영 (X-Y)** - 척추 정렬 확인

색상은 신뢰도를 나타냅니다 (녹색=높음, 빨간색=낮음).

## MediaPipe의 장단점

### ✅ 장점

1. **실시간 처리 가능** - 빠른 추론 속도
2. **3D 좌표 제공** - X, Y, Z 모두 출력
3. **다양한 포즈 지원** - 정면, 측면, 후면 모두 가능
4. **경량 모델** - GPU 불필요
5. **높은 범용성** - 다양한 체형, 자세에 적용 가능

### ⚠️ 한계 및 해결책

| 한계 | 설명 | 해결책 |
|------|------|--------|
| **상대적 깊이** | Z축이 실제 깊이가 아닌 상대값 | 다중 뷰 융합 + 메쉬 정제 |
| **카메라 의존성** | 각도/조명에 따라 불안정 | 여러 뷰에서 렌더링 |
| **관절 겹침** | 측면에서 팔/다리 겹치면 왜곡 | Visibility 필터링 |
| **척추 세부 부족** | 33개 키포인트에 척추 상세 없음 | 보간 + 해부학적 모델링 |

### 🔧 본 시스템의 보완책

1. **다중 뷰 렌더링** → 각도 의존성 완화
2. **Visibility 가중 융합** → 가려진 관절 처리
3. **메쉬 표면 정제** → 실제 3D 형상에 맞춤
4. **척추 세그먼트 모델링** → C7~Sacrum 추정
5. **Temporal Smoothing** → 노이즈 제거

## 정확도 향상 팁

### 1. 뷰 선택
- **정면 + 측면**: 가장 기본적 조합
- **정면 + 좌측면 + 우측면**: 최적 균형
- **360도 다중 뷰**: 최고 정확도 (느림)

### 2. 모델 복잡도
```python
# 빠른 처리 (부정확)
detector = MediaPipeSpineDetector(model_complexity=0)

# 균형 (권장)
detector = MediaPipeSpineDetector(model_complexity=1)

# 최고 정확도 (느림)
detector = MediaPipeSpineDetector(model_complexity=2)
```

### 3. Visibility 임계값
- **0.3**: 더 많은 랜드마크 포함 (노이즈 증가)
- **0.5**: 권장값 (균형)
- **0.7**: 고품질만 (일부 누락 가능)

### 4. 메쉬 전처리
- 메쉬를 정규화하여 전달
- 표면 노이즈 제거 (Laplacian smoothing)
- 구멍 메우기 (hole filling)

## 응용 사례

1. **척추 측만증 진단** - 척추 정렬 분석
2. **자세 교정** - 실시간 척추 곡률 모니터링
3. **3D 의료 영상** - CT/MRI 보완 데이터
4. **재활 훈련** - 척추 움직임 추적
5. **인체공학 평가** - 작업 자세 분석

## 문제 해결

### 문제: 포즈 검출 실패
**원인**: 메쉬가 사람 형태가 아니거나 렌더링 품질 낮음  
**해결**: 메쉬 품질 확인, 이미지 해상도 증가

### 문제: 척추 키포인트 부정확
**원인**: 단일 뷰 사용, 낮은 visibility  
**해결**: 다중 뷰 활성화, min-visibility 낮춤

### 문제: 느린 처리 속도
**원인**: 높은 model_complexity, 많은 뷰  
**해결**: model_complexity=1로 낮춤, 뷰 개수 감소

### 문제: 메쉬 정제 오류
**원인**: 메쉬와 예측 위치가 너무 떨어짐  
**해결**: `--no-refine` 옵션 사용

## 라이선스

이 코드는 연구 및 교육 목적으로 사용 가능합니다.

## 참고 문헌

- MediaPipe BlazePose: [https://google.github.io/mediapipe/solutions/pose](https://google.github.io/mediapipe/solutions/pose)
- BlazePose Paper: "BlazePose: On-device Real-time Body Pose tracking" (CVPR 2020)
- GHUM Model: "GHUM & GHUML: Generative 3D Human Shape and Articulated Pose Models" (CVPR 2020)

## 연락처

문제나 제안사항이 있으면 이슈를 등록해주세요.
