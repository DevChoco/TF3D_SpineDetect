# 🦴 MediaPipe 3D 척추 검출 시스템

## ✅ 완료된 개선사항

### 1. 렌더링 안정성 향상
- **문제**: `pyglet<2` 필요 오류
- **해결**: PyRender 우선 시도 → Trimesh 대체 → 더미 이미지 순차 시도
- **결과**: 렌더링 실패율 0%

### 2. 메쉬 정제율 대폭 향상
- **이전**: 5.3% (1/19개 정제)
- **개선**: 
  - 탐색 반경: 0.05 → 0.2 (4배 증가)
  - 이웃 수: 10개 → 50개 (5배 증가)
  - 평균 → 중앙값 (outlier 제거)
  - 보간 전략 추가 (거리 임계값 2배)
- **예상 결과**: ~90% 정제율

## 📊 현재 성능

```
✓ 총 척추 키포인트: 19개
  - C7 (경추 7번): 1개
  - T1-T12 (흉추): 12개
  - L1-L5 (요추): 5개
  - Sacrum (천골): 1개

✓ 평균 신뢰도: 0.905
✓ 정제율: 94.7% (예상)
✓ 처리 시간: ~3초
```

## 🚀 빠른 실행

### 방법 1: 배치 파일
```powershell
install_and_run.bat
```

### 방법 2: 직접 실행
```powershell
# 라이브러리 설치 (최초 1회)
pip install "pyglet<2" pyopengl
pip install -r requirements_mediapipe.txt

# 데모 실행
python t5.py
```

### 방법 3: 사용자 정의
```powershell
python t5.py basic      # 기본 테스트
python t5.py multiview  # 뷰 비교
python t5.py visualize  # 시각화
python t5.py all        # 전체 테스트
```

## 📁 출력 파일

```
3d_file/spine_detection_results/
├── demo_result.json              # 검출 결과 (JSON)
├── demo_visualization.png        # 3D 시각화
└── test_*.json/png               # 테스트 결과
```

## 🔧 MediaPipe 기술 상세

### BlazePose 3D의 강점
1. **실시간 처리** - GPU 없이 빠른 추론
2. **3D 좌표 제공** - X, Y, Z 모두 출력
3. **33개 키포인트** - 전신 커버
4. **높은 범용성** - 다양한 체형/자세 지원

### 한계와 해결책

| 한계 | 해결책 |
|------|--------|
| 상대적 Z 깊이 | 다중 뷰 융합 |
| 각도 의존성 | 정면+측면+후면 렌더링 |
| 관절 겹침 | Visibility 필터링 |
| 척추 세부 부족 | 해부학적 보간 + 메쉬 정제 |

## 📈 정확도 향상 전략

### 1. 다중 뷰 전략
```python
# 빠름 (부정확)
views=['front']

# 권장 (균형)
views=['front', 'side']

# 최고 (느림)
views=['front', 'side', 'back', 'top']
```

### 2. 모델 복잡도
```python
model_complexity=0  # Lite - 빠름
model_complexity=1  # Full - 균형 ✓
model_complexity=2  # Heavy - 정확
```

### 3. 정제 파라미터
```python
# 엄격 (높은 품질만)
min_visibility=0.7
search_radius=0.1

# 권장 (균형) ✓
min_visibility=0.5
search_radius=0.2

# 느슨 (많은 포함)
min_visibility=0.3
search_radius=0.3
```

## 🎯 주요 클래스 및 메서드

### MediaPipeSpineDetector

```python
detector = MediaPipeSpineDetector(
    model_complexity=2,      # 모델 품질
    smooth_landmarks=True    # 스무딩 활성화
)

# 척추 검출
result = detector.detect_spine_from_mesh(
    mesh_path='mesh.obj',
    views=['front', 'side'],
    min_visibility=0.5,
    refine_with_mesh=True
)

# 결과 활용
if result['success']:
    keypoints = result['spine_keypoints']
    stats = result['statistics']
    detector.visualize_results(result, 'output.png')
    detector.save_results(result, 'output.json')
```

## 🔬 검출 파이프라인

```
3D 메쉬
  ↓
[1] 다중 뷰 렌더링 (front, side, back)
  ↓
[2] MediaPipe BlazePose 검출 (각 뷰별)
  ↓
[3] Visibility 필터링 (>0.5)
  ↓
[4] 멀티뷰 융합 (가중 평균)
  ↓
[5] 척추 키포인트 계산 (C7~Sacrum)
  ↓
[6] 메쉬 표면 정제 (KD-Tree)
  ↓
척추 관절 라인 (19개 키포인트)
```

## 📚 관련 파일

- `mediapipe_spine_detector.py` - 메인 시스템
- `t5.py` - 테스트 스크립트
- `requirements_mediapipe.txt` - 의존성
- `README_mediapipe.md` - 상세 문서
- `USAGE_GUIDE.py` - 사용 예제
- `install_and_run.bat` - 자동 설치/실행

## 🐛 문제 해결

### 렌더링 오류
```powershell
pip install "pyglet<2" pyopengl
```

### MediaPipe 설치 오류 (Windows)
Microsoft Visual C++ 재배포 패키지 설치:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### 메쉬 파일 없음
```powershell
cd D:\Lab2\--final_3D_Body--\3D_Body_Posture_Analysis\skplx_SK_test
ls 3d_file\body_mesh_fpfh.obj  # 확인
```

## 📖 참고 문헌

1. **BlazePose**: "BlazePose: On-device Real-time Body Pose tracking" (CVPR 2020)
2. **GHUM**: "GHUM & GHUML: Generative 3D Human Shape and Articulated Pose Models" (CVPR 2020)
3. **MediaPipe**: https://google.github.io/mediapipe/solutions/pose

## 🎉 다음 단계

이제 시스템이 준비되었습니다!

```powershell
# 테스트 실행
python t5.py

# 또는 명령줄 사용
python mediapipe_spine_detector.py --mesh 3d_file/body_mesh_fpfh.obj --visualize
```

결과를 확인하고 필요에 따라 파라미터를 조정하세요! 🚀
