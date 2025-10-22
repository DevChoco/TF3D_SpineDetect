# 논문용 테스트 이미지 생성 스크립트

이 폴더는 논문에 사용할 비교 이미지를 생성하는 독립적인 스크립트들을 포함합니다.

## 📁 파일 구조

```
paper_test_images/
├── README.md                      # 이 파일
├── 1_mask_comparison.py           # 마스크 적용 전후 비교
├── 2_ransac_comparison.py         # RANSAC 성능 비교
├── 3_fpfh_comparison.py           # FPFH 특징 기반 정렬 비교
├── output_1_mask_comparison.png   # 생성된 이미지 (실행 후)
├── output_1_mask_process.png      # 생성된 이미지 (실행 후)
├── output_2_ransac_comparison.png # 생성된 이미지 (실행 후)
└── output_3_fpfh_comparison.png   # 생성된 이미지 (실행 후)
```

## 🎯 각 스크립트 설명

### 1. 마스크 적용 전후 비교 (`1_mask_comparison.py`)

**목적:** 마스크 처리가 포인트 클라우드 품질에 미치는 영향 시각화

**생성 이미지:**
- 좌측: 마스크 없이 생성한 포인트 클라우드 (노이즈 포함)
- 우측: 마스크 적용 후 포인트 클라우드 (깨끗한 결과)
- 추가: 마스크 생성 과정 (원본 깊이맵 → 이진 마스크)

**주요 메트릭:**
- 포인트 수 비교
- 노이즈 제거율 (%)
- 형태학적 연산 효과

**실행 방법:**
```bash
cd paper_test_images
python 1_mask_comparison.py
```

---

### 2. RANSAC 정렬 성능 비교 (`2_ransac_comparison.py`)

**목적:** RANSAC 기반 전역 정렬의 우수성 입증

**생성 이미지:**
- 좌측: 초기 상태 (정렬 전)
- 중앙: ICP만 사용 (로컬 최적화, 초기 위치에 민감)
- 우측: RANSAC + ICP (전역 최적화 후 정밀화)

**주요 메트릭:**
- 평균 정렬 오차 (mm)
- 표준편차
- Fitness 점수
- 성능 향상률 (%)

**실행 방법:**
```bash
cd paper_test_images
python 2_ransac_comparison.py
```

---

### 3. FPFH 특징 기반 정렬 비교 (`3_fpfh_comparison.py`)

**목적:** FPFH(Fast Point Feature Histogram) 특징의 효과 시연

**생성 이미지:**
- 상단: FPFH 특징 시각화 (33차원 히스토그램)
- 좌측 하단: 단순 ICP 정렬 (기하학적 특징만)
- 우측 하단: FPFH 기반 정렬 (로컬 표면 특징 활용)

**주요 메트릭:**
- 평균/중앙값 오차 (mm)
- 정밀도 (10mm 이내 포인트 비율)
- Fitness 점수
- 성능 개선 비율

**실행 방법:**
```bash
cd paper_test_images
python 3_fpfh_comparison.py
```

---

## 📊 논문 활용 가이드

### Figure 1: 전처리 단계의 중요성
- 사용 이미지: `output_1_mask_comparison.png`, `output_1_mask_process.png`
- 캡션 예시:
  ```
  Figure 1. Effect of mask preprocessing on point cloud quality.
  (a) Raw point cloud with noise and background artifacts.
  (b) Cleaned point cloud after morphological mask operations.
  The mask processing reduces noise by XX% while preserving body geometry.
  ```

### Figure 2: 전역 정렬의 필요성
- 사용 이미지: `output_2_ransac_comparison.png`
- 캡션 예시:
  ```
  Figure 2. Comparison of alignment methods.
  (a) Initial misaligned state.
  (b) ICP-only alignment (local optimization, susceptible to local minima).
  (c) RANSAC + ICP alignment (global initialization + refinement).
  RANSAC-based method achieves XX% better accuracy and is robust to initial pose.
  ```

### Figure 3: 특징 기반 정합의 우수성
- 사용 이미지: `output_3_fpfh_comparison.png`
- 캡션 예시:
  ```
  Figure 3. FPFH feature-based alignment performance.
  Top: FPFH feature visualization showing local surface geometry.
  Bottom: (a) Simple ICP using geometric correspondence only.
          (b) FPFH-based alignment leveraging 33D local descriptors.
  Feature-rich matching improves precision by XX% and fitness score by YY%.
  ```

---

## 🔧 커스터마이징

### 다른 데이터로 테스트하기

각 스크립트의 상단에서 경로를 수정:

```python
# 예: 2_ransac_comparison.py
front_path = r"your_path_to_front_depthmap.bmp"
right_path = r"your_path_to_right_depthmap.bmp"
```

### 시각화 파라미터 조정

```python
# 포인트 크기
opt.point_size = 3.0  # 더 크게: 5.0

# 카메라 각도
ctr.set_front([0.5, -0.3, -0.8])  # 원하는 각도로 변경

# 이미지 해상도
vis.create_window(visible=False, width=1200, height=800)  # 더 고해상도

# DPI 설정
plt.savefig(output_path, dpi=600)  # 논문용 고해상도: 600 DPI
```

---

## 📈 성능 메트릭 설명

### 1. 평균 오차 (Mean Error)
- 정의: 소스와 타겟 포인트 클라우드 간 평균 거리
- 단위: mm
- 낮을수록 좋음

### 2. 중앙값 오차 (Median Error)
- 정의: 오차 분포의 중앙값
- 이상치에 강건한 메트릭
- 단위: mm

### 3. Fitness Score
- 정의: 대응점 비율 (correspondence ratio)
- 범위: 0.0 ~ 1.0
- 높을수록 좋음

### 4. 정밀도 (Precision)
- 정의: 특정 임계값(예: 10mm) 이내 포인트 비율
- 단위: %
- 높을수록 좋음

---

## ⚠️ 주의사항

1. **메모리 사용:**
   - 큰 포인트 클라우드는 많은 메모리 필요
   - 시스템 메모리 16GB 이상 권장

2. **실행 시간:**
   - RANSAC은 반복 횟수가 많아 시간 소요
   - 각 스크립트 실행: 약 30-60초

3. **시각화 창:**
   - `visible=False` 설정으로 백그라운드 실행
   - 최종 결과만 `plt.show()`로 표시

4. **경로 설정:**
   - Windows 절대 경로 사용 (r"D:\path\to\file.bmp")
   - 상대 경로는 작동하지 않을 수 있음

---

## 📝 논문 작성 팁

### 방법론 섹션 (Methods)

```latex
\subsection{Preprocessing with Morphological Mask}
To improve point cloud quality, we apply a multi-stage mask generation process:
1) Binary thresholding ($0.2 < depth < 0.95$)
2) Morphological opening (remove salt noise)
3) Morphological closing (fill pepper noise)
4) Connected component analysis (extract main body)

As shown in Figure 1, this preprocessing reduces noise by XX\% while 
preserving anatomical structure.

\subsection{Global Alignment with RANSAC}
Unlike traditional ICP which is sensitive to initial pose, we employ 
RANSAC-based global alignment (Figure 2). This approach:
- Samples correspondence sets from FPFH features
- Estimates transformation via RANSAC (20,000 iterations)
- Refines with multi-scale ICP

Results demonstrate XX\% improvement in alignment accuracy compared to 
ICP-only methods.

\subsection{Feature-based Correspondence}
We utilize FPFH (Fast Point Feature Histogram) descriptors to establish 
robust correspondences between views (Figure 3). Each point is represented 
by a 33-dimensional histogram encoding local surface geometry, enabling 
accurate matching under partial overlap and viewpoint variation.
```

### 결과 섹션 (Results)

```latex
\subsection{Alignment Quality}
Table 1 summarizes alignment quality across different methods.
FPFH-based alignment achieves:
- Mean error: X.XX mm (YY\% improvement)
- Precision (10mm): ZZ.Z\% (WW\% improvement)
- Fitness score: 0.XXXX (highest among all methods)
```

---

## 🔗 의존성

모든 스크립트는 다음 라이브러리를 사용:

```python
- numpy
- open3d
- matplotlib
- opencv-python (cv2)
- PIL (Pillow)
```

설치:
```bash
pip install numpy open3d matplotlib opencv-python Pillow
```

---

## 📧 문의

이미지 생성 중 문제가 발생하면:
1. 깊이맵 경로가 올바른지 확인
2. Python 환경에 모든 라이브러리 설치 확인
3. 메모리 부족 시 포인트 클라우드 다운샘플링 고려

---

**생성 날짜:** 2025-10-22  
**버전:** 1.0  
**라이선스:** MIT
