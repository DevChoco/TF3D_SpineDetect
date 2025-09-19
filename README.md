# 3D_Body_Posture_Analysis

- `3d_pose1_main.py` : ICP 정렬 테스트 1 (실패)
- `3d_pose2_main.py` : ICP 정렬 테스트 2 (가능)
- `3d_pose2_main_addmask.py` : `3d_pose2_main.py`에 마스킹 전처리 과정추가 (가능)
- `3d_pose3_main_FPFH.py` : FPFH모델을 이용한 테스트 (가능)

- `3d_pose2-5_main.py` : 스켈레톤 파싱 테스트 (실패)
- `3d_pose2-6_main.py` : 스켈레톤 파싱 테스트 (실패)
- `3d_pose3_main_pose.py` : 스켈레톤 파싱 테스트 (가능)
- `3d_pose4_main_pose.py` : 스켈레톤 파싱 테스트 (실패)

----------
----------

# 원조 논문 정보

#### 1. **Besl & McKay (1992)**

* **제목**: *"A Method for Registration of 3‑D Shapes"*
* **저자**: Paul J. Besl 및 Neil D. McKay
* **저널**: *IEEE Transactions on Pattern Analysis and Machine Intelligence*
* **권·호**: Vol. 14, No. 2, pp. 239–256 (1992년 2월)
* **내용**: ICP 알고리즘을 처음 체계적으로 제안한 논문으로, 두 3D 포인트 셋을 정합(registration)하는 대표적인 방법으로 확립되었습니다. ICP 알고리즘의 수렴 특성, 반복 구조, 오차 최소화 방식 등이 논의되어 있습니다. ([Colab][1], [OUCI][2], [ACM 디지털 라이브러리][3])

#### 2. **Arun et al. (1987)**

* **제목**: *"Least‑Squares Fitting of Two 3‑D Point Sets"*
* **저자**: K. S. Arun, T. S. Huang, S. D. Blostein
* **저널**: *IEEE Transactions on Pattern Analysis and Machine Intelligence*, Vol. 9, No. 5, pp. 698–700 (1987년 5월)
* **내용**: 두 포인트 셋 간의 rigid 변환(회전 및 병진)을 찾기 위한 최소자승 기반 닫힌 형식 해(Closed‑form solution)를 제안했으며, ICP 알고리즘의 “변환 최적화” 단계에서 핵심적으로 활용되는 방식입니다. ([Space Frontiers][4], [WBLDB][5], [Illinois Experts][6])

---

### ICP 알고리즘의 맥락

ICP는 크게 두 단계의 반복 구조로 이루어집니다:

1. **Correspondence (대응점 찾기)**
   현재 추정된 변환을 기준으로, 모델 포인트 셋의 각 점에 대해 타겟 셋에서 가장 가까운 점을 찾아 대응을 구성합니다.

2. **Transformation Estimation (변환 추정)**
   대응된 점 쌍들에 대해, 회전과 병진을 추정해 오차(보통 평균 제곱 오차)를 최소화합니다. 이 단계에서 Arun et al. (1987)의 방법이 널리 활용됩니다.

이 과정을 반복(iteration)하며 수렴할 때까지 정합 정밀도를 높입니다. 이러한 구조는 Besl & McKay (1992)에서 공식화되었습니다. ([위키백과][7])

---

### 요약 테이블

| 역할                 | 논문 및 연도             | 주요 기여                            |
| ------------------ | ------------------- | -------------------------------- |
| **최초 ICP 알고리즘 제안** | Besl & McKay (1992) | ICP 반복 구조 및 수렴 특성 제시             |
| **변환 계산 방식 제안**    | Arun et al. (1987)  | 변환(R, T)을 닫힌 형식으로 계산하는 SVD 방법 제시 |

---

### 결론

ICP 정렬 알고리즘의 원조로는 **Besl & McKay (1992)** 논문이 대표적으로 인정됩니다. 이 알고리즘의 핵심 변환 해법(least-squares rigid alignment)은 **Arun et al. (1987)** 논문에서 닫힌 형식으로 해법을 제공한 것이 ICP 발전에 큰 영향을 미쳤습니다.

더 깊이 있는 설명이나 구현 참고가 필요하시면 언제든지 말씀 주세요!

[1]: https://colab.ws/articles/10.1109%2F34.121791.?utm_source=chatgpt.com "A method for registration of 3-D shapes | CoLab"
[2]: https://ouci.dntb.gov.ua/en/works/4Mox82v9/?utm_source=chatgpt.com "A method for registration of 3-D shapes"
[3]: https://dl.acm.org/doi/abs/10.1109/34.121791?utm_source=chatgpt.com "A Method for Registration of 3-D Shapes | IEEE Transactions on Pattern Analysis and Machine Intelligence"
[4]: https://spacefrontiers.org/r/10.1109/tpami.1987.4767965?utm_source=chatgpt.com "Least-Squares Fitting of Two 3-D Point Sets | Space Frontiers"
[5]: https://wbldb.lievers.net/10349520.html?utm_source=chatgpt.com "Least-squares fitting of two 3-D point sets"
[6]: https://experts.illinois.edu/en/publications/least-squares-fitting-of-two-3-d-point-sets?utm_source=chatgpt.com "Least-Squares Fitting of Two 3-D Point Sets - Illinois Experts"
[7]: https://en.wikipedia.org/wiki/Point-set_registration?utm_source=chatgpt.com "Point-set registration"

----------
----------

# 대표적인 DCP기반 최신 학술 논문 소개

### 1. **Deep Closest Point (DCP)** – Wang & Solomon, 2019

* **논문 제목**: *Deep Closest Point: Learning Representations for Point Cloud Registration*
* **주요 내용**: ICP의 한계를 극복하기 위해 딥러닝 기반으로 포인트셋 정합을 수행. 포인트 클라우드 임베딩, attention 기반 soft matching, differentiable SVD를 통합한 end-to-end 구조로, 여러 실험에서 ICP 및 Go‑ICP, FGR, PointNetLK보다 우수한 성능을 보여줌 ([arXiv][1]).

---

### 2. **DeepMatch: Toward Lightweight in Point Cloud Registration** – Qi 등, 2022

* **핵심 포인트**: DCP의 모델 복잡성과 높은 연산 비용을 줄이기 위해 설계된 경량화 알고리즘. 구조는 간결하며, DCP를 능가하는 성능을 상대적으로 적은 GPU 메모리와 연산으로 실현 ([Frontiers][2], [PMC][3]).

---

### 3. **MEDPNet: Multiscale Efficient Deep Closest Point** – Du 등, 2024

* **핵심 내용**: DCP 구조를 개선한 적용 사례 중 하나. Transformer attention 대신 Efficient Attention을 도입해 메모리 및 연산 효율을 높였으며, 이후 multiscale feature fusion과 ICP, NDT를 조합해 die-casting 분야에서 고정밀 정합을 달성 ([arXiv][4]).

---

### 4. **Deep Weighted Consensus**, 2021

* **핵심 내용**: DCP와 다른 학습 기반 방법보다도 더욱 robust한 정합. dense한 correspondence confidence map을 학습하여, 큰 회전과 높은 잡음 환경에서도 안정적인 정합을 수행 ([arXiv][5]).

---

### 5. **Mahalanobis DCP (MDCP)** – 2024

* **핵심 내용**: DCP에 Mahalanobis 기반 similarity 측정 방식을 적용한 개선 버전. transformer 포함/미포함 두 가지 variant로 구성되고, 다양한 데이터셋(ModelNet40, FAUST, Stanford3D)에서 정합 정밀도 향상을 보임 ([arXiv][6]).

---

### 6. **기타 주목할 학습 기반 접근들**

* **DOPNet**: Multi-level feature 기반 딥러닝 정합 구조 ([MDPI][7]).
* **PointCNT**: deep learning 기반 end-to-end, global feature 활용 정합 방식 ([MDPI][8]).

---

## 정리 테이블

| 알고리즘                        | 연도   | 주요 특징                                                       |
| --------------------------- | ---- | ----------------------------------------------------------- |
| **DCP**                     | 2019 | 딥러닝 기반 soft matching + SVD, ICP 대비 성능 우수                    |
| **DeepMatch**               | 2022 | 경량 구조, DCP보다 빠르고 적은 리소스로 정확도 향상                             |
| **MEDPNet**                 | 2024 | Efficient Attention 도입, multiscale fusion + ICP/NDT, 정밀도 향상 |
| **Deep Weighted Consensus** | 2021 | dense confidence map, 잡음/회전에 강함                             |
| **MDCP**                    | 2024 | Mahalanobis similarity 기반, transformer 유무 variant, 정밀도 향상   |
| **DOPNet**, **PointCNT**    | 최근   | 다양한 네트워크 구성 기반 정합 구조 소개                                     |

---

### 추천 순서

1. **DCP** – 기본 구조 이해용으로 가장 먼저 읽어볼 만합니다.
2. **DeepMatch** – DCP 구조를 간소화하고 빠른 처리가 필요한 경우 유용합니다.
3. **MEDPNet** – 실제 산업(주조) 환경에서 정밀도와 효율 모두 중요한 경우 강력 추천.
4. **Deep Weighted Consensus** – 잡음 많고 큰 회전이 포함된 환경에서 탁월합니다.
5. **MDCP** – 정밀한 정합이 특히 필요한 경우 Mahalanobis 기법이 유리.
6. 관심이 있다면 **DOPNet**이나 **PointCNT**도 참고하세요.

---

추가로 각 논문의 구현 코드나 성능 비교, 응용 분야 기반 추천이 필요하시면 언제든지 말씀 주세요!

[1]: https://arxiv.org/abs/1905.03304?utm_source=chatgpt.com "Deep Closest Point: Learning Representations for Point Cloud Registration"
[2]: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.891158/full?utm_source=chatgpt.com "Frontiers | DeepMatch: Toward Lightweight in Point Cloud Registration"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9339710/?utm_source=chatgpt.com "DeepMatch: Toward Lightweight in Point Cloud Registration - PMC"
[4]: https://arxiv.org/abs/2403.09996?utm_source=chatgpt.com "MEDPNet: Achieving High-Precision Adaptive Registration for Complex Die Castings"
[5]: https://arxiv.org/abs/2105.02714?utm_source=chatgpt.com "Deep Weighted Consensus: Dense correspondence confidence maps for 3D shape registration"
[6]: https://arxiv.org/html/2409.06267v1?utm_source=chatgpt.com "Mahalanobis k-NN: A Statistical Lens for Robust Point-Cloud Registrations"
[7]: https://www.mdpi.com/1424-8220/22/21/8217?utm_source=chatgpt.com "DOPNet: Achieving Accurate and Efficient Point Cloud Registration Based on Deep Learning and Multi-Level Features"
[8]: https://www.mdpi.com/2072-4292/15/14/3545?utm_source=chatgpt.com "PointCNT: A One-Stage Point Cloud Registration Approach Based on Complex Network Theory"

----------
----------

# 대표 논문 추천

### **PRNet: Self‑Supervised Learning for Partial‑to‑Partial Registration** (Wang & Solomon, 2019)

* **핵심 아이디어**: 부분적으로만 겹치는 두 포인트 클라우드를 정합하는 **완전 자가 지도 학습** 방식.
* **기술 요소**: 키포인트 감지기, 대응 쌍 예측, 기하적 표현 학습을 통합.
* **특징**: DCP와 PointNetLK를 뛰어넘는 성능, 특히 부분 겹침(partial overlap) 상황에서 강력함.([arXiv][1])

---

### **ROPNet: Representative Overlapping Points Network** (Zhu et al., 2021)

* **핵심 아이디어**: 부분 겹침 문제를 “partial → complete” 정합으로 변환. 대표 겹침 점(overlapping points)을 예측해 대응을 강화.
* **기술 요소**: global feature 기반 context-guided 모듈, Transformer로 특징 강화, weighted SVD로 변환 계산.
* **결과**: ModelNet40 기준 부분 겹침과 잡음 환경에서 최첨단 성능 달성.([arXiv][2])

---

### **ReAgent: Imitation & Reinforcement Learning 기반 정합** (Bauer et al., 2021)

* **핵심 아이디어**: 포인트 클라우드 정합을 강화학습(RL) 에이전트 역할로 모델링.
* **기술 요소**: 모방학습(imitation learning)으로 초기 정책 구성, 그 후 보상 기반 정책 최적화.
* **장점**: 초기값에 덜 민감하고 노이즈에도 강함. ModelNet40, ScanObjectNN 실험 및 LINEMOD 포즈 추정에서 SOTA 성능 달성.([arXiv][3])

---

### **UDPReg: Unsupervised Deep Probabilistic Registration** (Mei et al., 2023)

* **핵심 아이디어**: **비지도 학습 + 확률적 GMM 기반** 정합. 레이블 없이 학습 가능.
* **기술 요소**: 포인트 클라우드를 Gaussian Mixture Model로 표현, Sinkhorn 알고리즘으로 분포적 대응 계산, self-/cross-consistency와 contrastive loss로 학습.
* **결과**: 3DMatch/LoMatch, ModelNet 기반 벤치마크에서 경쟁력 있는 성능.([arXiv][4])

---

## 추가 방식들 (한눈 요약)

| 방법                             | 주요 특징                                                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **PointCNT**                   | 그래프 네트워크 기반 one-stage 방식, global feature로 바로 변환 예측, correspondence 불필요([MDPI][5])                                   |
| **DeepMatch**                  | per‑point feature 추출 후 간단한 conv + SVD, 리소스 효율적 딥러닝 정합([Frontiers][6])                                               |
| **Transformer & Attention 기반** | PREDATOR, Lepard, GeoTransformer, Peal 등 활용 – overlap-aware attention, transformer를 통한 대응 예측([MDPI][7], [arXiv][8]) |

---

## 요약 정리

* **PRNet**: 자가 지도 학습으로 키포인트 & 대응을 동시에 구축.
* **ROPNet**: 부분 겹침 문제를 대표 포인트 중심으로 해결.
* **ReAgent**: 강화학습 기반 에이전트를 통한 반복 정합.
* **UDPReg**: 비지도 + 확률적 접근(GMM+Sinkhorn).
* **PointCNT / DeepMatch / Transformer 기반 모델들**: 그래프 기반, 효율적 구조 또는 attention 활용 등 다양한 접근.

---

혹시 특정 논문들의 구현 코드, 비교 분석, 혹은 어떤 상황에서 유리한지에 대한 상세 정보가 필요하시면 말씀해주세요! 더 깊이 있게 도와드릴게요.

[1]: https://arxiv.org/abs/1910.12240?utm_source=chatgpt.com "PRNet: Self-Supervised Learning for Partial-to-Partial Registration"
[2]: https://arxiv.org/abs/2107.02583?utm_source=chatgpt.com "Point Cloud Registration using Representative Overlapping Points"
[3]: https://arxiv.org/abs/2103.15231?utm_source=chatgpt.com "ReAgent: Point Cloud Registration using Imitation and Reinforcement Learning"
[4]: https://arxiv.org/abs/2303.13290?utm_source=chatgpt.com "Unsupervised Deep Probabilistic Approach for Partial Point Cloud Registration"
[5]: https://www.mdpi.com/2072-4292/15/14/3545?utm_source=chatgpt.com "PointCNT: A One-Stage Point Cloud Registration Approach Based on Complex Network Theory"
[6]: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.891158/full?utm_source=chatgpt.com "Frontiers | DeepMatch: Toward Lightweight in Point Cloud Registration"
[7]: https://www.mdpi.com/1424-8220/22/21/8217?utm_source=chatgpt.com "DOPNet: Achieving Accurate and Efficient Point Cloud Registration Based on Deep Learning and Multi-Level Features"
[8]: https://arxiv.org/html/2404.14034v1?utm_source=chatgpt.com "PointDifformer: Robust Point Cloud Registration with Neural Diffusion and Transformer"

----------
----------

## 신체특화 주요 논문 요약

### 1. **HumanReg: Self-supervised Non-rigid Registration of Human Point Cloud** (2023)

* **주요 내용**: 사람 신체 포인트 클라우드의 **비강직 변형**을 다루기 위한 end-to-end self-supervised 학습 방식.
* **핵심 기술**: body prior 도입, 자체 합성 데이터셋(HumanSyn4D), 특수 손실 함수 설계.
* **성과**: CAPE‑512 데이터셋에서 최첨단 성능 달성, 실제 데이터에서도 우수한 정합 품질.([arXiv][1])

---

### 2. **Robust Human Registration with Body Part Segmentation on Noisy Point Clouds** (2025)

* **내용 요약**: 사람 신체 포인트 클라우드에서 각 점에 **신체 부위 레이블**을 예측하고, 이를 바탕으로 SMPL‑X 템플릿 fitting을 수행하는 **하이브리드 방식**.
* **핵심 기술**: body-part segmentation → centroid 기반 초기 포즈 추정 → 전체 정합(global refinement).
* **특징**: 잡음 많고 배경이 복잡한 현실 데이터(예: InterCap, EgoBody, BEHAVE)에서도 뛰어난 성능.([arXiv][2])

---

### 3. **Multilevel Active Registration for Kinect Human Body Scans** (2018)

* **접근 방식**: 고해상도 템플릿 메시를 **저품질 Kinect 스캔에 자동으로 변형하여 정합**.
* **핵심 기술**: body 전체와 각 부위별 **통계적 형태 모델(statistical shape models)** 기반의 coarse-to-fine fitting.
* **장점**: 수동 보정 없이 자동 정합 가능, 저비용 센서에서도 비교적 높은 정확도.([arXiv][3])

---

### 4. **Dense Human Body Correspondences Using Convolutional Networks** (2015)

* **핵심 아이디어**: 2D depth map 픽셀 수준에서 **body region classification**을 통해 **밀집 대응점(dense correspondence)** 학습.
* **특징**: 사람의 다양한 포즈 및 의복에도 견고한 real-time 대응 생성, correspondence 기반 정합에 활용 가능.([arXiv][4])

---

### 5. **A Framework for Accurate Point Cloud Based Registration of Full 3D Human Body Scans** (2017)

* **방법 요약**: 전체 3D body 스캔과 템플릿 간 **비강직 정합**을 위한 여러 단계 기반 파이프라인 제안.
* **주요 단계**: prior matches 설정 → global 및 partial non-rigid registration → 후처리.
* **응용 사례**: 애니메이션 가능한 가상 아바타 생성 등 실제 활용에도 적합.([DFKI][5])

---

## 간략 정리 테이블

| 논문명 (년도)                                   | 정합 방식          | 핵심 기술 및 특징                                              |
| ------------------------------------------ | -------------- | ------------------------------------------------------- |
| **HumanReg (2023)**                        | 자체 지도, 비강직     | body prior + self-supervised 학습, 높은 정확도                 |
| **Robust Human Registration (2025)**       | 부위 분류 기반 하이브리드 | segmentation 기반 SMPL‑X fitting                          |
| **Multilevel Active Registration (2018)**  | 템플릿 변형         | 통계 shape model 기반 coarse-to-fine                        |
| **Dense Correspondences (2015)**           | CNN 기반 대응점     | region classification 통한 real-time dense correspondence |
| **Accurate Registration Framework (2017)** | 여러 단계 정합       | fully automated non-rigid registration pipeline         |

---

## 추천 순서 및 활용 팁

1. **HumanReg** — 최신 self-supervised non-rigid 정합, high fidelity 정합이 필요하면 우선 추천.
2. **Robust Human Registration with Segmentation** — 잡음·클러터 많은 현실 데이터에서 강력.
3. **Dense Correspondences** — 실시간 대응점 생성 기반 정합, correspondence 활용 정합 시 유리.
4. **Multilevel Active Registration** — 저품질 Kinect 데이터 활용 시 유용.
5. **Accurate Registration Framework** — 전체 파이프라인 구조를 참고할 때 적절.

---

더 궁금한 점이 있으시면 언제든지 말씀해 주세요! 구현 코드, 데이터셋, 상세 비교 등도 도와드릴 수 있습니다.

[1]: https://arxiv.org/abs/2312.05462?utm_source=chatgpt.com "HumanReg: Self-supervised Non-rigid Registration of Human Point Cloud"
[2]: https://arxiv.org/abs/2504.03602?utm_source=chatgpt.com "Robust Human Registration with Body Part Segmentation on Noisy Point Clouds"
[3]: https://arxiv.org/abs/1811.10175?utm_source=chatgpt.com "Multilevel active registration for kinect human body scans: from low quality to high quality"
[4]: https://arxiv.org/abs/1511.05904?utm_source=chatgpt.com "Dense Human Body Correspondences Using Convolutional Networks"
[5]: https://www.dfki.de/web/forschung/projekte-publikationen/publikation/8996?utm_source=chatgpt.com "A Framework for an Accurate Point Cloud Based Registration of Full 3D Human Body Scans"


----------
----------

## 🎯 최종 목표 요약

**입력**: 4방향 뎁스맵 (Front, Back, Left, Right)
**목표**:

* 사람의 **완전한 3D 메쉬 복원**
* **정밀한 정렬 (sub-millimeter 수준)**
* 의복 포함 가능 or SMPL 기반 가능 여부는 선택 사항

---

## ✅ 최적의 정밀 정렬/복원 모델 추천 (Top 2)

### 🔹 1. **ICON (Implicit Clothed Humans)** – ⭐️ 최고 정밀도 + 신체 prior

* **특징**:

  * 뎁스맵 또는 이미지 기반 3D 복원
  * SMPL + implicit surface fusion → 의복 포함 정밀 복원 가능
* **정확도**:

  * sub-millimeter까지 가능한 수준
  * 여러 뷰(depth 또는 RGB)를 함께 학습 or 테스트할 수 있어 정렬 품질이 매우 뛰어남
* **장점**:

  * 사람이 휘어진 자세, 의복 포함한 경우도 잘 복원
  * 뎁스맵 → normal map으로 변환 후 사용하는 방식도 가능
* **입력 확장**:

  * 정면 기준으로 설계되었으나, 좌/우/후면도 입력에 포함 가능 (멀티 뷰 확장 구현 있음)

📌 [GitHub: ICON](https://github.com/YuliangXiu/ICON)

---

### 🔹 2. **PIFuHD (Pixel-Aligned Implicit Function – High Def)** – 고해상도 복원

* **특징**:

  * 1024x1024 resolution 기반의 사람 복원
  * implicit surface → smooth하고 고정밀 메쉬 복원 가능
* **정확도**:

  * 정면 1장만으로도 뛰어나며, 4방향 활용 시 더욱 정밀한 복원 가능
* **활용 방법**:

  * 각 뎁스맵을 입력으로 projection feature로 활용
  * 복수 뷰 입력하려면 네트워크 확장 or fusion 전략 필요

📌 [GitHub: PIFuHD](https://github.com/facebookresearch/pifuhd)

---

## 🔧 보조 처리 (정렬 정확도 ↑)

### 📌 정렬/융합 전처리 팁

| 단계                     | 방법                                        | 툴                    |
| ---------------------- | ----------------------------------------- | -------------------- |
| 뎁스맵 → 포인트 클라우드         | `Open3D.camera.PinholeCameraIntrinsic` 이용 | Open3D               |
| 정렬                     | RANSAC + Point-to-plane ICP               | Open3D               |
| Fusion                 | TSDF (Truncated Signed Distance Function) | Open3D, KinectFusion |
| Surface Reconstruction | Marching Cubes / Poisson Reconstruction   | MeshLab, Open3D      |

---

## 🧪 논문 기반 비교 요약

| 모델              | 구조                | 입력                    | 장점                     | 정밀도   |
| --------------- | ----------------- | --------------------- | ---------------------- | ----- |
| **ICON**        | SMPL + Implicit   | RGB/Depth (멀티뷰 가능)    | 의복 포함, 포즈 다양성          | ⭐⭐⭐⭐⭐ |
| **PIFuHD**      | Implicit surface  | RGB/Depth (멀티뷰 확장 가능) | 매우 고해상도, 부드러운 메쉬       | ⭐⭐⭐⭐☆ |
| **HumanReg**    | 비강직 등록            | Point cloud           | 정합 특화, self-supervised | ⭐⭐⭐⭐  |
| **MonoHuman**   | SMPL 기반 fitting   | RGB or depth          | 단일 뷰에서 mesh 예측         | ⭐⭐⭐   |
| **Open3D TSDF** | Volumetric fusion | Depth (멀티 뷰)          | 빠르고 직관적                | ⭐⭐☆   |

---

## ✅ 최종 추천 조합 (실제 적용 기준)

| 구성 단계             | 추천 도구/모델                           |
| ----------------- | ---------------------------------- |
| **1. 포인트 변환**     | Open3D로 뎁스맵 → 포인트셋                 |
| **2. 초기 정렬**      | Open3D의 RANSAC + ICP               |
| **3. 정밀 복원**      | **ICON** 또는 **PIFuHD (멀티 뷰 확장)**   |
| **4. 융합 및 메쉬 생성** | TSDF + Marching Cubes / Poisson    |
| **5. 메쉬 후처리**     | MeshLab / Blender / Instant Meshes |

---

## 💬 추가 옵션

* 📌 **학습 없는 방법** 원하면 → Open3D + Poisson Fusion + SMPL fitting (하지만 정밀도는 떨어짐)
* 📌 **리얼타임 추론이 목적**이면 ICON은 무겁고, 단순한 model-free 방식이 낫습니다.
* 📌 **정렬만 하고 mesh는 나중에 생성**한다면 → HumanReg + SMPLify-X 기반으로 가도 OK.

---

필요하시면:

* 실제 코드 예제 (Open3D + ICON 전처리)
* 4-view depth map → ICON 입력 변환 방법
* 성능 비교 테이블 (FPS, 정확도, 메모리 사용 등)

바로 정리해드릴게요. 어떤 게 더 필요하신가요?

