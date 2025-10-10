# 3D_Body_Posture_Analysis

## Main Pipline
- `Fixed_Main_3D_pose` : `Main_3D_pose` + Vertex reduce + Mesh 채우기 + 와이어 프레임 투명
- `Main_3D_pose` : FPFH를 이용한 정렬수행 + 시각화

## 3d_models Viewer
> 3d_models 비교분석을 위해만듬

```
> 사용자의 PC환경에 수정하여 사용

cd "d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\Fixed_Main_3D_pose"; python -m http.server 8000
```
- 접속 : `http://localhost:8000/obj_viewer.html`

-------------

- `3d_pose1_main.py` : ICP 정렬 테스트 1 (실패)
- `3d_pose2_main.py` : ICP 정렬 테스트 2 (가능)
- `3d_pose2_main_addmask.py` : `3d_pose2_main.py`에 마스킹 전처리 과정추가 (가능)
- `3d_pose3_main_FPFH.py` : FPFH모델을 이용한 테스트 (가능)

- `3d_pose2-5_main.py` : 스켈레톤 파싱 테스트 (실패)
- `3d_pose2-6_main.py` : 스켈레톤 파싱 테스트 (실패)
- `3d_pose3_main_pose.py` : 스켈레톤 파싱 테스트 (가능)
- `3d_pose4_main_pose.py` : 스켈레톤 파싱 테스트 (실패)
