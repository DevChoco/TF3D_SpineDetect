@echo off
echo ============================================================
echo MediaPipe 척추 검출 시스템 설치 및 실행
echo ============================================================
echo.

echo [1/3] 필수 라이브러리 설치 중...
pip install "pyglet<2" pyopengl
if %errorlevel% neq 0 (
    echo 오류: pyglet 설치 실패
    pause
    exit /b 1
)

echo.
echo [2/3] 추가 의존성 확인 중...
pip install mediapipe opencv-python numpy scipy trimesh pyrender Pillow matplotlib tqdm
if %errorlevel% neq 0 (
    echo 경고: 일부 라이브러리 설치 실패
)

echo.
echo [3/3] 척추 검출 데모 실행 중...
python t5.py

echo.
echo ============================================================
echo 완료!
echo ============================================================
pause
