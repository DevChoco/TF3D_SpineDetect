"""
MediaPipe 기반 척추 검출 테스트 스크립트
t5.py - 빠른 테스트 및 데모용
"""

from mediapipe_spine_detector import MediaPipeSpineDetector
import os


def test_basic_detection():
    """기본 척추 검출 테스트"""
    print("=" * 60)
    print("테스트 1: 기본 척추 검출")
    print("=" * 60)

    mesh_path = 'skplx_SK_test/3d_file/body_mesh_fpfh.obj'

    if not os.path.exists(mesh_path):
        print(f"오류: 메쉬 파일을 찾을 수 없습니다: {mesh_path}")
        return
    
    # 검출기 생성
    detector = MediaPipeSpineDetector(
        model_complexity=2,      # 최고 품질
        smooth_landmarks=True
    )
    
    # 척추 검출
    result = detector.detect_spine_from_mesh(
        mesh_path=mesh_path,
        views=['front', 'side'],  # 정면 + 측면
        min_visibility=0.5,
        refine_with_mesh=True
    )
    
    # 결과 출력
    if result['success']:
        print("\n✓ 검출 성공!")
        print(f"  - 총 척추 키포인트: {len(result['spine_keypoints'])}개")
        print(f"  - 평균 신뢰도: {result['statistics']['avg_confidence']:.3f}")
        print(f"  - 사용된 뷰: {', '.join(result['views_used'])}")
        print(f"  - 정제된 키포인트: {result['statistics']['refined_keypoints']}개")
        
        # 일부 키포인트 출력
        print("\n주요 척추 키포인트:")
        important_keypoints = ['C7', 'T6', 'T12', 'L3', 'Sacrum']
        for kp in result['spine_keypoints']:
            if kp['name'] in important_keypoints:
                refined_mark = "✓" if kp.get('refined', False) else "✗"
                print(f"  [{refined_mark}] {kp['name']:8s}: "
                      f"({kp['x']:6.3f}, {kp['y']:6.3f}, {kp['z']:6.3f}) "
                      f"confidence={kp['confidence']:.3f}")
        
        # 결과 저장
        os.makedirs('3d_file/spine_detection_results', exist_ok=True)
        json_path = '3d_file/spine_detection_results/test_mediapipe_result.json'
        detector.save_results(result, json_path)
        
        return result
    else:
        print(f"\n✗ 검출 실패: {result.get('error', '알 수 없는 오류')}")
        return None


def test_multiview_comparison():
    """다중 뷰 비교 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: 다중 뷰 정확도 비교")
    print("=" * 60)

    mesh_path = 'skplx_SK_test/3d_file/body_mesh_fpfh.obj'

    if not os.path.exists(mesh_path):
        print(f"오류: 메쉬 파일을 찾을 수 없습니다: {mesh_path}")
        return
    
    detector = MediaPipeSpineDetector(model_complexity=1)
    
    test_cases = [
        (['front'], '정면만'),
        (['side'], '측면만'),
        (['front', 'side'], '정면+측면'),
        (['front', 'side', 'back'], '정면+측면+후면'),
    ]
    
    results = []
    
    for views, description in test_cases:
        print(f"\n테스트 중: {description} ({', '.join(views)})")
        
        result = detector.detect_spine_from_mesh(
            mesh_path=mesh_path,
            views=views,
            min_visibility=0.5,
            refine_with_mesh=True
        )
        
        if result['success']:
            stats = result['statistics']
            print(f"  ✓ 키포인트: {stats['total_keypoints']}개, "
                  f"평균 신뢰도: {stats['avg_confidence']:.3f}")
            results.append((description, stats))
        else:
            print(f"  ✗ 실패")
    
    # 비교 결과 출력
    if results:
        print("\n" + "-" * 60)
        print("뷰 구성별 성능 비교:")
        print("-" * 60)
        print(f"{'설정':<20s} {'키포인트':<12s} {'평균 신뢰도':<12s} {'정제율'}")
        print("-" * 60)
        
        for description, stats in results:
            refine_rate = stats['refined_keypoints'] / stats['total_keypoints'] * 100
            print(f"{description:<20s} {stats['total_keypoints']:<12d} "
                  f"{stats['avg_confidence']:<12.3f} {refine_rate:.1f}%")


def test_visibility_threshold():
    """Visibility 임계값 비교 테스트"""
    print("\n" + "=" * 60)
    print("테스트 3: Visibility 임계값 영향")
    print("=" * 60)
    
    mesh_path = 'skplx_SK_test/3d_file/body_mesh_fpfh.obj'
    
    if not os.path.exists(mesh_path):
        print(f"오류: 메쉬 파일을 찾을 수 없습니다: {mesh_path}")
        return
    
    detector = MediaPipeSpineDetector(model_complexity=1)
    
    thresholds = [0.3, 0.5, 0.7]
    
    print("\nVisibility 임계값별 검출 키포인트 수:")
    print("-" * 60)
    
    for threshold in thresholds:
        result = detector.detect_spine_from_mesh(
            mesh_path=mesh_path,
            views=['front', 'side'],
            min_visibility=threshold,
            refine_with_mesh=True
        )
        
        if result['success']:
            stats = result['statistics']
            print(f"임계값 {threshold:.1f}: "
                  f"{stats['total_keypoints']}개 키포인트, "
                  f"평균 신뢰도 {stats['avg_confidence']:.3f}")


def test_visualization():
    """시각화 테스트"""
    print("\n" + "=" * 60)
    print("테스트 4: 결과 시각화")
    print("=" * 60)
    
    mesh_path = 'skplx_SK_test/3d_file/body_mesh_fpfh.obj'
    
    if not os.path.exists(mesh_path):
        print(f"오류: 메쉬 파일을 찾을 수 없습니다: {mesh_path}")
        return
    
    detector = MediaPipeSpineDetector(model_complexity=2)
    
    result = detector.detect_spine_from_mesh(
        mesh_path=mesh_path,
        views=['front', 'side', 'back'],
        min_visibility=0.5,
        refine_with_mesh=True
    )
    
    if result['success']:
        output_path = '3d_file/spine_detection_results/test_mediapipe_visualization.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n시각화 생성 중... ({output_path})")
        detector.visualize_results(result, output_path)
        print("✓ 시각화 완료!")
    else:
        print("✗ 검출 실패, 시각화를 생성할 수 없습니다.")


def quick_demo():
    """빠른 데모 (명령줄 인자 없이 실행)"""
    print("\n" + "=" * 60)
    print("MediaPipe 척추 검출 빠른 데모")
    print("=" * 60)
    
    mesh_path = 'skplx_SK_test/3d_file/body_mesh_fpfh.obj'
    
    if not os.path.exists(mesh_path):
        print(f"\n오류: 메쉬 파일을 찾을 수 없습니다: {mesh_path}")
        print("skplx_SK_test/3d_file/body_mesh_fpfh.obj 파일이 있는지 확인하세요.")
        return
    
    print(f"\n메쉬 파일: {mesh_path}")
    print("설정: 정면+측면 뷰, 최고 품질 모델")
    
    detector = MediaPipeSpineDetector(
        model_complexity=2,
        smooth_landmarks=True
    )
    
    result = detector.detect_spine_from_mesh(
        mesh_path=mesh_path,
        views=['front', 'side'],
        min_visibility=0.5,
        refine_with_mesh=True
    )
    
    if result['success']:
        print("\n" + "=" * 60)
        print("검출 결과 요약")
        print("=" * 60)
        
        stats = result['statistics']
        print(f"✓ 총 척추 키포인트: {stats['total_keypoints']}개")
        print(f"✓ 정제된 키포인트: {stats['refined_keypoints']}개 "
              f"({stats['refined_keypoints']/stats['total_keypoints']*100:.1f}%)")
        print(f"✓ 평균 신뢰도: {stats['avg_confidence']:.3f}")
        print(f"✓ 검출된 뷰: {stats['views_detected']}개")
        
        # 척추 세그먼트별 통계
        print("\n척추 세그먼트별 키포인트:")
        segments = {}
        for kp in result['spine_keypoints']:
            level = kp['level']
            if level not in segments:
                segments[level] = []
            segments[level].append(kp)
        
        for level, keypoints in segments.items():
            avg_conf = sum(kp['confidence'] for kp in keypoints) / len(keypoints)
            print(f"  - {level:20s}: {len(keypoints):2d}개 (평균 신뢰도: {avg_conf:.3f})")
        
        # 저장
        os.makedirs('3d_file/spine_detection_results', exist_ok=True)
        json_path = '3d_file/spine_detection_results/demo_result.json'
        vis_path = '3d_file/spine_detection_results/demo_visualization.png'
        
        detector.save_results(result, json_path)
        detector.visualize_results(result, vis_path)
        
        print(f"\n저장 완료:")
        print(f"  - JSON: {json_path}")
        print(f"  - 시각화: {vis_path}")
        
    else:
        print(f"\n✗ 검출 실패: {result.get('error', '알 수 없는 오류')}")


def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) > 1:
        # 명령줄 인자가 있으면 특정 테스트 실행
        test_name = sys.argv[1]
        
        if test_name == 'basic':
            test_basic_detection()
        elif test_name == 'multiview':
            test_multiview_comparison()
        elif test_name == 'visibility':
            test_visibility_threshold()
        elif test_name == 'visualize':
            test_visualization()
        elif test_name == 'all':
            test_basic_detection()
            test_multiview_comparison()
            test_visibility_threshold()
            test_visualization()
        else:
            print(f"사용법: python t5.py [basic|multiview|visibility|visualize|all]")
            print("또는 인자 없이 실행하면 빠른 데모가 실행됩니다.")
    else:
        # 인자 없으면 빠른 데모
        quick_demo()


if __name__ == '__main__':
    main()
