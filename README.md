# Spinal Line Detection for Posture Evaluation through Train-ing-free 3D Human Body Reconstruction with 2D Depth Images
> STRIDE : **S**pinal line de**T**ection for postu**R**e evaluat**I**on via training-free 3D human bo**D**y r**E**construction with 2D depth images

## Abstract
The spinal angle is an important indicator of body balance. It is important to restore the 3D shape of the human body and estimate the spine center line. Existing mul-ti-image-based body restoration methods require expensive equipment and complex pro-cedures, and single image-based body restoration methods have limitations in that it is difficult to accurately estimate the internal structure such as the spine center line due to occlusion and viewpoint limitation. This study proposes a method to compensate for the shortcomings of the multi-image-based method and to solve the limitations of the sin-gle-image method. We propose a 3D body posture analysis system that integrates depth images from four directions to restore a 3D human model and automatically estimate the spine center line. Through hierarchical matching of global and fine registration, restora-tion to noise and occlusion is performed. Also, the Adaptive Vertex Reduction is applied to maintain the resolution and shape reliability of the mesh, and the accuracy and stabil-ity of spinal angle estimation are simultaneously secured by using the Level of Detail en-semble. The proposed method achieves high-precision 3D spine registration estimation without relying on training data or complex neural network models, and the verification confirms the improvement of matching quality.

## Installation

```
python main.py
python -m http.server 8000
```
- Web access : `http://localhost:8000/Test_viewer/obj_all_lod_comparison_viewer.html`
