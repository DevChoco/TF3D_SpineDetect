"""
Mesh Generation and Saving Module (Advanced Vertex Reduction)

This module provides the following features:
- Generate mesh from point cloud
- Poisson surface reconstruction
- Ball Pivoting Algorithm
- Advanced vertex reduction and mesh optimization
- Save in various formats (OBJ, PLY, STL)
"""

import numpy as np
import open3d as o3d
import os
from modules.mesh_optimizer import (
    smart_vertex_reduction, 
    adaptive_mesh_optimization,
    create_lod_hierarchy,
    measure_optimization_quality,
    save_optimized_mesh,
    analyze_mesh_complexity
)


def create_mesh_from_pointcloud(pcd):
    """
    Generate mesh from point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): Open3D PointCloud object
    
    Returns:
        o3d.geometry.TriangleMesh: Open3D TriangleMesh object or None
    """
    try:
        print(f"Point cloud info: {len(pcd.points)} points")
        
        if len(pcd.points) < 100:
            print("Too few points to generate mesh.")
            return None
        
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        print("Generating mesh using Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=9,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"Generated mesh info: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        
        mesh.compute_vertex_normals()
        
        if pcd.has_colors():
            avg_color = np.mean(np.asarray(pcd.colors), axis=0)
            mesh.paint_uniform_color(avg_color)
        
        return mesh
        
    except Exception as e:
        print(f"Error during mesh generation: {e}")
        
        try:
            print("Trying Ball Pivoting Algorithm...")
            
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 2 * avg_dist
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector([radius, radius * 2])
            )
            
            if len(mesh.triangles) > 0:
                print(f"Mesh generated with Ball Pivoting: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                mesh.compute_vertex_normals()
                return mesh
            else:
                print("Ball Pivoting mesh generation failed")
                return None
                
        except Exception as e2:
            print(f"Error during Ball Pivoting mesh generation: {e2}")
            return None


def simplify_mesh(mesh, reduction_ratio=0.5, method="quadric", preserve_boundary=True, adaptive=False):
    """
    Reduce mesh complexity (Advanced Vertex Reduction).
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh to simplify
        reduction_ratio (float): Triangle reduction ratio (0.1 = 90% reduction, 0.5 = 50% reduction)
        method (str): Simplification method ("quadric", "cluster", "average", "adaptive")
        preserve_boundary (bool): Whether to preserve boundaries
        adaptive (bool): Whether to use adaptive simplification
        
    Returns:
        o3d.geometry.TriangleMesh: Simplified mesh
    """
    if mesh is None:
        return None
    
    original_triangles = len(mesh.triangles)
    original_vertices = len(mesh.vertices)
    target_triangles = max(100, int(original_triangles * reduction_ratio))
    
    print(f"Advanced mesh simplification started:")
    print(f"  Original: {original_vertices:,} vertices, {original_triangles:,} triangles")
    print(f"  Target: {target_triangles:,} triangles (reduction: {(1-reduction_ratio)*100:.1f}%)")
    print(f"  Method: {method}, boundary preservation: {preserve_boundary}, adaptive: {adaptive}")
    
    try:
        if method == "quadric" or method == "adaptive":
            if adaptive:
                mesh.compute_vertex_normals()
                simplified_mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=target_triangles,
                    maximum_error=0.01,
                    boundary_weight=1.0 if preserve_boundary else 0.1
                )
            else:
                simplified_mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=target_triangles
                )
            
        elif method == "cluster":
            bbox_extent = mesh.get_axis_aligned_bounding_box().get_extent().max()
            if adaptive:
                complexity_factor = min(2.0, original_triangles / 10000)
                voxel_size = bbox_extent / (100 * complexity_factor)
            else:
                voxel_size = bbox_extent / 100
                
            simplified_mesh = mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average
            )
            
        elif method == "progressive":
            simplified_mesh = mesh
            steps = 5
            step_ratio = pow(reduction_ratio, 1.0/steps)
            
            for i in range(steps):
                step_target = max(100, int(len(simplified_mesh.triangles) * step_ratio))
                print(f"    Step {i+1}/{steps}: {len(simplified_mesh.triangles):,} → {step_target:,} triangles")
                simplified_mesh = simplified_mesh.simplify_quadric_decimation(
                    target_number_of_triangles=step_target
                )
                
        elif method == "edge_collapse":
            voxel_size = mesh.get_axis_aligned_bounding_box().get_extent().max() / 150
            simplified_mesh = mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Quadric
            )
            
        else:
            print(f"Unknown simplification method: {method}. Using Quadric method.")
            simplified_mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles
            )
        
        final_triangles = len(simplified_mesh.triangles)
        final_vertices = len(simplified_mesh.vertices)
        actual_reduction = (original_triangles - final_triangles) / original_triangles * 100
        
        print(f"  Result: {final_vertices:,} vertices, {final_triangles:,} triangles")
        print(f"  Actual reduction: {actual_reduction:.1f}%")
        
        quality_score = measure_mesh_quality(simplified_mesh)
        print(f"  Mesh quality score: {quality_score:.3f}/1.000")
        
        return simplified_mesh
        
    except Exception as e:
        print(f"Error during mesh simplification: {e}")
        return mesh


def measure_mesh_quality(mesh):
    """
    Measure mesh quality.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh to measure quality
        
    Returns:
        float: Quality score (0.0 ~ 1.0, higher is better)
    """
    if mesh is None or len(mesh.triangles) == 0:
        return 0.0
    
    try:
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        
        areas = []
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas.append(area)
        
        areas = np.array(areas)
        area_variance = np.var(areas) / (np.mean(areas) + 1e-8)
        area_score = 1.0 / (1.0 + area_variance)
        
        mesh_copy = mesh.copy()
        mesh_copy.remove_non_manifold_edges()
        manifold_score = len(mesh_copy.triangles) / len(mesh.triangles)
        
        quality_score = 0.6 * area_score + 0.4 * manifold_score
        
        return min(1.0, max(0.0, quality_score))
        
    except Exception as e:
        print(f"Error measuring mesh quality: {e}")
        return 0.5


def optimize_mesh(mesh, enable_simplification=True, reduction_ratio=0.3, optimization_level="standard"):
    """
    Perform advanced mesh optimization (including advanced vertex reduction).
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh to optimize
        enable_simplification (bool): Whether to enable mesh simplification
        reduction_ratio (float): Mesh simplification ratio
        optimization_level (str): Optimization level ("fast", "standard", "high_quality")
        
    Returns:
        o3d.geometry.TriangleMesh: Optimized mesh
    """
    if mesh is None:
        return None
    
    try:
        print(f"\nAdvanced mesh optimization started... (level: {optimization_level})")
        initial_quality = measure_mesh_quality(mesh)
        print(f"Initial mesh quality: {initial_quality:.3f}")
        
        print("Step 1: Basic mesh cleanup...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        if enable_simplification:
            print("Step 2: Intelligent mesh complexity reduction...")
            
            if optimization_level == "fast":
                mesh = simplify_mesh(mesh, reduction_ratio=reduction_ratio, 
                                   method="cluster", preserve_boundary=False, adaptive=False)
                
            elif optimization_level == "high_quality":
                mesh = simplify_mesh(mesh, reduction_ratio=reduction_ratio, 
                                   method="adaptive", preserve_boundary=True, adaptive=True)
                
            else:
                mesh = simplify_mesh(mesh, reduction_ratio=reduction_ratio, 
                                   method="quadric", preserve_boundary=True, adaptive=False)
        
        print("Step 3: Adaptive mesh smoothing...")
        if optimization_level == "fast":
            smooth_iterations = 1
        elif optimization_level == "high_quality":
            smooth_iterations = 5
        else:
            smooth_iterations = 3
            
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)
        
        print("Step 4: Normal vector recalculation...")
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        final_quality = measure_mesh_quality(mesh)
        print(f"Final mesh quality: {final_quality:.3f} (improvement: {final_quality-initial_quality:+.3f})")
        
        print("Advanced mesh optimization complete!")
        return mesh
        
    except Exception as e:
        print(f"Error during mesh optimization: {e}")
        return mesh
def create_and_save_mesh(pcd, output_dir="output/3d_models", base_filename="body_mesh", 
                        create_lod=True, reduction_ratio=0.3, optimization_level="standard",
                        custom_lod_levels=None, enable_quality_analysis=True, enable_hole_filling=False):
    """
    Generate high-quality mesh from point cloud and apply advanced vertex reduction.
    
    Args:
        pcd (o3d.geometry.PointCloud): Point cloud
        output_dir (str): Output directory
        base_filename (str): Base filename
        create_lod (bool): Whether to create multiple LOD level meshes
        reduction_ratio (float): Default mesh simplification ratio
        optimization_level (str): Optimization level ("fast", "standard", "high_quality")
        custom_lod_levels (dict): Custom LOD levels
        enable_quality_analysis (bool): Whether to enable quality analysis
        enable_hole_filling (bool): (Deprecated) Kept for backward compatibility
        
    Returns:
        tuple: (mesh object, saved file paths list)
    """
    print("\n=== Advanced Mesh Generation and Vertex Reduction ===")
    print("Converting point cloud to high-quality mesh and applying intelligent vertex reduction...")
    
    mesh = create_mesh_from_pointcloud(pcd)
    
    saved_files = []
    
    if mesh is not None:
        print(f"\nOriginal mesh info: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
        
        if enable_quality_analysis:
            complexity_analysis = analyze_mesh_complexity(mesh)
            print(f"Mesh complexity score: {complexity_analysis['complexity_score']:.3f}")
        
        print(f"\nApplying intelligent vertex reduction... (target reduction: {(1-reduction_ratio)*100:.1f}%)")
        
        if optimization_level == "high_quality":
            optimized_mesh = adaptive_mesh_optimization(mesh, complexity_level="auto")
            optimized_mesh = smart_vertex_reduction(optimized_mesh, target_ratio=reduction_ratio, quality_priority=True)
            
        elif optimization_level == "fast":
            optimized_mesh = smart_vertex_reduction(mesh, target_ratio=reduction_ratio, quality_priority=False)
            
        else:
            optimized_mesh = smart_vertex_reduction(mesh, target_ratio=reduction_ratio, quality_priority=True)
        
        if enable_quality_analysis:
            quality_info = measure_optimization_quality(mesh, optimized_mesh)
            print(f"\n=== Final Optimization Results ===")
            print(f"Vertex reduction: {quality_info['vertex_reduction_percent']:.1f}%")
            print(f"Triangle reduction: {quality_info['triangle_reduction_percent']:.1f}%")
            print(f"Surface area preservation: {quality_info['area_preservation_percent']:.1f}%")
            print(f"Overall quality score: {quality_info['overall_quality_score']:.1f}/100")
        else:
            quality_info = None
        
        saved_files = save_optimized_mesh(optimized_mesh, output_dir, base_filename, quality_info)
        
        if create_lod:
            print("\n=== Multi-LOD Mesh Generation ===")
            lod_meshes = create_lod_hierarchy(mesh, custom_lod_levels)
            
            for lod_name, lod_mesh in lod_meshes.items():
                lod_filename = f"{base_filename}_{lod_name}"
                
                if enable_quality_analysis:
                    lod_quality = measure_optimization_quality(mesh, lod_mesh)
                else:
                    lod_quality = None
                
                lod_files = save_optimized_mesh(lod_mesh, output_dir, lod_filename, lod_quality)
                saved_files.extend(lod_files)
            
            print(f"\nTotal {len(lod_meshes)} LOD levels created.")
            print(f"Total files saved: {len(saved_files)}")
        
        print(f"\n=== Mesh Generation and Optimization Complete ===")
        print("Optimized with intelligent vertex reduction")
        print(f"Total {len(saved_files)} files saved")
        
        return optimized_mesh, saved_files
    else:
        print("Mesh generation failed.")
        return None, []