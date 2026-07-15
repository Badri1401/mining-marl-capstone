#!/usr/bin/env python3
"""
Generate Robot STL Meshes using Trimesh
========================================
Procedurally generates all robot component meshes as STL files.
No Blender required - uses pure Python geometry.

Robot Specs (from requirements):
- Chassis: 200mm diameter × 52mm height (dual-layer acrylic)
- Wheels: 65mm diameter × 26mm width (rubber)
- Caster: 15mm radius ball
- LiDAR mount: 30mm diameter × 20mm height cylinder
- Camera mount: 40mm × 30mm × 25mm box

Usage:
    python generate_robot_meshes.py [--output-dir PATH]
"""

import numpy as np
import argparse
import os
from pathlib import Path

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh not installed. Run: pip install trimesh")
    exit(1)


def create_chassis_mesh(diameter: float = 0.200, height: float = 0.052) -> trimesh.Trimesh:
    """
    Create chassis as a cylinder with flat top/bottom.
    
    Args:
        diameter: Chassis diameter in meters (default: 200mm)
        height: Chassis height in meters (default: 52mm)
    
    Returns:
        trimesh.Trimesh: Chassis mesh centered at origin
    """
    radius = diameter / 2.0
    
    # Create cylinder along Z-axis, centered at origin
    mesh = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=64  # Smooth circular profile
    )
    
    return mesh


def create_wheel_mesh(diameter: float = 0.065, width: float = 0.026) -> trimesh.Trimesh:
    """
    Create wheel as a cylinder (hub) with tread pattern suggestion.
    
    Args:
        diameter: Wheel diameter in meters (default: 65mm)
        width: Wheel width in meters (default: 26mm)
    
    Returns:
        trimesh.Trimesh: Wheel mesh, rotation axis along Y
    """
    radius = diameter / 2.0
    
    # Create cylinder along Y-axis (wheel rotation axis)
    # Trimesh creates along Z, so we'll rotate
    mesh = trimesh.creation.cylinder(
        radius=radius,
        height=width,
        sections=32
    )
    
    # Rotate 90 degrees around X to align cylinder axis with Y
    rotation = trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]
    )
    mesh.apply_transform(rotation)
    
    return mesh


def create_caster_mesh(radius: float = 0.015) -> trimesh.Trimesh:
    """
    Create caster wheel as a sphere.
    
    Args:
        radius: Caster ball radius in meters (default: 15mm)
    
    Returns:
        trimesh.Trimesh: Caster sphere mesh
    """
    mesh = trimesh.creation.icosphere(
        subdivisions=3,
        radius=radius
    )
    
    return mesh


def create_lidar_mount_mesh(diameter: float = 0.030, height: float = 0.020) -> trimesh.Trimesh:
    """
    Create LiDAR mount as a small cylinder on top of chassis.
    
    Args:
        diameter: Mount diameter in meters (default: 30mm)
        height: Mount height in meters (default: 20mm)
    
    Returns:
        trimesh.Trimesh: LiDAR mount mesh
    """
    radius = diameter / 2.0
    
    mesh = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=32
    )
    
    return mesh


def create_lidar_sensor_mesh(diameter: float = 0.070, height: float = 0.040) -> trimesh.Trimesh:
    """
    Create LiDAR sensor body (RPLidar A2 approximate dimensions).
    
    Args:
        diameter: Sensor diameter in meters (default: 70mm)
        height: Sensor height in meters (default: 40mm)
    
    Returns:
        trimesh.Trimesh: LiDAR sensor mesh
    """
    radius = diameter / 2.0
    
    mesh = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=48
    )
    
    return mesh


def create_camera_mount_mesh(
    width: float = 0.040, 
    depth: float = 0.030, 
    height: float = 0.025
) -> trimesh.Trimesh:
    """
    Create camera mount as a box bracket.
    
    Args:
        width: Mount width in meters (X-axis, default: 40mm)
        depth: Mount depth in meters (Y-axis, default: 30mm)
        height: Mount height in meters (Z-axis, default: 25mm)
    
    Returns:
        trimesh.Trimesh: Camera mount mesh
    """
    mesh = trimesh.creation.box(
        extents=[width, depth, height]
    )
    
    return mesh


def create_camera_sensor_mesh(
    width: float = 0.090,
    depth: float = 0.025,
    height: float = 0.025
) -> trimesh.Trimesh:
    """
    Create depth camera sensor body (Intel RealSense D435i approximate).
    
    Args:
        width: Camera width in meters (X-axis, default: 90mm)
        depth: Camera depth in meters (Y-axis, default: 25mm)
        height: Camera height in meters (Z-axis, default: 25mm)
    
    Returns:
        trimesh.Trimesh: Camera sensor mesh
    """
    mesh = trimesh.creation.box(
        extents=[width, depth, height]
    )
    
    return mesh


def generate_all_meshes(output_dir: str) -> dict:
    """
    Generate all robot component meshes and save as STL files.
    
    Args:
        output_dir: Directory to save STL files
    
    Returns:
        dict: Mapping of component names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    meshes = {
        'chassis': create_chassis_mesh(),
        'wheel': create_wheel_mesh(),
        'caster': create_caster_mesh(),
        'lidar_mount': create_lidar_mount_mesh(),
        'lidar_sensor': create_lidar_sensor_mesh(),
        'camera_mount': create_camera_mount_mesh(),
        'camera_sensor': create_camera_sensor_mesh(),
    }
    
    saved_files = {}
    
    for name, mesh in meshes.items():
        filepath = output_path / f"{name}.stl"
        mesh.export(str(filepath), file_type='stl')
        saved_files[name] = str(filepath)
        print(f"  ✓ Generated: {filepath.name} ({mesh.vertices.shape[0]} vertices)")
    
    return saved_files


def verify_meshes(output_dir: str) -> bool:
    """
    Verify all generated meshes are valid.
    
    Args:
        output_dir: Directory containing STL files
    
    Returns:
        bool: True if all meshes are valid
    """
    output_path = Path(output_dir)
    required_files = [
        'chassis.stl', 'wheel.stl', 'caster.stl',
        'lidar_mount.stl', 'lidar_sensor.stl',
        'camera_mount.stl', 'camera_sensor.stl'
    ]
    
    all_valid = True
    
    print("\n[VERIFY] Checking mesh validity...")
    
    for filename in required_files:
        filepath = output_path / filename
        
        if not filepath.exists():
            print(f"  ✗ Missing: {filename}")
            all_valid = False
            continue
        
        try:
            mesh = trimesh.load(str(filepath))
            
            # Check mesh properties
            is_watertight = mesh.is_watertight
            is_valid = mesh.is_volume
            vertex_count = mesh.vertices.shape[0]
            face_count = mesh.faces.shape[0]
            
            status = "✓" if is_valid else "⚠"
            watertight_str = "watertight" if is_watertight else "open"
            
            print(f"  {status} {filename}: {vertex_count} verts, {face_count} faces, {watertight_str}")
            
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {e}")
            all_valid = False
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Generate robot component STL meshes using trimesh"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for STL files (default: ../meshes relative to script)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify generated meshes after creation'
    )
    
    args = parser.parse_args()
    
    # Default output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'meshes'
    else:
        output_dir = Path(args.output_dir)
    
    print("="*60)
    print("MINING ROBOT MESH GENERATOR")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\n[GENERATE] Creating robot component meshes...")
    
    saved_files = generate_all_meshes(str(output_dir))
    
    print(f"\n[COMPLETE] Generated {len(saved_files)} mesh files")
    
    if args.verify:
        verify_meshes(str(output_dir))
    
    # Print summary
    print("\n" + "="*60)
    print("MESH SPECIFICATIONS")
    print("="*60)
    print("""
Component       | Dimensions (mm)      | Notes
----------------|----------------------|---------------------------
chassis         | Ø200 × H52           | Dual-layer acrylic body
wheel           | Ø65 × W26            | Rubber tread, Y-axis rotation
caster          | Ø30 (R15)            | Ball caster, passive
lidar_mount     | Ø30 × H20            | Mounting cylinder
lidar_sensor    | Ø70 × H40            | RPLidar A2 approximation
camera_mount    | 40 × 30 × 25         | Bracket for depth camera
camera_sensor   | 90 × 25 × 25         | RealSense D435i approximation
""")
    
    return saved_files


if __name__ == "__main__":
    main()
