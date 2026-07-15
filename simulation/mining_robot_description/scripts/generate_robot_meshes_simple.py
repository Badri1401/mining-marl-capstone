#!/usr/bin/env python3
"""
Simple Robot Mesh Generator for Mining Robot
Generates STL files using pure Python/NumPy (no trimesh dependency)
Based on robot specifications:
- Chassis: 200mm diameter x 52mm height dual-layer acrylic
- Wheels: 65mm x 26mm rubber wheels  
- Mass: ~650g total

Author: Mining Robot Team
"""

import numpy as np
import struct
import os
from pathlib import Path


class STLWriter:
    """Simple STL file writer using pure Python"""
    
    @staticmethod
    def write_stl(filename: str, vertices: np.ndarray, faces: np.ndarray):
        """
        Write a binary STL file
        
        Args:
            filename: Output filename
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
        """
        # Calculate face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normals = normals / norms
        
        # Write binary STL
        with open(filename, 'wb') as f:
            # 80 byte header
            f.write(b'\x00' * 80)
            
            # Number of triangles
            f.write(struct.pack('<I', len(faces)))
            
            # Write each triangle
            for i, face in enumerate(faces):
                # Normal vector
                f.write(struct.pack('<fff', *normals[i]))
                
                # Three vertices
                for vi in face:
                    f.write(struct.pack('<fff', *vertices[vi]))
                
                # Attribute byte count (unused, set to 0)
                f.write(struct.pack('<H', 0))
        
        print(f"  Written: {filename} ({len(faces)} triangles)")


def create_cylinder_mesh(radius: float, height: float, segments: int = 32) -> tuple:
    """
    Create a cylinder mesh
    
    Returns:
        (vertices, faces) tuple
    """
    vertices = []
    faces = []
    
    # Create top and bottom circle vertices
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    
    # Bottom center vertex
    vertices.append([0, 0, -height/2])
    # Bottom ring vertices
    for angle in angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, -height/2])
    
    # Top center vertex
    top_center_idx = len(vertices)
    vertices.append([0, 0, height/2])
    # Top ring vertices  
    for angle in angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, height/2])
    
    # Bottom cap faces (fan triangulation)
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([0, i + 1, next_i + 1])
    
    # Top cap faces (fan triangulation)
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([top_center_idx, top_center_idx + next_i + 1, top_center_idx + i + 1])
    
    # Side faces (quads split into triangles)
    for i in range(segments):
        next_i = (i + 1) % segments
        bottom1 = i + 1
        bottom2 = next_i + 1
        top1 = top_center_idx + i + 1
        top2 = top_center_idx + next_i + 1
        
        # Two triangles per quad
        faces.append([bottom1, bottom2, top1])
        faces.append([bottom2, top2, top1])
    
    return np.array(vertices), np.array(faces)


def create_box_mesh(width: float, depth: float, height: float) -> tuple:
    """
    Create a box mesh
    
    Returns:
        (vertices, faces) tuple
    """
    w, d, h = width/2, depth/2, height/2
    
    vertices = np.array([
        # Bottom face
        [-w, -d, -h],  # 0
        [ w, -d, -h],  # 1
        [ w,  d, -h],  # 2
        [-w,  d, -h],  # 3
        # Top face
        [-w, -d,  h],  # 4
        [ w, -d,  h],  # 5
        [ w,  d,  h],  # 6
        [-w,  d,  h],  # 7
    ])
    
    faces = np.array([
        # Bottom
        [0, 2, 1],
        [0, 3, 2],
        # Top
        [4, 5, 6],
        [4, 6, 7],
        # Front
        [0, 1, 5],
        [0, 5, 4],
        # Back
        [2, 3, 7],
        [2, 7, 6],
        # Left
        [0, 4, 7],
        [0, 7, 3],
        # Right
        [1, 2, 6],
        [1, 6, 5],
    ])
    
    return vertices, faces


def create_sphere_mesh(radius: float, rings: int = 16, segments: int = 32) -> tuple:
    """
    Create a UV sphere mesh
    
    Returns:
        (vertices, faces) tuple
    """
    vertices = []
    faces = []
    
    # Create vertices
    for ring in range(rings + 1):
        phi = np.pi * ring / rings  # 0 to pi
        for seg in range(segments):
            theta = 2 * np.pi * seg / segments
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            vertices.append([x, y, z])
    
    # Create faces
    for ring in range(rings):
        for seg in range(segments):
            next_seg = (seg + 1) % segments
            
            v0 = ring * segments + seg
            v1 = ring * segments + next_seg
            v2 = (ring + 1) * segments + seg
            v3 = (ring + 1) * segments + next_seg
            
            # Skip degenerate triangles at poles
            if ring > 0:
                faces.append([v0, v2, v1])
            if ring < rings - 1:
                faces.append([v1, v2, v3])
    
    return np.array(vertices), np.array(faces)


def generate_chassis_mesh(output_path: str):
    """
    Generate chassis mesh: 200mm diameter x 52mm dual-layer
    """
    print("Generating chassis mesh...")
    
    # Main chassis cylinder
    radius = 0.100  # 100mm = 200mm diameter
    height = 0.052  # 52mm total height
    
    vertices, faces = create_cylinder_mesh(radius, height, segments=48)
    STLWriter.write_stl(output_path, vertices, faces)


def generate_wheel_mesh(output_path: str):
    """
    Generate wheel mesh: 65mm diameter x 26mm width
    """
    print("Generating wheel mesh...")
    
    radius = 0.0325  # 32.5mm
    width = 0.026    # 26mm width
    
    # Wheel is oriented so axle goes along Y axis
    vertices, faces = create_cylinder_mesh(radius, width, segments=32)
    
    # Rotate 90 degrees around X to orient correctly
    # Wheel axis should be Y, not Z
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    vertices = vertices @ rotation_matrix.T
    
    STLWriter.write_stl(output_path, vertices, faces)


def generate_caster_mesh(output_path: str):
    """
    Generate caster ball mesh: 15mm diameter
    """
    print("Generating caster mesh...")
    
    radius = 0.0075  # 7.5mm
    
    vertices, faces = create_sphere_mesh(radius, rings=12, segments=24)
    STLWriter.write_stl(output_path, vertices, faces)


def generate_lidar_mount_mesh(output_path: str):
    """
    Generate LiDAR mount/housing mesh
    RPLiDAR A2 approximate dimensions: 70mm diameter x 41mm height
    """
    print("Generating LiDAR mount mesh...")
    
    radius = 0.035   # 35mm radius
    height = 0.041   # 41mm height
    
    vertices, faces = create_cylinder_mesh(radius, height, segments=32)
    STLWriter.write_stl(output_path, vertices, faces)


def generate_camera_mount_mesh(output_path: str):
    """
    Generate camera mount mesh
    Intel RealSense D435i approximate dimensions: 90mm x 25mm x 25mm
    """
    print("Generating camera mount mesh...")
    
    width = 0.090   # 90mm
    depth = 0.025   # 25mm
    height = 0.025  # 25mm
    
    vertices, faces = create_box_mesh(width, depth, height)
    STLWriter.write_stl(output_path, vertices, faces)


def main():
    """Generate all robot mesh files"""
    print("=" * 60)
    print("Mining Robot Mesh Generator (Simple/Pure Python)")
    print("=" * 60)
    print()
    
    # Get script directory and mesh output directory
    script_dir = Path(__file__).parent.absolute()
    mesh_dir = script_dir.parent / "meshes"
    
    # Create mesh directory if it doesn't exist
    mesh_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {mesh_dir}")
    print()
    
    # Generate all meshes
    print("Generating robot component meshes...")
    print("-" * 40)
    
    generate_chassis_mesh(str(mesh_dir / "chassis.stl"))
    generate_wheel_mesh(str(mesh_dir / "wheel.stl"))
    generate_caster_mesh(str(mesh_dir / "caster_ball.stl"))
    generate_lidar_mount_mesh(str(mesh_dir / "lidar_mount.stl"))
    generate_camera_mount_mesh(str(mesh_dir / "camera_mount.stl"))
    
    print()
    print("-" * 40)
    print("All meshes generated successfully!")
    print()
    
    # List generated files
    print("Generated files:")
    for stl_file in sorted(mesh_dir.glob("*.stl")):
        size_kb = stl_file.stat().st_size / 1024
        print(f"  - {stl_file.name}: {size_kb:.1f} KB")
    
    print()
    print("=" * 60)
    print("Mesh generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
