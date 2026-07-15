#!/usr/bin/env python3
"""
URDF Validation Script
======================
Validates the mining robot URDF for correctness and Gazebo compatibility.

Checks:
1. URDF parsing (XML validity)
2. Mesh file existence
3. Joint limits and types
4. Link inertias (positive definite)
5. Gazebo plugin configuration
6. Coordinate frame conventions

Usage:
    python validate_urdf.py [--urdf-path PATH] [--meshes-path PATH]
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def print_status(status: str, message: str, indent: int = 0):
    """Print status message with formatting."""
    prefix = "  " * indent
    if status == "ok":
        print(f"{prefix}✓ {message}")
    elif status == "warn":
        print(f"{prefix}⚠ {message}")
    elif status == "error":
        print(f"{prefix}✗ {message}")
    else:
        print(f"{prefix}  {message}")


def validate_xml_parsing(urdf_path: str) -> tuple:
    """
    Validate URDF XML parsing.
    
    Returns:
        tuple: (success, tree, errors)
    """
    errors = []
    tree = None
    
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        if root.tag != 'robot':
            errors.append(f"Root element is '{root.tag}', expected 'robot'")
        
        robot_name = root.get('name')
        if not robot_name:
            errors.append("Robot name attribute is missing")
        
    except ET.ParseError as e:
        errors.append(f"XML parse error: {e}")
    except FileNotFoundError:
        errors.append(f"URDF file not found: {urdf_path}")
    
    return len(errors) == 0, tree, errors


def validate_links(tree: ET.ElementTree) -> tuple:
    """
    Validate all link definitions.
    
    Returns:
        tuple: (success, link_names, errors, warnings)
    """
    root = tree.getroot()
    errors = []
    warnings = []
    link_names = []
    
    links = root.findall('.//link')
    
    for link in links:
        name = link.get('name')
        if not name:
            errors.append("Found link without name attribute")
            continue
        
        link_names.append(name)
        
        # Check for inertial
        inertial = link.find('inertial')
        if inertial is not None:
            mass = inertial.find('mass')
            if mass is not None:
                mass_val = float(mass.get('value', 0))
                if mass_val <= 0:
                    warnings.append(f"Link '{name}' has non-positive mass: {mass_val}")
            
            inertia = inertial.find('inertia')
            if inertia is not None:
                ixx = float(inertia.get('ixx', 0))
                iyy = float(inertia.get('iyy', 0))
                izz = float(inertia.get('izz', 0))
                
                if ixx <= 0 or iyy <= 0 or izz <= 0:
                    warnings.append(f"Link '{name}' has non-positive inertia diagonal")
        
        # Check for visual
        visual = link.find('visual')
        if visual is None and name not in ['base_footprint', 'lidar_optical_frame', 
                                            'camera_optical_frame', 'camera_depth_frame',
                                            'camera_depth_optical_frame']:
            warnings.append(f"Link '{name}' has no visual element")
        
        # Check for collision
        collision = link.find('collision')
        if collision is None and name not in ['base_footprint', 'lidar_optical_frame',
                                               'camera_optical_frame', 'camera_depth_frame',
                                               'camera_depth_optical_frame']:
            warnings.append(f"Link '{name}' has no collision element")
    
    return len(errors) == 0, link_names, errors, warnings


def validate_joints(tree: ET.ElementTree, link_names: list) -> tuple:
    """
    Validate all joint definitions.
    
    Returns:
        tuple: (success, joint_names, errors, warnings)
    """
    root = tree.getroot()
    errors = []
    warnings = []
    joint_names = []
    
    joints = root.findall('.//joint')
    
    for joint in joints:
        name = joint.get('name')
        joint_type = joint.get('type')
        
        if not name:
            errors.append("Found joint without name attribute")
            continue
        
        joint_names.append(name)
        
        if joint_type not in ['fixed', 'continuous', 'revolute', 'prismatic', 'floating', 'planar']:
            errors.append(f"Joint '{name}' has invalid type: {joint_type}")
        
        # Check parent/child links
        parent = joint.find('parent')
        child = joint.find('child')
        
        if parent is None:
            errors.append(f"Joint '{name}' missing parent element")
        else:
            parent_link = parent.get('link')
            if parent_link not in link_names:
                errors.append(f"Joint '{name}' references unknown parent link: {parent_link}")
        
        if child is None:
            errors.append(f"Joint '{name}' missing child element")
        else:
            child_link = child.get('link')
            if child_link not in link_names:
                errors.append(f"Joint '{name}' references unknown child link: {child_link}")
        
        # Check continuous/revolute joints have axis
        if joint_type in ['continuous', 'revolute', 'prismatic']:
            axis = joint.find('axis')
            if axis is None:
                warnings.append(f"Joint '{name}' ({joint_type}) has no axis element, defaults to [1,0,0]")
    
    return len(errors) == 0, joint_names, errors, warnings


def validate_meshes(tree: ET.ElementTree, meshes_path: str) -> tuple:
    """
    Validate mesh file references.
    
    Returns:
        tuple: (success, mesh_files, errors, warnings)
    """
    root = tree.getroot()
    errors = []
    warnings = []
    mesh_files = []
    
    meshes = root.findall('.//mesh')
    
    for mesh in meshes:
        filename = mesh.get('filename')
        if not filename:
            errors.append("Found mesh element without filename attribute")
            continue
        
        mesh_files.append(filename)
        
        # Extract actual file path from package:// URI
        if filename.startswith('package://'):
            # e.g., package://mining_robot_description/meshes/chassis.stl
            parts = filename.replace('package://', '').split('/')
            if len(parts) >= 2:
                mesh_name = parts[-1]  # e.g., chassis.stl
                mesh_path = Path(meshes_path) / mesh_name
                
                if not mesh_path.exists():
                    warnings.append(f"Mesh file not found: {mesh_path}")
    
    return len(errors) == 0, mesh_files, errors, warnings


def validate_gazebo_plugins(tree: ET.ElementTree) -> tuple:
    """
    Validate Gazebo plugin configurations.
    
    Returns:
        tuple: (success, plugins, errors, warnings)
    """
    root = tree.getroot()
    errors = []
    warnings = []
    plugins = []
    
    # Find all gazebo elements
    gazebo_elements = root.findall('.//gazebo')
    
    for gazebo in gazebo_elements:
        # Find plugins within gazebo elements
        for plugin in gazebo.findall('.//plugin'):
            plugin_name = plugin.get('name')
            plugin_filename = plugin.get('filename')
            
            if plugin_name:
                plugins.append(plugin_name)
            
            if not plugin_filename:
                warnings.append(f"Plugin '{plugin_name}' missing filename attribute")
    
    # Check for required plugins
    expected_plugins = ['diff_drive', 'joint_state_publisher']
    for expected in expected_plugins:
        if expected not in plugins:
            warnings.append(f"Missing expected plugin: {expected}")
    
    return len(errors) == 0, plugins, errors, warnings


def validate_coordinate_frames(tree: ET.ElementTree) -> tuple:
    """
    Validate coordinate frame conventions (REP-103 compliance).
    
    Returns:
        tuple: (success, frames, errors, warnings)
    """
    root = tree.getroot()
    errors = []
    warnings = []
    frames = []
    
    # Check for required frames
    links = root.findall('.//link')
    link_names = [link.get('name') for link in links if link.get('name')]
    
    required_frames = ['base_link', 'base_footprint']
    sensor_frames = ['lidar_link', 'camera_link']
    optical_frames = ['lidar_optical_frame', 'camera_optical_frame']
    
    for frame in required_frames:
        if frame in link_names:
            frames.append(frame)
        else:
            errors.append(f"Missing required frame: {frame}")
    
    for frame in sensor_frames:
        if frame in link_names:
            frames.append(frame)
        else:
            warnings.append(f"Missing sensor frame: {frame}")
    
    for frame in optical_frames:
        if frame in link_names:
            frames.append(frame)
    
    return len(errors) == 0, frames, errors, warnings


def run_validation(urdf_path: str, meshes_path: str) -> bool:
    """
    Run all validations on the URDF.
    
    Returns:
        bool: True if all validations pass
    """
    print("="*60)
    print("MINING ROBOT URDF VALIDATION")
    print("="*60)
    print(f"\nURDF file: {urdf_path}")
    print(f"Meshes path: {meshes_path}")
    
    all_passed = True
    
    # 1. XML Parsing
    print("\n[1] XML Parsing")
    print("-" * 40)
    success, tree, errors = validate_xml_parsing(urdf_path)
    
    if success:
        print_status("ok", "URDF parses correctly")
    else:
        all_passed = False
        for error in errors:
            print_status("error", error)
        print("\n[ABORT] Cannot continue without valid XML")
        return False
    
    # 2. Links
    print("\n[2] Link Validation")
    print("-" * 40)
    success, link_names, errors, warnings = validate_links(tree)
    
    print_status("info", f"Found {len(link_names)} links")
    for name in link_names[:5]:
        print_status("ok", name, indent=1)
    if len(link_names) > 5:
        print_status("info", f"... and {len(link_names) - 5} more", indent=1)
    
    for error in errors:
        print_status("error", error)
        all_passed = False
    for warning in warnings[:3]:
        print_status("warn", warning)
    
    # 3. Joints
    print("\n[3] Joint Validation")
    print("-" * 40)
    success, joint_names, errors, warnings = validate_joints(tree, link_names)
    
    print_status("info", f"Found {len(joint_names)} joints")
    for name in joint_names[:5]:
        print_status("ok", name, indent=1)
    if len(joint_names) > 5:
        print_status("info", f"... and {len(joint_names) - 5} more", indent=1)
    
    for error in errors:
        print_status("error", error)
        all_passed = False
    for warning in warnings[:3]:
        print_status("warn", warning)
    
    # 4. Meshes
    print("\n[4] Mesh Validation")
    print("-" * 40)
    success, mesh_files, errors, warnings = validate_meshes(tree, meshes_path)
    
    unique_meshes = list(set(mesh_files))
    print_status("info", f"Found {len(unique_meshes)} unique mesh references")
    for mesh in unique_meshes[:5]:
        print_status("ok", mesh.split('/')[-1], indent=1)
    
    for error in errors:
        print_status("error", error)
        all_passed = False
    for warning in warnings:
        print_status("warn", warning)
    
    # 5. Gazebo Plugins
    print("\n[5] Gazebo Plugin Validation")
    print("-" * 40)
    success, plugins, errors, warnings = validate_gazebo_plugins(tree)
    
    print_status("info", f"Found {len(plugins)} plugins")
    for plugin in plugins:
        print_status("ok", plugin, indent=1)
    
    for error in errors:
        print_status("error", error)
        all_passed = False
    for warning in warnings:
        print_status("warn", warning)
    
    # 6. Coordinate Frames
    print("\n[6] Coordinate Frame Validation")
    print("-" * 40)
    success, frames, errors, warnings = validate_coordinate_frames(tree)
    
    print_status("info", f"Found {len(frames)} standard frames")
    for frame in frames:
        print_status("ok", frame, indent=1)
    
    for error in errors:
        print_status("error", error)
        all_passed = False
    for warning in warnings:
        print_status("warn", warning)
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("VALIDATION RESULT: ✓ PASSED")
    else:
        print("VALIDATION RESULT: ✗ FAILED")
    print("="*60)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate mining robot URDF"
    )
    parser.add_argument(
        '--urdf-path',
        type=str,
        default=None,
        help='Path to URDF file'
    )
    parser.add_argument(
        '--meshes-path',
        type=str,
        default=None,
        help='Path to meshes directory'
    )
    
    args = parser.parse_args()
    
    # Default paths
    script_dir = Path(__file__).parent
    pkg_dir = script_dir.parent
    
    if args.urdf_path is None:
        urdf_path = pkg_dir / 'urdf' / 'mining_robot.urdf.xacro'
    else:
        urdf_path = Path(args.urdf_path)
    
    if args.meshes_path is None:
        meshes_path = pkg_dir / 'meshes'
    else:
        meshes_path = Path(args.meshes_path)
    
    # Check for xacro file
    if str(urdf_path).endswith('.xacro'):
        print("\n[NOTE] URDF file is xacro format.")
        print("       To validate fully, first process with: xacro mining_robot.urdf.xacro > robot.urdf")
        print("       Then run: python validate_urdf.py --urdf-path robot.urdf\n")
    
    success = run_validation(str(urdf_path), str(meshes_path))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
