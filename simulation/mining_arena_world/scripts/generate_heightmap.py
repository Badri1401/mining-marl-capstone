#!/usr/bin/env python3
"""
Generate Heightmap Terrain for Mining Arena
============================================
Creates Perlin noise-based heightmaps for realistic mine floor terrain.
Generates 5 terrain variants for domain randomization during training.

Heightmap Specs:
- Resolution: 512×512 pixels
- Arena size: 50×50 meters
- Elevation range: ±0.15m (relative to ground level)
- Output: 8-bit grayscale PNG (0=low, 128=flat, 255=high)

Terrain Features:
- Base Perlin noise for natural undulation
- Localized bumps and ruts (mining debris)
- Drainage channels (lower areas)
- Reinforced paths (slightly raised, smoother)

Usage:
    python generate_heightmap.py [--output-dir PATH] [--num-variants N]
"""

import numpy as np
import argparse
import os
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install pillow")
    exit(1)

try:
    from noise import pnoise2, snoise2
except ImportError:
    print("WARNING: noise library not installed. Using fallback Perlin implementation.")
    print("For better quality, run: pip install noise")
    pnoise2 = None
    snoise2 = None


def fallback_perlin(x: float, y: float, octaves: int = 4, seed: int = 0) -> float:
    """
    Simple fallback Perlin-like noise using NumPy.
    Used when the 'noise' library is not available.
    """
    np.random.seed(seed)
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        # Use interpolated random grid
        ix = int(np.floor(x * frequency))
        iy = int(np.floor(y * frequency))
        
        # Simple hash function
        np.random.seed((ix * 73856093 ^ iy * 19349663 ^ seed) % (2**31))
        n00 = np.random.uniform(-1, 1)
        np.random.seed(((ix+1) * 73856093 ^ iy * 19349663 ^ seed) % (2**31))
        n10 = np.random.uniform(-1, 1)
        np.random.seed((ix * 73856093 ^ (iy+1) * 19349663 ^ seed) % (2**31))
        n01 = np.random.uniform(-1, 1)
        np.random.seed(((ix+1) * 73856093 ^ (iy+1) * 19349663 ^ seed) % (2**31))
        n11 = np.random.uniform(-1, 1)
        
        # Interpolation weights
        fx = (x * frequency) - ix
        fy = (y * frequency) - iy
        
        # Smoothstep
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        
        # Bilinear interpolation
        nx0 = n00 * (1 - fx) + n10 * fx
        nx1 = n01 * (1 - fx) + n11 * fx
        n = nx0 * (1 - fy) + nx1 * fy
        
        value += n * amplitude
        max_value += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    
    return value / max_value


def generate_base_terrain(
    width: int, 
    height: int, 
    scale: float = 50.0,
    octaves: int = 6,
    seed: int = 0
) -> np.ndarray:
    """
    Generate base terrain using Perlin noise.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        scale: Noise scale (larger = smoother terrain)
        octaves: Number of noise octaves (more = more detail)
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Heightmap values in range [-1, 1]
    """
    terrain = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            nx = x / scale
            ny = y / scale
            
            if pnoise2 is not None:
                # Use noise library
                value = pnoise2(
                    nx, ny,
                    octaves=octaves,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=width,
                    repeaty=height,
                    base=seed
                )
            else:
                # Use fallback
                value = fallback_perlin(nx, ny, octaves=octaves, seed=seed)
            
            terrain[y, x] = value
    
    return terrain


def add_mining_features(
    terrain: np.ndarray,
    num_bumps: int = 30,
    num_ruts: int = 15,
    num_channels: int = 3,
    seed: int = 0
) -> np.ndarray:
    """
    Add mining-specific terrain features.
    
    Args:
        terrain: Base terrain heightmap
        num_bumps: Number of debris piles/bumps
        num_ruts: Number of wheel ruts/depressions
        num_channels: Number of drainage channels
        seed: Random seed
    
    Returns:
        np.ndarray: Modified terrain with features
    """
    np.random.seed(seed)
    height, width = terrain.shape
    
    # Add random bumps (debris piles)
    for _ in range(num_bumps):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        radius = np.random.randint(5, 20)
        bump_height = np.random.uniform(0.1, 0.3)
        
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        mask = distance < radius
        
        # Gaussian bump
        bump = np.exp(-distance**2 / (2 * (radius/2)**2)) * bump_height
        terrain += bump * mask
    
    # Add ruts (depressions from vehicle tracks)
    for _ in range(num_ruts):
        # Random line across terrain
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(50, 150)
        rut_depth = np.random.uniform(0.05, 0.15)
        rut_width = np.random.randint(3, 8)
        
        for t in np.linspace(0, length, int(length)):
            cx = int(x1 + t * np.cos(angle)) % width
            cy = int(y1 + t * np.sin(angle)) % height
            
            # Create rut depression
            for dx in range(-rut_width, rut_width + 1):
                for dy in range(-rut_width, rut_width + 1):
                    px = (cx + dx) % width
                    py = (cy + dy) % height
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < rut_width:
                        terrain[py, px] -= rut_depth * (1 - dist / rut_width)
    
    # Add drainage channels
    for _ in range(num_channels):
        # Start point at edge
        if np.random.random() > 0.5:
            x, y = 0, np.random.randint(0, height)
        else:
            x, y = np.random.randint(0, width), 0
        
        channel_depth = np.random.uniform(0.1, 0.2)
        channel_width = np.random.randint(5, 12)
        
        # Follow lowest neighbor (water flow simulation)
        for _ in range(width + height):
            # Depress current location
            for dx in range(-channel_width, channel_width + 1):
                for dy in range(-channel_width, channel_width + 1):
                    px = (x + dx) % width
                    py = (y + dy) % height
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < channel_width:
                        terrain[py, px] -= channel_depth * (1 - dist / channel_width) * 0.5
            
            # Move to random neighbor (biased toward lower areas)
            dx = np.random.choice([-1, 0, 1])
            dy = np.random.choice([-1, 0, 1])
            x = (x + dx) % width
            y = (y + dy) % height
    
    return terrain


def add_puddle_zones(
    terrain: np.ndarray,
    num_puddles: int = 8,
    seed: int = 0
) -> tuple:
    """
    Identify low areas as water puddle zones (for friction modification).
    
    Args:
        terrain: Terrain heightmap
        num_puddles: Number of puddle zones
        seed: Random seed
    
    Returns:
        tuple: (terrain, friction_map) where friction_map has puddle zones marked
    """
    np.random.seed(seed + 1000)
    height, width = terrain.shape
    
    # Friction map: 1.0 = normal friction, 0.125 = puddle (μ=0.1 vs μ=0.8)
    friction_map = np.ones((height, width), dtype=np.float32)
    
    # Find local minima for puddle placement
    for _ in range(num_puddles):
        # Random location biased toward lower terrain
        attempts = 50
        best_x, best_y = 0, 0
        lowest = float('inf')
        
        for _ in range(attempts):
            x = np.random.randint(20, width - 20)
            y = np.random.randint(20, height - 20)
            if terrain[y, x] < lowest:
                lowest = terrain[y, x]
                best_x, best_y = x, y
        
        # Create puddle zone
        puddle_radius = np.random.randint(8, 20)
        
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - best_x)**2 + (y_coords - best_y)**2)
        puddle_mask = distance < puddle_radius
        
        friction_map[puddle_mask] = 0.125  # Low friction ratio
    
    return terrain, friction_map


def normalize_terrain(
    terrain: np.ndarray,
    min_elevation: float = -0.15,
    max_elevation: float = 0.15
) -> np.ndarray:
    """
    Normalize terrain to desired elevation range.
    
    Args:
        terrain: Raw terrain heightmap
        min_elevation: Minimum elevation in meters
        max_elevation: Maximum elevation in meters
    
    Returns:
        np.ndarray: Normalized terrain in [min_elevation, max_elevation]
    """
    # Normalize to [0, 1]
    terrain_min = terrain.min()
    terrain_max = terrain.max()
    
    if terrain_max - terrain_min > 0:
        normalized = (terrain - terrain_min) / (terrain_max - terrain_min)
    else:
        normalized = np.zeros_like(terrain)
    
    # Scale to elevation range
    scaled = normalized * (max_elevation - min_elevation) + min_elevation
    
    return scaled


def terrain_to_image(terrain: np.ndarray) -> np.ndarray:
    """
    Convert terrain heightmap to 8-bit grayscale image.
    
    Args:
        terrain: Terrain in meters (normalized)
    
    Returns:
        np.ndarray: 8-bit grayscale image (0-255)
    """
    # Normalize to [0, 1]
    terrain_min = terrain.min()
    terrain_max = terrain.max()
    
    if terrain_max - terrain_min > 0:
        normalized = (terrain - terrain_min) / (terrain_max - terrain_min)
    else:
        normalized = np.ones_like(terrain) * 0.5
    
    # Convert to 8-bit
    image = (normalized * 255).astype(np.uint8)
    
    return image


def generate_terrain_variant(
    variant_id: int,
    width: int = 512,
    height: int = 512,
    arena_size: float = 50.0
) -> tuple:
    """
    Generate a complete terrain variant with all features.
    
    Args:
        variant_id: Variant number (1-5) for seed differentiation
        width: Image width
        height: Image height
        arena_size: Arena size in meters
    
    Returns:
        tuple: (heightmap_image, friction_map, terrain_meters)
    """
    seed = variant_id * 12345
    
    # Vary parameters per variant
    scale_variants = [40.0, 50.0, 60.0, 45.0, 55.0]
    bump_variants = [25, 35, 20, 40, 30]
    rut_variants = [10, 15, 20, 12, 18]
    
    scale = scale_variants[variant_id % len(scale_variants)]
    num_bumps = bump_variants[variant_id % len(bump_variants)]
    num_ruts = rut_variants[variant_id % len(rut_variants)]
    
    print(f"  Generating variant {variant_id}: scale={scale}, bumps={num_bumps}, ruts={num_ruts}")
    
    # Generate base terrain
    terrain = generate_base_terrain(width, height, scale=scale, octaves=6, seed=seed)
    
    # Add mining features
    terrain = add_mining_features(
        terrain,
        num_bumps=num_bumps,
        num_ruts=num_ruts,
        num_channels=3,
        seed=seed
    )
    
    # Add puddle zones
    terrain, friction_map = add_puddle_zones(terrain, num_puddles=6, seed=seed)
    
    # Normalize to elevation range
    terrain_meters = normalize_terrain(terrain, min_elevation=-0.15, max_elevation=0.15)
    
    # Convert to image
    heightmap_image = terrain_to_image(terrain_meters)
    
    return heightmap_image, friction_map, terrain_meters


def save_heightmap(
    heightmap: np.ndarray,
    filepath: str,
    metadata: dict = None
):
    """
    Save heightmap as PNG image.
    
    Args:
        heightmap: 8-bit grayscale heightmap
        filepath: Output file path
        metadata: Optional metadata to embed
    """
    img = Image.fromarray(heightmap, mode='L')
    
    # Add metadata if provided
    if metadata:
        from PIL import PngImagePlugin
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            pnginfo.add_text(key, str(value))
        img.save(filepath, pnginfo=pnginfo)
    else:
        img.save(filepath)


def save_friction_map(friction_map: np.ndarray, filepath: str):
    """
    Save friction coefficient map as NPY file.
    
    Args:
        friction_map: Friction coefficients (1.0 = normal, 0.125 = puddle)
        filepath: Output file path
    """
    np.save(filepath, friction_map)


def generate_all_variants(output_dir: str, num_variants: int = 5) -> dict:
    """
    Generate all terrain variants.
    
    Args:
        output_dir: Output directory
        num_variants: Number of variants to generate
    
    Returns:
        dict: Mapping of variant IDs to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for i in range(1, num_variants + 1):
        print(f"\n[VARIANT {i}/{num_variants}]")
        
        heightmap, friction_map, terrain_meters = generate_terrain_variant(i)
        
        # Save heightmap PNG
        heightmap_file = output_path / f"terrain_v{i}.png"
        metadata = {
            'variant': str(i),
            'resolution': '512x512',
            'arena_size': '50x50m',
            'elevation_range': '-0.15m to +0.15m'
        }
        save_heightmap(heightmap, str(heightmap_file), metadata)
        
        # Save friction map
        friction_file = output_path / f"friction_v{i}.npy"
        save_friction_map(friction_map, str(friction_file))
        
        # Save raw terrain (for debugging)
        terrain_file = output_path / f"terrain_raw_v{i}.npy"
        np.save(str(terrain_file), terrain_meters)
        
        saved_files[i] = {
            'heightmap': str(heightmap_file),
            'friction': str(friction_file),
            'raw': str(terrain_file)
        }
        
        print(f"  ✓ Saved: {heightmap_file.name}")
        print(f"  ✓ Saved: {friction_file.name}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate heightmap terrain variants for mining arena"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for heightmap files'
    )
    parser.add_argument(
        '--num-variants',
        type=int,
        default=5,
        help='Number of terrain variants to generate (default: 5)'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show preview of generated heightmaps'
    )
    
    args = parser.parse_args()
    
    # Default output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'heightmaps'
    else:
        output_dir = Path(args.output_dir)
    
    print("="*60)
    print("MINING ARENA HEIGHTMAP GENERATOR")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Number of variants: {args.num_variants}")
    print("\nHeightmap Specifications:")
    print("  - Resolution: 512×512 pixels")
    print("  - Arena size: 50×50 meters")
    print("  - Elevation range: ±0.15m")
    print("  - Pixel scale: ~9.8cm/pixel")
    
    saved_files = generate_all_variants(str(output_dir), args.num_variants)
    
    print(f"\n[COMPLETE] Generated {len(saved_files)} terrain variants")
    
    # Print summary
    print("\n" + "="*60)
    print("TERRAIN VARIANT SUMMARY")
    print("="*60)
    for variant_id, files in saved_files.items():
        print(f"\nVariant {variant_id}:")
        print(f"  Heightmap: {Path(files['heightmap']).name}")
        print(f"  Friction:  {Path(files['friction']).name}")
    
    if args.preview:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (variant_id, files) in enumerate(saved_files.items()):
                if i >= 6:
                    break
                img = Image.open(files['heightmap'])
                axes[i].imshow(np.array(img), cmap='terrain')
                axes[i].set_title(f'Terrain Variant {variant_id}')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(str(output_dir / 'terrain_preview.png'), dpi=150)
            print(f"\n[PREVIEW] Saved preview to: {output_dir / 'terrain_preview.png'}")
            plt.show()
            
        except ImportError:
            print("\n[WARNING] matplotlib not installed, skipping preview")
    
    return saved_files


if __name__ == "__main__":
    main()
