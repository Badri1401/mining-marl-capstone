#!/usr/bin/env python3
"""
Simple Heightmap Generator for Mining Arena
Generates heightmap PNG files using pure Python Perlin noise implementation
Creates terrain with realistic mining environment elevations.

Output: 512x512 PNG grayscale images
- Black (0) = -0.15m elevation
- Gray (128) = 0m elevation  
- White (255) = +0.15m elevation

Author: Mining Robot Team
"""

import numpy as np
from pathlib import Path
from PIL import Image


class PerlinNoise:
    """
    Pure Python Perlin Noise implementation
    """
    
    def __init__(self, seed: int = 0):
        """
        Initialize with a random seed for reproducibility
        """
        np.random.seed(seed)
        self.p = np.arange(256, dtype=int)
        np.random.shuffle(self.p)
        self.p = np.tile(self.p, 2)
    
    def _fade(self, t):
        """Fade function: 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, a, b, t):
        """Linear interpolation"""
        return a + t * (b - a)
    
    def _grad(self, hash_val, x, y):
        """
        Gradient function - simplified 2D version
        """
        h = hash_val & 3
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        else:
            return -x - y
    
    def noise2d(self, x: float, y: float) -> float:
        """
        Generate 2D Perlin noise value at position (x, y)
        Returns value in range [-1, 1]
        """
        # Grid cell coordinates
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        
        # Relative position within cell
        x -= np.floor(x)
        y -= np.floor(y)
        
        # Fade curves
        u = self._fade(x)
        v = self._fade(y)
        
        # Hash coordinates of corners
        A = self.p[X] + Y
        AA = self.p[A]
        AB = self.p[A + 1]
        B = self.p[X + 1] + Y
        BA = self.p[B]
        BB = self.p[B + 1]
        
        # Gradient contributions
        g1 = self._grad(self.p[AA], x, y)
        g2 = self._grad(self.p[BA], x - 1, y)
        g3 = self._grad(self.p[AB], x, y - 1)
        g4 = self._grad(self.p[BB], x - 1, y - 1)
        
        # Interpolate
        return self._lerp(
            self._lerp(g1, g2, u),
            self._lerp(g3, g4, u),
            v
        )
    
    def octave_noise(self, x: float, y: float, octaves: int = 4, 
                     persistence: float = 0.5) -> float:
        """
        Generate fractal/octave noise (sum of multiple frequencies)
        """
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            total += self.noise2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        
        return total / max_value


def generate_heightmap(size: int, seed: int, octaves: int = 4, 
                       scale: float = 5.0, persistence: float = 0.5) -> np.ndarray:
    """
    Generate a heightmap using Perlin noise
    
    Args:
        size: Image size (width and height)
        seed: Random seed for reproducibility
        octaves: Number of noise octaves (more = more detail)
        scale: Scale factor (larger = more zoomed in)
        persistence: How much each octave contributes
        
    Returns:
        uint8 numpy array of heightmap values [0-255]
    """
    noise = PerlinNoise(seed)
    heightmap = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        for x in range(size):
            # Normalize coordinates to 0-scale range
            nx = x / size * scale
            ny = y / size * scale
            
            # Get noise value
            value = noise.octave_noise(nx, ny, octaves, persistence)
            
            # Store in heightmap (value is in range [-1, 1])
            heightmap[y, x] = value
    
    # Normalize to [0, 255] range
    min_val = heightmap.min()
    max_val = heightmap.max()
    if max_val - min_val > 0:
        heightmap = (heightmap - min_val) / (max_val - min_val)
    else:
        heightmap = np.full_like(heightmap, 0.5)
    
    # Apply bias toward flat areas (mining floors)
    # This creates more gradual transitions
    heightmap = np.power(heightmap, 0.8)  # Slight bias toward higher values
    
    # Convert to uint8
    heightmap_uint8 = (heightmap * 255).astype(np.uint8)
    
    return heightmap_uint8


def add_mining_features(heightmap: np.ndarray, seed: int) -> np.ndarray:
    """
    Add mining-specific terrain features:
    - Tunnels (dark linear features)
    - Pits (circular depressions)
    - Ridges (elevated areas)
    """
    np.random.seed(seed + 100)  # Different seed for features
    
    size = heightmap.shape[0]
    heightmap_float = heightmap.astype(np.float32)
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:size, :size]
    
    # Add some pit depressions (circular)
    num_pits = np.random.randint(3, 8)
    for _ in range(num_pits):
        cx = np.random.randint(size // 4, 3 * size // 4)
        cy = np.random.randint(size // 4, 3 * size // 4)
        radius = np.random.randint(20, 60)
        depth = np.random.uniform(20, 50)
        
        # Distance from center
        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        
        # Smooth falloff
        mask = np.clip(1 - dist / radius, 0, 1)
        mask = mask ** 2  # Smooth edges
        
        heightmap_float -= mask * depth
    
    # Add some ridges (elevated linear features)
    num_ridges = np.random.randint(2, 5)
    for _ in range(num_ridges):
        # Random line through the terrain
        angle = np.random.uniform(0, np.pi)
        offset = np.random.uniform(-size/4, size/4)
        width = np.random.uniform(15, 30)
        height_add = np.random.uniform(10, 30)
        
        # Distance to line
        line_dist = np.abs(
            np.cos(angle) * (x_coords - size/2) + 
            np.sin(angle) * (y_coords - size/2) - offset
        )
        
        mask = np.clip(1 - line_dist / width, 0, 1)
        mask = mask ** 2
        
        heightmap_float += mask * height_add
    
    # Clip to valid range
    heightmap_float = np.clip(heightmap_float, 0, 255)
    
    return heightmap_float.astype(np.uint8)


def generate_terrain_variant(variant_id: int, output_path: Path, size: int = 512):
    """
    Generate a specific terrain variant
    
    Variants:
        0 - Gentle rolling (training start)
        1 - Moderate hills  
        2 - Deep tunnels
        3 - Rocky ridges
        4 - Complex mixed terrain
    """
    
    # Different parameters for each variant
    variant_params = {
        0: {"octaves": 3, "scale": 3.0, "persistence": 0.3, "name": "gentle_rolling"},
        1: {"octaves": 4, "scale": 4.0, "persistence": 0.5, "name": "moderate_hills"},
        2: {"octaves": 5, "scale": 6.0, "persistence": 0.6, "name": "deep_tunnels"},
        3: {"octaves": 4, "scale": 5.0, "persistence": 0.7, "name": "rocky_ridges"},
        4: {"octaves": 6, "scale": 8.0, "persistence": 0.5, "name": "complex_mixed"},
    }
    
    params = variant_params.get(variant_id, variant_params[0])
    
    print(f"  Generating variant {variant_id}: {params['name']}")
    print(f"    Parameters: octaves={params['octaves']}, scale={params['scale']}, persistence={params['persistence']}")
    
    # Generate base heightmap
    seed = 42 + variant_id * 1000  # Reproducible seeds
    heightmap = generate_heightmap(
        size=size,
        seed=seed,
        octaves=params["octaves"],
        scale=params["scale"],
        persistence=params["persistence"]
    )
    
    # Add mining-specific features
    heightmap = add_mining_features(heightmap, seed)
    
    # Save as PNG
    filename = f"terrain_{params['name']}.png"
    filepath = output_path / filename
    
    img = Image.fromarray(heightmap, mode='L')  # 'L' = grayscale
    img.save(filepath)
    
    print(f"    Saved: {filename} ({size}x{size} pixels)")
    
    # Print elevation statistics
    min_elev = (heightmap.min() / 255.0 - 0.5) * 0.30  # Map to -0.15 to +0.15
    max_elev = (heightmap.max() / 255.0 - 0.5) * 0.30
    mean_elev = (heightmap.mean() / 255.0 - 0.5) * 0.30
    
    print(f"    Elevation: min={min_elev:.3f}m, max={max_elev:.3f}m, mean={mean_elev:.3f}m")
    
    return filepath


def main():
    """Generate all terrain heightmaps"""
    print("=" * 60)
    print("Mining Arena Heightmap Generator")  
    print("=" * 60)
    print()
    
    # Get script directory and output directory
    script_dir = Path(__file__).parent.absolute()
    heightmap_dir = script_dir.parent / "heightmaps"
    
    # Create output directory
    heightmap_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {heightmap_dir}")
    print()
    
    # Generate all 5 terrain variants
    print("Generating terrain variants...")
    print("-" * 40)
    
    generated_files = []
    for variant_id in range(5):
        filepath = generate_terrain_variant(variant_id, heightmap_dir, size=512)
        generated_files.append(filepath)
        print()
    
    print("-" * 40)
    print()
    
    # Summary
    print("Generated heightmaps:")
    for filepath in generated_files:
        size_kb = filepath.stat().st_size / 1024
        print(f"  - {filepath.name}: {size_kb:.1f} KB")
    
    print()
    print("Heightmap specification:")
    print("  - Format: 8-bit grayscale PNG")
    print("  - Size: 512 x 512 pixels")
    print("  - Elevation range: -0.15m to +0.15m")
    print("  - Pixel value 0 = -0.15m (lowest)")
    print("  - Pixel value 128 = 0m (ground level)")
    print("  - Pixel value 255 = +0.15m (highest)")
    print()
    print("=" * 60)
    print("Heightmap generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
