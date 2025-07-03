import numpy as np
import pickle
import sys

class FractalCompressor:
    """
    A simple fractal compression implementation for 2D tensors (e.g., weight matrices or images).
    """
    def __init__(self, range_size=4, domain_size=8, step=2):
        self.range_size = range_size  # size of range block (R x R)
        self.domain_size = domain_size  # size of domain block (D x D), D = 2R typically
        self.step = step  # search step for domain blocks
        self.transforms = []  # list of (range_pos, domain_pos, scale, offset)

    def compress(self, tensor: np.ndarray):
        h, w = tensor.shape
        R, D = self.range_size, self.domain_size
        
        # Precompute all domain blocks downsampled to range size
        domain_patches = []  # list of (downscaled_block, (i, j))
        for i in range(0, h - D + 1, self.step):
            for j in range(0, w - D + 1, self.step):
                block = tensor[i:i+D, j:j+D]
                # downsample by averaging each 2x2 into 1 pixel
                small = block.reshape(R, D//R, R, D//R).mean(axis=(1,3))
                domain_patches.append((small, (i, j)))
        
        self.transforms = []
        
        # For each non-overlapping range block, find best matching domain transform
        for i in range(0, h - R + 1, R):
            for j in range(0, w - R + 1, R):
                rng = tensor[i:i+R, j:j+R]
                best_error = float('inf')
                best_transform = None
                # search best domain match
                for small, (di, dj) in domain_patches:
                    # find optimal linear mapping: s*x + o ~ y
                    x = small.flatten()
                    y = rng.flatten()
                    var_x = np.var(x)
                    if var_x < 1e-5:
                        s = 0.0
                    else:
                        s = np.cov(x, y)[0,1] / var_x
                    o = y.mean() - s * x.mean()
                    
                    approx = (s * x + o).reshape(R, R)
                    err = np.mean((rng - approx)**2)
                    
                    if err < best_error:
                        best_error = err
                        best_transform = ((i, j), (di, dj), s, o)
                
                self.transforms.append(best_transform)

    def decompress(self, shape: tuple, iterations: int = 8) -> np.ndarray:
        """
        Reconstruct the tensor of given shape using the stored transforms.
        """
        h, w = shape
        R, D = self.range_size, self.domain_size
        img = np.zeros((h, w))
        
        for _ in range(iterations):
            new_img = img.copy()
            for (ri, rj), (di, dj), s, o in self.transforms:
                block = img[di:di+D, dj:dj+D]
                # downsample
                small = block.reshape(R, D//R, R, D//R).mean(axis=(1,3))
                new_img[ri:ri+R, rj:rj+R] = s * small + o
            img = new_img
        
        return img

def demo():
    """
    Demonstration of fractal compression on a 2D tensor.
    """
    # Create a more interesting test pattern - a simple geometric pattern
    size = 32
    tensor = np.zeros((size, size), dtype=np.float32)
    
    # Create a pattern with some self-similarity
    for i in range(size):
        for j in range(size):
            # Create a pattern with diagonal stripes and some noise
            tensor[i, j] = 0.5 * np.sin(0.3 * (i + j)) + 0.3 * np.cos(0.5 * i) + 0.2 * np.random.random()
    
    # Add some geometric shapes for more structure
    tensor[8:12, 8:12] = 1.0  # square
    tensor[20:24, 20:24] = 0.0  # another square
    
    print("Original Tensor (32x32 with geometric pattern):")
    print("Shape:", tensor.shape)
    print("Min:", np.min(tensor), "Max:", np.max(tensor))
    print("Sample (top-left 8x8):")
    print(np.round(tensor[:8, :8], 3))
    
    # Compression parameters
    range_size = 4    # 4x4 range blocks
    domain_size = 8   # 8x8 domain blocks
    step = 2          # step size for domain search
    iterations = 10   # decompression iterations
    
    # Step 1: Initialize compressor and compress
    compressor = FractalCompressor(range_size=range_size, domain_size=domain_size, step=step)
    print(f"\nCompressing with {range_size}x{range_size} range blocks and {domain_size}x{domain_size} domain blocks...")
    
    compressor.compress(tensor)
    
    print(f"Number of transforms stored: {len(compressor.transforms)}")
    
    # Show a few example transforms
    print("\nSample transforms (range_pos, domain_pos, scale, offset):")
    for i, transform in enumerate(compressor.transforms[:5]):
        (ri, rj), (di, dj), s, o = transform
        print(f"  Transform {i+1}: Range({ri},{rj}) <- Domain({di},{dj}) * {s:.3f} + {o:.3f}")
    
    # Step 2: Decompress
    print(f"\nDecompressing with {iterations} iterations...")
    reconstructed = compressor.decompress(tensor.shape, iterations=iterations)
    
    print("Reconstructed Tensor:")
    print("Shape:", reconstructed.shape)
    print("Min:", np.min(reconstructed), "Max:", np.max(reconstructed))
    print("Sample (top-left 8x8):")
    print(np.round(reconstructed[:8, :8], 3))
    
    # Step 3: Evaluate compression quality
    mse = np.mean((tensor - reconstructed) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    print(f"\nQuality Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    
    # Step 4: Estimate compression ratio
    # Original size in bits (assuming 32-bit floats)
    original_bits = tensor.size * 32
    
    # Compressed size: each transform has 4 integers (positions) + 2 floats (scale, offset)
    # Assuming 16 bits for positions, 32 bits for floats
    compressed_bits = len(compressor.transforms) * (4 * 16 + 2 * 32)
    
    # Add overhead for storing compression parameters
    overhead_bits = 3 * 32  # range_size, domain_size, step
    total_compressed_bits = compressed_bits + overhead_bits
    
    compression_ratio = original_bits / total_compressed_bits
    
    print(f"\nCompression Analysis:")
    print(f"Original size: {original_bits} bits ({original_bits/8:.0f} bytes)")
    print(f"Compressed size: {total_compressed_bits} bits ({total_compressed_bits/8:.0f} bytes)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Step 5: Test serialization (to demonstrate practical storage)
    print(f"\nTesting serialization...")
    
    # Serialize the compressor (transforms + parameters)
    serialized = pickle.dumps({
        'transforms': compressor.transforms,
        'range_size': compressor.range_size,
        'domain_size': compressor.domain_size,
        'step': compressor.step,
        'shape': tensor.shape
    })
    
    actual_compressed_bytes = len(serialized)
    actual_ratio = (tensor.size * tensor.itemsize) / actual_compressed_bytes
    
    print(f"Actual serialized size: {actual_compressed_bytes} bytes")
    print(f"Actual compression ratio: {actual_ratio:.2f}x")

if __name__ == "__main__":
    demo()