import numpy as np
import time
from fractal_compression import FractalCompressor

def demo():
    """
    Demonstration of fractal compression on a 2D tensor.
    """
    # Create a sample tensor (16x16) with gradient + checkerboard pattern
    size = 16
    x = np.linspace(0, 1, size)
    tensor = np.outer(x, x)
    tensor += (np.indices((size, size)).sum(axis=0) % 2) * 0.2

    print("Original Tensor:")
    print(np.round(tensor, 3))

    # Initialize compressor
    compressor = FractalCompressor(range_size=4, domain_size=8, step=2)

    # Compress
    start_t = time.time()
    transforms = compressor.compress(tensor)
    enc_time = time.time() - start_t
    print(f"\nCompression Time: {enc_time:.4f}s")
    print(f"Number of transforms: {len(transforms)}")

    # Decompress
    start_t = time.time()
    reconstructed = compressor.decompress(transforms, tensor.shape, iterations=10)
    dec_time = time.time() - start_t
    print(f"Decompression Time: {dec_time:.4f}s")

    # Compute MSE
    mse = np.mean((tensor - reconstructed) ** 2)
    print(f"MSE: {mse:.6f}")

    print("\nReconstructed Tensor:")
    print(np.round(reconstructed, 3))

if __name__ == "__main__":
    demo()
