import numpy as np
import time
import tracemalloc
from huffman_quantization import block_quantize, block_dequantize, build_huffman_tree, get_codes, huffman_encode, huffman_decode
from fractal_compression import FractalCompressor

def test_huffman_quantization():
    """Test Huffman quantization with three non-trivial cases"""
    print("=== HUFFMAN QUANTIZATION TESTS ===")
    
    # Test Case 1: Smooth gradient (compressible)
    x, y = np.meshgrid(np.linspace(0, 10, 32), np.linspace(0, 10, 32))
    smooth = np.sin(x) * np.cos(y)
    
    # Test Case 2: Random noise (less compressible)
    np.random.seed(42)
    noise = np.random.randn(32, 32)
    
    # Test Case 3: Sparse data (highly compressible)
    sparse = np.zeros((32, 32))
    sparse[::4, ::4] = np.random.randn(8, 8) * 3
    
    test_cases = [
        ("Smooth Gradient", smooth),
        ("Random Noise", noise), 
        ("Sparse Data", sparse)
    ]
    
    for name, tensor in test_cases:
        print(f"\n--- {name} ---")
        print(f"Input shape: {tensor.shape}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        # Quantize
        quant, params = block_quantize(tensor, block_size=4, num_bits=4)
        
        # Build Huffman tree and encode
        tree = build_huffman_tree(quant)
        codes = get_codes(tree)
        encoded = huffman_encode(quant, codes)
        
        # Decode and dequantize
        decoded_quant = huffman_decode(encoded, tree, quant.shape)
        reconstructed = block_dequantize(decoded_quant, params, block_size=4)
        
        # Verify reconstruction
        mse = np.mean((tensor - reconstructed)**2)
        print(f"Expected: Perfect reconstruction within quantization error")
        print(f"Actual MSE: {mse:.6f}")
        print(f"Max absolute error: {np.max(np.abs(tensor - reconstructed)):.6f}")
        
        # Compression metrics
        original_bits = tensor.size * 32  # float32
        huffman_bits = len(encoded)
        compression_ratio = original_bits / huffman_bits
        print(f"Compression: {original_bits} → {huffman_bits} bits ({compression_ratio:.2f}x)")

def test_fractal_compression():
    """Test fractal compression with three non-trivial cases"""
    print("\n=== FRACTAL COMPRESSION TESTS ===")
    
    # Test Case 1: Self-similar pattern
    size = 32
    pattern = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            pattern[i, j] = np.sin(0.2 * (i + j)) + 0.5 * np.cos(0.3 * i)
    
    # Test Case 2: Geometric shapes
    geometric = np.zeros((32, 32))
    geometric[4:12, 4:12] = 1.0  # square
    geometric[20:28, 20:28] = 0.5  # another square
    geometric[10:22, 10:22] += 0.3  # overlapping
    
    # Test Case 3: Mixed content
    mixed = np.random.randn(32, 32) * 0.1
    mixed[8:16, 8:16] = np.sin(np.linspace(0, 4*np.pi, 64)).reshape(8, 8)
    
    test_cases = [
        ("Self-similar Pattern", pattern),
        ("Geometric Shapes", geometric),
        ("Mixed Content", mixed)
    ]
    
    for name, tensor in test_cases:
        print(f"\n--- {name} ---")
        print(f"Input shape: {tensor.shape}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        compressor = FractalCompressor(range_size=4, domain_size=8, step=2)
        compressor.compress(tensor)
        reconstructed = compressor.decompress(tensor.shape, iterations=8)
        
        mse = np.mean((tensor - reconstructed)**2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        print(f"Expected: Approximate reconstruction with some loss")
        print(f"Actual MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
        print(f"Transforms stored: {len(compressor.transforms)}")
        
        # Compression estimate
        original_bits = tensor.size * 32
        compressed_bits = len(compressor.transforms) * (4 * 16 + 2 * 32)  # positions + scale/offset
        compression_ratio = original_bits / compressed_bits
        print(f"Compression: {original_bits} → {compressed_bits} bits ({compression_ratio:.2f}x)")

def benchmark_performance():
    """Benchmark runtime and memory usage"""
    print("\n=== PERFORMANCE BENCHMARKS ===")
    
    sizes = [64, 128, 256]
    
    for size in sizes:
        print(f"\n--- Testing {size}x{size} tensor ---")
        np.random.seed(42)
        tensor = np.random.randn(size, size)
        
        # Huffman Quantization Benchmark
        tracemalloc.start()
        start_time = time.time()
        
        quant, params = block_quantize(tensor, block_size=8, num_bits=4)
        tree = build_huffman_tree(quant)
        codes = get_codes(tree)
        encoded = huffman_encode(quant, codes)
        
        huffman_time = time.time() - start_time
        current, peak_huffman = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Huffman: {huffman_time:.3f}s, Peak memory: {peak_huffman/1024/1024:.1f} MB")
        
        # Fractal Compression Benchmark
        tracemalloc.start()
        start_time = time.time()
        
        compressor = FractalCompressor(range_size=4, domain_size=8, step=2)
        compressor.compress(tensor)
        reconstructed = compressor.decompress(tensor.shape, iterations=5)
        
        fractal_time = time.time() - start_time
        current, peak_fractal = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Fractal: {fractal_time:.3f}s, Peak memory: {peak_fractal/1024/1024:.1f} MB")

if __name__ == "__main__":
    test_huffman_quantization()
    test_fractal_compression()
    benchmark_performance()
    print("\n=== SUMMARY ===")
    print("• Huffman quantization: Lossless compression, good for low-entropy data")
    print("• Fractal compression: Lossy compression, exploits self-similarity")
    print("• Both scale differently with data size and content characteristics")