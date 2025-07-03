import numpy as np

from huffman_quantization import (
    block_quantize,
    block_dequantize,
    build_huffman_tree,
    get_codes,
    huffman_encode,
    huffman_decode
)


def demo():
    """
    Demonstration of block-based quantization followed by Huffman coding.
    """
    # Create a small example tensor (8x8) with a gradient
    tensor = np.linspace(0, 1, 64, dtype=np.float32).reshape((8, 8))
    print("Original Tensor (8x8 gradient):")
    print(tensor)

    # Quantization parameters
    block_size = 4   # 4x4 blocks
    num_bits = 3     # 3-bit uniform quantization per block

    # Step 1: Quantize
    quantized, params = block_quantize(tensor, block_size, num_bits)
    print("\nQuantized Codes:")
    print(quantized)

    # Step 2: Build Huffman tree & codes
    flat_codes = quantized.flatten()
    tree = build_huffman_tree(flat_codes)
    code_map = get_codes(tree)
    print("\nHuffman Codes for each symbol:")
    for symbol, code in sorted(code_map.items()):
        print(f"  {symbol}: {code}")

    # Step 3: Encode with Huffman
    encoded_bits = huffman_encode(flat_codes, code_map)
    bit_length = len(encoded_bits)
    print(f"\nTotal bits after Huffman encoding: {bit_length}")

    # Step 4: Decode back to quantized symbols
    decoded_flat = huffman_decode(encoded_bits, tree, shape=quantized.shape)
    decoded_quant = decoded_flat.reshape(quantized.shape)

    # Step 5: Dequantize
    recon = block_dequantize(decoded_quant, params, block_size)
    print("\nReconstructed Tensor after Dequantization:")
    print(np.round(recon, 3))

    # Step 6: Evaluate
    mse = np.mean((tensor - recon) ** 2)
    original_size = tensor.size * tensor.itemsize * 8  # bits
    compressed_size = bit_length + len(params) * (32 + 32)  # bits for params (min, step)
    ratio = original_size / compressed_size
    print(f"\nMSE: {mse:.6f}")
    print(f"Approx. Compression Ratio: {ratio:.2f}x")


if __name__ == "__main__":
    demo()
