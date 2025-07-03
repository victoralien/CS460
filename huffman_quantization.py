import numpy as np
from collections import Counter
import heapq

# Block-based uniform quantization
def block_quantize(tensor: np.ndarray, block_size: int, num_bits: int):
    h, w = tensor.shape
    quant = np.zeros_like(tensor, dtype=np.int32)
    params = {}
    levels = 2 ** num_bits
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = tensor[i:i+block_size, j:j+block_size]
            mn, mx = block.min(), block.max()
            step = (mx - mn) / (levels - 1) if mx != mn else 1.0
            codes = np.round((block - mn) / step).astype(np.int32)
            quant[i:i+block_size, j:j+block_size] = codes
            params[(i, j)] = (mn, step)
    return quant, params

# Dequantization
def block_dequantize(quant: np.ndarray, params: dict, block_size: int):
    h, w = quant.shape
    recon = np.zeros((h, w), dtype=np.float64)
    for (i, j), (mn, step) in params.items():
        codes = quant[i:i+block_size, j:j+block_size]
        recon[i:i+block_size, j:j+block_size] = codes * step + mn
    return recon

# Huffman coding
class Node:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data: np.ndarray):
    freq = Counter(data.flatten())
    heap = [Node(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return heap[0]
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent = Node(None, a.freq + b.freq)
        parent.left, parent.right = a, b
        heapq.heappush(heap, parent)
    return heap[0]


def get_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        get_codes(node.left, prefix + "0", codebook)
        get_codes(node.right, prefix + "1", codebook)
    return codebook


def huffman_encode(data: np.ndarray, codes: dict):
    flat = data.flatten()
    return "".join(codes[s] for s in flat)


def huffman_decode(bits: str, root: Node, shape: tuple):
    decoded = []
    node = root
    for b in bits:
        node = node.left if b == '0' else node.right
        if node.symbol is not None:
            decoded.append(node.symbol)
            node = root
    return np.array(decoded).reshape(shape)
