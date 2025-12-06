"""
Image compression techniques implementation.
"""

import numpy as np
from collections import Counter, defaultdict
import heapq
from utils.helpers import ensure_grayscale
from utils.validators import validate_image_array


class CompressionEngine:
    """Handle various image compression techniques."""
    
    # ============================================================================
    # HUFFMAN CODING
    # ============================================================================
    
    class HuffmanNode:
        """Node for Huffman tree."""
        def __init__(self, symbol=None, frequency=0, left=None, right=None):
            self.symbol = symbol
            self.frequency = frequency
            self.left = left
            self.right = right
        
        def __lt__(self, other):
            return self.frequency < other.frequency
    
    @staticmethod
    def huffman_encode(image):
        """
        Apply Huffman coding compression.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image)
        flat_image = gray_image.flatten()
        
        # Count symbol frequencies
        frequency = Counter(flat_image)
        
        # Build Huffman tree
        heap = []
        for symbol, freq in frequency.items():
            node = CompressionEngine.HuffmanNode(symbol=symbol, frequency=freq)
            heapq.heappush(heap, node)
        
        # Build tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = CompressionEngine.HuffmanNode(
                frequency=left.frequency + right.frequency,
                left=left,
                right=right
            )
            heapq.heappush(heap, merged)
        
        # Generate codes
        root = heap[0]
        huffman_codes = {}
        
        def generate_codes(node, code=""):
            if node is None:
                return
            if node.symbol is not None:
                huffman_codes[node.symbol] = code if code else "0"
                return
            generate_codes(node.left, code + "0")
            generate_codes(node.right, code + "1")
        
        generate_codes(root)
        
        # Encode image
        encoded_data = []
        for pixel in flat_image:
            encoded_data.append(huffman_codes[pixel])
        
        # Calculate sizes
        original_bits = len(flat_image) * 8  # 8 bits per pixel
        compressed_bits = sum(len(code) for code in encoded_data)
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'method': 'Huffman Coding',
            'huffman_codes': huffman_codes,
            'encoded_data': encoded_data,
            'original_size_bits': original_bits,
            'compressed_size_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    # ============================================================================
    # GOLOMB-RICE CODING
    # ============================================================================
    
    @staticmethod
    def golomb_rice_encode(image, m=4):
        """
        Apply Golomb-Rice coding.
        
        Args:
            image: Input image
            m: Rice parameter (power of 2)
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image)
        flat_image = gray_image.flatten()
        
        # Calculate differences (predictive coding)
        differences = [int(flat_image[0])]
        for i in range(1, len(flat_image)):
            diff = int(flat_image[i]) - int(flat_image[i-1])
            differences.append(diff)
        
        # Encode using Golomb-Rice
        k = int(np.log2(m))
        encoded_bits = 0
        
        for diff in differences:
            # Map negative to positive
            mapped = 2 * abs(diff) if diff >= 0 else 2 * abs(diff) - 1
            
            # Quotient and remainder
            q = mapped // m
            r = mapped % m
            
            # Unary code for quotient + binary for remainder
            encoded_bits += (q + 1) + k
        
        original_bits = len(flat_image) * 8
        compression_ratio = original_bits / encoded_bits if encoded_bits > 0 else 0
        
        return {
            'method': 'Golomb-Rice Coding',
            'parameter_m': m,
            'differences': differences,
            'original_size_bits': original_bits,
            'compressed_size_bits': encoded_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    # ============================================================================
    # ARITHMETIC CODING
    # ============================================================================
    
    @staticmethod
    def arithmetic_encode(image):
        """
        Apply Arithmetic coding (simplified version).
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image)
        flat_image = gray_image.flatten()
        
        # Calculate probabilities
        frequency = Counter(flat_image)
        total_pixels = len(flat_image)
        probabilities = {symbol: freq / total_pixels for symbol, freq in frequency.items()}
        
        # Create cumulative probability ranges
        cumulative = {}
        cumsum = 0.0
        for symbol in sorted(probabilities.keys()):
            cumulative[symbol] = (cumsum, cumsum + probabilities[symbol])
            cumsum += probabilities[symbol]
        
        # Encode
        low = 0.0
        high = 1.0
        
        for pixel in flat_image:
            range_size = high - low
            symbol_low, symbol_high = cumulative[pixel]
            high = low + range_size * symbol_high
            low = low + range_size * symbol_low
        
        # Final value
        encoded_value = (low + high) / 2
        
        # Estimate bits needed
        compressed_bits = int(-np.log2(high - low)) + 1
        original_bits = len(flat_image) * 8
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'method': 'Arithmetic Coding',
            'encoded_value': encoded_value,
            'probabilities': probabilities,
            'original_size_bits': original_bits,
            'compressed_size_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    # ============================================================================
    # LZW CODING
    # ============================================================================
    
    @staticmethod
    def lzw_encode(image):
        """
        Apply LZW (Lempel-Ziv-Welch) coding.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image)
        flat_image = gray_image.flatten().tolist()
        
        # Initialize dictionary with single-pixel entries
        dictionary = {(i,): i for i in range(256)}
        dict_size = 256
        
        # Encode
        encoded = []
        current = ()
        
        for pixel in flat_image:
            current_plus = current + (pixel,)
            if current_plus in dictionary:
                current = current_plus
            else:
                encoded.append(dictionary[current])
                dictionary[current_plus] = dict_size
                dict_size += 1
                current = (pixel,)
        
        if current:
            encoded.append(dictionary[current])
        
        # Calculate sizes
        original_bits = len(flat_image) * 8
        bits_per_code = int(np.ceil(np.log2(dict_size)))
        compressed_bits = len(encoded) * bits_per_code
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'method': 'LZW Coding',
            'encoded_data': encoded,
            'dictionary_size': dict_size,
            'original_size_bits': original_bits,
            'compressed_size_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    # ============================================================================
    # RUN-LENGTH ENCODING (RLE)
    # ============================================================================
    
    @staticmethod
    def rle_encode(image):
        """
        Apply Run-Length Encoding.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image)
        flat_image = gray_image.flatten()
        
        # Encode runs
        encoded = []
        if len(flat_image) == 0:
            return {
                'method': 'Run-Length Encoding',
                'encoded_data': [],
                'original_size_bits': 0,
                'compressed_size_bits': 0,
                'compression_ratio': 0,
                'space_saving_percent': 0
            }
        
        current_value = flat_image[0]
        count = 1
        
        for i in range(1, len(flat_image)):
            if flat_image[i] == current_value and count < 255:
                count += 1
            else:
                encoded.append((current_value, count))
                current_value = flat_image[i]
                count = 1
        
        encoded.append((current_value, count))
        
        # Calculate sizes
        original_bits = len(flat_image) * 8
        compressed_bits = len(encoded) * (8 + 8)  # value + count
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'method': 'Run-Length Encoding',
            'encoded_data': encoded,
            'run_count': len(encoded),
            'original_size_bits': original_bits,
            'compressed_size_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    # ============================================================================
    # SYMBOL-BASED CODING
    # ============================================================================
    
    @staticmethod
    def symbol_based_encode(image):
        """
        Apply Symbol-based coding (dictionary-based).
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image)
        flat_image = gray_image.flatten()
        
        # Find unique symbols and create dictionary
        unique_symbols = np.unique(flat_image)
        symbol_dict = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
        
        # Encode
        encoded = [symbol_dict[pixel] for pixel in flat_image]
        
        # Calculate sizes
        original_bits = len(flat_image) * 8
        bits_per_symbol = int(np.ceil(np.log2(len(unique_symbols))))
        compressed_bits = len(encoded) * bits_per_symbol + len(unique_symbols) * 8
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'method': 'Symbol-Based Coding',
            'encoded_data': encoded,
            'symbol_dictionary': symbol_dict,
            'unique_symbols': len(unique_symbols),
            'original_size_bits': original_bits,
            'compressed_size_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    # ============================================================================
    # BIT-PLANE CODING
    # ============================================================================
    
    @staticmethod
    def bitplane_encode(image):
        """
        Apply Bit-plane coding.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image)
        
        # Extract bit planes
        bit_planes = []
        for bit in range(8):
            plane = (gray_image >> bit) & 1
            bit_planes.append(plane)
        
        # Compress each bit plane using RLE
        total_compressed_bits = 0
        compressed_planes = []
        
        for plane in bit_planes:
            flat_plane = plane.flatten()
            rle_result = CompressionEngine.rle_encode(plane)
            compressed_planes.append(rle_result)
            total_compressed_bits += rle_result['compressed_size_bits']
        
        original_bits = gray_image.size * 8
        compression_ratio = original_bits / total_compressed_bits if total_compressed_bits > 0 else 0
        
        return {
            'method': 'Bit-Plane Coding',
            'bit_planes': bit_planes,
            'compressed_planes': compressed_planes,
            'original_size_bits': original_bits,
            'compressed_size_bits': total_compressed_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    # ============================================================================
    # BLOCK TRANSFORM CODING (DCT)
    # ============================================================================
    
    @staticmethod
    def dct_encode(image, block_size=8, quality=50):
        """
        Apply DCT (Discrete Cosine Transform) block coding.
        
        Args:
            image: Input image
            block_size: Size of blocks (default 8x8)
            quality: Quality factor (1-100)
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image).astype(np.float64)
        height, width = gray_image.shape
        
        # Quantization matrix (JPEG-like)
        quantization_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float64)
        
        # Adjust quantization based on quality
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        quantization_matrix = np.floor((quantization_matrix * scale + 50) / 100)
        quantization_matrix[quantization_matrix < 1] = 1
        
        # Process blocks
        dct_blocks = []
        quantized_blocks = []
        non_zero_coeffs = 0
        total_coeffs = 0
        
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Extract block
                block = gray_image[i:min(i+block_size, height), 
                                  j:min(j+block_size, width)]
                
                # Pad if necessary
                if block.shape[0] < block_size or block.shape[1] < block_size:
                    padded_block = np.zeros((block_size, block_size))
                    padded_block[:block.shape[0], :block.shape[1]] = block
                    block = padded_block
                
                # Apply DCT
                dct_block = CompressionEngine._dct2d(block - 128)
                dct_blocks.append(dct_block)
                
                # Quantize
                quantized = np.round(dct_block / quantization_matrix)
                quantized_blocks.append(quantized)
                
                # Count non-zero coefficients
                non_zero_coeffs += np.count_nonzero(quantized)
                total_coeffs += quantized.size
        
        # Calculate compression metrics
        original_bits = gray_image.size * 8
        # Estimate compressed size based on non-zero coefficients
        compressed_bits = non_zero_coeffs * 8  # Simplified estimation
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'method': 'DCT Block Transform',
            'block_size': block_size,
            'quality': quality,
            'dct_blocks': dct_blocks,
            'quantized_blocks': quantized_blocks,
            'non_zero_coefficients': non_zero_coeffs,
            'total_coefficients': total_coeffs,
            'sparsity': (1 - non_zero_coeffs / total_coeffs) * 100,
            'original_size_bits': original_bits,
            'compressed_size_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    @staticmethod
    def _dct2d(block):
        """
        Compute 2D DCT of a block.
        
        Args:
            block: Input block
            
        Returns:
            DCT coefficients
        """
        N = block.shape[0]
        dct_result = np.zeros_like(block, dtype=np.float64)
        
        for u in range(N):
            for v in range(N):
                sum_val = 0.0
                for x in range(N):
                    for y in range(N):
                        sum_val += block[x, y] * \
                                  np.cos((2 * x + 1) * u * np.pi / (2 * N)) * \
                                  np.cos((2 * y + 1) * v * np.pi / (2 * N))
                
                # Apply normalization
                cu = 1 / np.sqrt(2) if u == 0 else 1
                cv = 1 / np.sqrt(2) if v == 0 else 1
                dct_result[u, v] = (2 / N) * cu * cv * sum_val
        
        return dct_result
    
    @staticmethod
    def _idct2d(dct_block):
        """
        Compute 2D inverse DCT.
        
        Args:
            dct_block: DCT coefficients
            
        Returns:
            Reconstructed block
        """
        N = dct_block.shape[0]
        block = np.zeros_like(dct_block, dtype=np.float64)
        
        for x in range(N):
            for y in range(N):
                sum_val = 0.0
                for u in range(N):
                    for v in range(N):
                        cu = 1 / np.sqrt(2) if u == 0 else 1
                        cv = 1 / np.sqrt(2) if v == 0 else 1
                        sum_val += cu * cv * dct_block[u, v] * \
                                  np.cos((2 * x + 1) * u * np.pi / (2 * N)) * \
                                  np.cos((2 * y + 1) * v * np.pi / (2 * N))
                
                block[x, y] = (2 / N) * sum_val
        
        return block
    
    # ============================================================================
    # PREDICTIVE CODING
    # ============================================================================
    
    @staticmethod
    def predictive_encode(image):
        """
        Apply Predictive coding (DPCM - Differential Pulse Code Modulation).
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image)
        height, width = gray_image.shape
        
        # Create difference image
        differences = np.zeros_like(gray_image, dtype=np.int16)
        
        # First pixel is unchanged
        differences[0, 0] = gray_image[0, 0]
        
        # First row (predict from left)
        for j in range(1, width):
            differences[0, j] = int(gray_image[0, j]) - int(gray_image[0, j-1])
        
        # Remaining rows (predict from top and left)
        for i in range(1, height):
            # First column (predict from top)
            differences[i, 0] = int(gray_image[i, 0]) - int(gray_image[i-1, 0])
            
            # Other pixels (average of left and top)
            for j in range(1, width):
                predicted = (int(gray_image[i-1, j]) + int(gray_image[i, j-1])) // 2
                differences[i, j] = int(gray_image[i, j]) - predicted
        
        # Encode differences
        flat_diff = differences.flatten()
        
        # Calculate histogram of differences
        diff_range = np.max(flat_diff) - np.min(flat_diff)
        
        # Use Huffman coding on differences
        huffman_result = CompressionEngine.huffman_encode(
            (differences - np.min(differences)).astype(np.uint8)
        )
        
        original_bits = gray_image.size * 8
        compressed_bits = huffman_result['compressed_size_bits']
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'method': 'Predictive Coding (DPCM)',
            'differences': differences,
            'diff_range': diff_range,
            'huffman_result': huffman_result,
            'original_size_bits': original_bits,
            'compressed_size_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    # ============================================================================
    # WAVELET CODING
    # ============================================================================
    
    @staticmethod
    def wavelet_encode(image, levels=3):
        """
        Apply Wavelet coding (simplified Haar wavelet).
        
        Args:
            image: Input image
            levels: Number of decomposition levels
            
        Returns:
            Dictionary with compression results
        """
        gray_image = ensure_grayscale(image).astype(np.float64)
        
        # Apply Haar wavelet transform
        coefficients = CompressionEngine._haar_wavelet_2d(gray_image, levels)
        
        # Threshold small coefficients (compression)
        threshold = np.std(coefficients) * 0.5
        compressed_coeffs = coefficients.copy()
        compressed_coeffs[np.abs(compressed_coeffs) < threshold] = 0
        
        # Count non-zero coefficients
        non_zero = np.count_nonzero(compressed_coeffs)
        total = compressed_coeffs.size
        
        # Calculate sizes
        original_bits = gray_image.size * 8
        compressed_bits = non_zero * 16  # Assuming 16 bits per coefficient
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'method': 'Wavelet Coding (Haar)',
            'levels': levels,
            'coefficients': coefficients,
            'compressed_coefficients': compressed_coeffs,
            'threshold': threshold,
            'non_zero_coefficients': non_zero,
            'total_coefficients': total,
            'sparsity': (1 - non_zero / total) * 100,
            'original_size_bits': original_bits,
            'compressed_size_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
        }
    
    @staticmethod
    def _haar_wavelet_2d(image, levels):
        """
        Apply 2D Haar wavelet transform.
        
        Args:
            image: Input image
            levels: Number of decomposition levels
            
        Returns:
            Wavelet coefficients
        """
        result = image.copy()
        height, width = result.shape
        
        for level in range(levels):
            # Current size to process
            h = height // (2 ** level)
            w = width // (2 ** level)
            
            if h < 2 or w < 2:
                break
            
            # Process rows
            temp = np.zeros((h, w))
            for i in range(h):
                temp[i, :] = CompressionEngine._haar_1d(result[i, :w])
            
            # Process columns
            for j in range(w):
                temp[:, j] = CompressionEngine._haar_1d(temp[:, j])
            
            result[:h, :w] = temp
        
        return result
    
    @staticmethod
    def _haar_1d(signal):
        """
        Apply 1D Haar wavelet transform.
        
        Args:
            signal: 1D signal
            
        Returns:
            Transformed signal
        """
        n = len(signal)
        output = np.zeros(n)
        
        # Approximation coefficients (averages)
        for i in range(n // 2):
            output[i] = (signal[2*i] + signal[2*i + 1]) / np.sqrt(2)
        
        # Detail coefficients (differences)
        for i in range(n // 2):
            output[n // 2 + i] = (signal[2*i] - signal[2*i + 1]) / np.sqrt(2)
        
        return output
    
    # ============================================================================
    # COMPARISON METHOD
    # ============================================================================
    
    @staticmethod
    def compare_all_methods(image):
        """
        Compare all compression methods on the given image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with results from all methods
        """
        results = {}
        
        print("Running Huffman Coding...")
        results['huffman'] = CompressionEngine.huffman_encode(image)
        
        print("Running Golomb-Rice Coding...")
        results['golomb_rice'] = CompressionEngine.golomb_rice_encode(image)
        
        print("Running Arithmetic Coding...")
        results['arithmetic'] = CompressionEngine.arithmetic_encode(image)
        
        print("Running LZW Coding...")
        results['lzw'] = CompressionEngine.lzw_encode(image)
        
        print("Running Run-Length Encoding...")
        results['rle'] = CompressionEngine.rle_encode(image)
        
        print("Running Symbol-Based Coding...")
        results['symbol'] = CompressionEngine.symbol_based_encode(image)
        
        print("Running Bit-Plane Coding...")
        results['bitplane'] = CompressionEngine.bitplane_encode(image)
        
        print("Running DCT Block Transform...")
        results['dct'] = CompressionEngine.dct_encode(image)
        
        print("Running Predictive Coding...")
        results['predictive'] = CompressionEngine.predictive_encode(image)
        
        print("Running Wavelet Coding...")
        results['wavelet'] = CompressionEngine.wavelet_encode(image)
        
        return results