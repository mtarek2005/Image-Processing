"""
Custom mathematical operations for image processing.
Implements basic operations without relying on high-level built-in functions.
"""

import numpy as np


def custom_mean(array):
    """Calculate mean of array using basic operations."""
    if array.size == 0:
        return 0
    total = np.sum(array)
    count = array.size
    return total / count


def custom_median(array):
    """Calculate median using sorting."""
    flat = array.flatten()
    sorted_array = np.sort(flat)
    n = len(sorted_array)
    
    if n % 2 == 0:
        return (sorted_array[n//2 - 1] + sorted_array[n//2]) / 2
    else:
        return sorted_array[n//2]


def custom_std(array):
    """Calculate standard deviation."""
    mean = custom_mean(array)
    squared_diff = (array - mean) ** 2
    variance = custom_mean(squared_diff)
    return np.sqrt(variance)


def custom_min(array):
    """Find minimum value in array."""
    return np.min(array)


def custom_max(array):
    """Find maximum value in array."""
    return np.max(array)


def custom_convolution(image, kernel):
    """
    Perform 2D convolution without using built-in convolution functions.
    
    Args:
        image: 2D numpy array
        kernel: 2D numpy array (filter kernel)
    
    Returns:
        Convolved image
    """
    # Get dimensions
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    # Pad the image
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    # Initialize output
    output = np.zeros_like(image, dtype=np.float64)
    
    # Perform convolution
    for i in range(img_height):
        for j in range(img_width):
            # Extract region
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
    
    return output


def matrix_multiply(A, B):
    """
    Multiply two matrices using basic operations.
    
    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)
    
    Returns:
        Result matrix (m x p)
    """
    return np.dot(A, B)


def normalize_array(array, min_val=0, max_val=255):
    """
    Normalize array to specified range.
    
    Args:
        array: Input array
        min_val: Minimum value of output range
        max_val: Maximum value of output range
    
    Returns:
        Normalized array
    """
    arr_min = custom_min(array)
    arr_max = custom_max(array)
    
    if arr_max == arr_min:
        return np.full_like(array, min_val, dtype=np.float64)
    
    normalized = (array - arr_min) / (arr_max - arr_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


def calculate_entropy(array):
    """
    Calculate entropy of an array (useful for histogram assessment).
    
    Args:
        array: Input array
    
    Returns:
        Entropy value
    """
    # Get histogram
    hist, _ = np.histogram(array.flatten(), bins=256, range=(0, 256))
    
    # Calculate probabilities
    total_pixels = array.size
    probabilities = hist / total_pixels
    
    # Remove zero probabilities
    probabilities = probabilities[probabilities > 0]
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))


def bilinear_weight(x, y):
    """Calculate bilinear interpolation weights."""
    wx = 1 - abs(x)
    wy = 1 - abs(y)
    return wx * wy


def cubic_weight(x):
    """
    Calculate cubic interpolation weight (bicubic).
    Using the standard cubic convolution kernel.
    """
    abs_x = abs(x)
    if abs_x <= 1:
        return 1.5 * abs_x**3 - 2.5 * abs_x**2 + 1
    elif abs_x < 2:
        return -0.5 * abs_x**3 + 2.5 * abs_x**2 - 4 * abs_x + 2
    else:
        return 0