"""
Helper functions for common operations.
"""

import numpy as np
from PIL import Image


def numpy_to_pil(array):
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array
    
    Returns:
        PIL Image
    """
    # Ensure array is in valid range
    array = np.clip(array, 0, 255).astype(np.uint8)
    
    if array.ndim == 2:
        # Grayscale
        return Image.fromarray(array, mode='L')
    elif array.ndim == 3:
        if array.shape[2] == 3:
            # RGB
            return Image.fromarray(array, mode='RGB')
        elif array.shape[2] == 4:
            # RGBA
            return Image.fromarray(array, mode='RGBA')
    
    raise ValueError("Invalid array shape for image conversion")


def pil_to_numpy(image):
    """
    Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image
    
    Returns:
        Numpy array
    """
    return np.array(image)


def normalize_image(image, min_val=0, max_val=255):
    """
    Normalize image to specified range.
    
    Args:
        image: Input image array
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Normalized image
    """
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max == img_min:
        return np.full_like(image, min_val, dtype=np.float64)
    
    normalized = (image - img_min) / (img_max - img_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


def clip_values(image, min_val=0, max_val=255):
    """
    Clip image values to specified range.
    
    Args:
        image: Input image
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clipped image
    """
    return np.clip(image, min_val, max_val)


def rgb_to_single_channel(image, channel):
    """
    Extract single channel from RGB image.
    
    Args:
        image: RGB image
        channel: Channel index (0=R, 1=G, 2=B)
    
    Returns:
        Single channel image
    """
    if image.ndim != 3 or image.shape[2] < channel + 1:
        raise ValueError("Invalid channel selection")
    
    return image[:, :, channel]


def ensure_grayscale(image):
    """
    Ensure image is grayscale (convert if necessary).
    
    Args:
        image: Input image
    
    Returns:
        Grayscale image
    """
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        # Convert to grayscale using standard weights
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        raise ValueError("Invalid image dimensions")


def create_image_grid(images, rows, cols):
    """
    Create a grid of images for display.
    
    Args:
        images: List of images
        rows: Number of rows
        cols: Number of columns
    
    Returns:
        Grid image
    """
    if len(images) != rows * cols:
        raise ValueError("Number of images must match grid dimensions")
    
    # Assume all images have same shape
    img_height, img_width = images[0].shape[:2]
    
    # Create empty grid
    if images[0].ndim == 2:
        grid = np.zeros((img_height * rows, img_width * cols), dtype=images[0].dtype)
    else:
        grid = np.zeros((img_height * rows, img_width * cols, images[0].shape[2]), dtype=images[0].dtype)
    
    # Fill grid
    idx = 0
    for i in range(rows):
        for j in range(cols):
            y_start = i * img_height
            y_end = (i + 1) * img_height
            x_start = j * img_width
            x_end = (j + 1) * img_width
            
            grid[y_start:y_end, x_start:x_end] = images[idx]
            idx += 1
    
    return grid


def get_file_size_str(size_bytes):
    """
    Convert file size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "2.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def pad_image(image, pad_width, mode='edge'):
    """
    Pad image with specified mode.
    
    Args:
        image: Input image
        pad_width: Padding width
        mode: Padding mode ('edge', 'constant', 'reflect')
    
    Returns:
        Padded image
    """
    if image.ndim == 2:
        return np.pad(image, pad_width, mode=mode)
    else:
        return np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode=mode)


def calculate_psnr(original, processed):
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Args:
        original: Original image
        processed: Processed image
    
    Returns:
        PSNR value in dB
    """
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr