"""
Validation functions for input parameters and file formats.
"""

import os
import numpy as np


def validate_image_format(file_path):
    """
    Validate if file is a supported image format.
    
    Args:
        file_path: Path to image file
    
    Returns:
        Boolean indicating validity
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in valid_extensions:
        return False, f"Unsupported format. Valid formats: {', '.join(valid_extensions)}"
    
    return True, "Valid format"


def validate_kernel_size(size):
    """
    Validate kernel size (must be positive odd integer).
    
    Args:
        size: Kernel size
    
    Returns:
        Boolean indicating validity
    """
    if not isinstance(size, int):
        return False, "Kernel size must be an integer"
    
    if size <= 0:
        return False, "Kernel size must be positive"
    
    if size % 2 == 0:
        return False, "Kernel size must be odd"
    
    return True, "Valid kernel size"


def validate_transformation_params(params, param_type):
    """
    Validate transformation parameters.
    
    Args:
        params: Transformation parameters (dict or values)
        param_type: Type of transformation
    
    Returns:
        Boolean indicating validity
    """
    if param_type == 'translate':
        if not isinstance(params, dict) or 'tx' not in params or 'ty' not in params:
            return False, "Translation requires 'tx' and 'ty' parameters"
        return True, "Valid translation parameters"
    
    elif param_type == 'scale':
        if not isinstance(params, dict) or 'sx' not in params or 'sy' not in params:
            return False, "Scaling requires 'sx' and 'sy' parameters"
        if params['sx'] <= 0 or params['sy'] <= 0:
            return False, "Scale factors must be positive"
        return True, "Valid scale parameters"
    
    elif param_type == 'rotate':
        if not isinstance(params, (int, float)):
            return False, "Rotation requires an angle value"
        return True, "Valid rotation parameter"
    
    elif param_type == 'shear':
        if not isinstance(params, (int, float)):
            return False, "Shear requires a factor value"
        return True, "Valid shear parameter"
    
    return False, "Unknown transformation type"


def validate_image_array(image):
    """
    Validate if array is a proper image.
    
    Args:
        image: Numpy array
    
    Returns:
        Boolean indicating validity
    """
    if not isinstance(image, np.ndarray):
        return False, "Image must be a numpy array"
    
    if image.ndim not in [2, 3]:
        return False, "Image must be 2D (grayscale) or 3D (color)"
    
    if image.ndim == 3 and image.shape[2] not in [3, 4]:
        return False, "Color image must have 3 (RGB) or 4 (RGBA) channels"
    
    if image.size == 0:
        return False, "Image is empty"
    
    return True, "Valid image array"


def validate_threshold(threshold):
    """
    Validate threshold value for binary conversion.
    
    Args:
        threshold: Threshold value
    
    Returns:
        Boolean indicating validity
    """
    if not isinstance(threshold, (int, float)):
        return False, "Threshold must be a number"
    
    if threshold < 0 or threshold > 255:
        return False, "Threshold must be between 0 and 255"
    
    return True, "Valid threshold"


def validate_interpolation_size(new_size):
    """
    Validate new size for interpolation.
    
    Args:
        new_size: Tuple of (width, height)
    
    Returns:
        Boolean indicating validity
    """
    if not isinstance(new_size, (tuple, list)) or len(new_size) != 2:
        return False, "Size must be a tuple of (width, height)"
    
    if new_size[0] <= 0 or new_size[1] <= 0:
        return False, "Dimensions must be positive"
    
    if not isinstance(new_size[0], int) or not isinstance(new_size[1], int):
        return False, "Dimensions must be integers"
    
    return True, "Valid size"


def validate_crop_region(region, image_shape):
    """
    Validate crop region coordinates.
    
    Args:
        region: Tuple of (x1, y1, x2, y2)
        image_shape: Shape of the image
    
    Returns:
        Boolean indicating validity
    """
    if not isinstance(region, (tuple, list)) or len(region) != 4:
        return False, "Region must be a tuple of (x1, y1, x2, y2)"
    
    x1, y1, x2, y2 = region
    height, width = image_shape[:2]
    
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False, "Crop region exceeds image boundaries"
    
    if x2 <= x1 or y2 <= y1:
        return False, "Invalid crop region dimensions"
    
    return True, "Valid crop region"