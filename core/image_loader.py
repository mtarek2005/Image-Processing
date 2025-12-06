"""
Image loading and basic information extraction.
"""

import os
import numpy as np
from PIL import Image
from utils.validators import validate_image_format
from utils.helpers import pil_to_numpy, get_file_size_str


class ImageLoader:
    """Handle image loading and information extraction."""
    
    def __init__(self):
        self.image = None
        self.image_array = None
        self.file_path = None
        self.original_format = None
        
    def load_image(self, file_path):
        """
        Load image from file path.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Tuple of (numpy array, success message) or (None, error message)
        """
        # Validate file format
        is_valid, message = validate_image_format(file_path)
        if not is_valid:
            return None, message
        
        try:
            # Load image using PIL
            self.image = Image.open(file_path)
            self.file_path = file_path
            self.original_format = self.image.format
            
            # Convert to numpy array
            self.image_array = pil_to_numpy(self.image)
            
            return self.image_array, "Image loaded successfully"
            
        except Exception as e:
            return None, f"Error loading image: {str(e)}"
    
    def get_resolution(self):
        """
        Get image resolution.
        
        Returns:
            Tuple of (width, height) or None if no image loaded
        """
        if self.image_array is None:
            return None
        
        if self.image_array.ndim == 2:
            height, width = self.image_array.shape
        else:
            height, width = self.image_array.shape[:2]
        
        return (width, height)
    
    def get_size(self):
        """
        Get image file size.
        
        Returns:
            File size as formatted string or None if no image loaded
        """
        if self.file_path is None or not os.path.exists(self.file_path):
            return None
        
        size_bytes = os.path.getsize(self.file_path)
        return get_file_size_str(size_bytes)
    
    def get_size_bytes(self):
        """
        Get image file size in bytes.
        
        Returns:
            File size in bytes or None if no image loaded
        """
        if self.file_path is None or not os.path.exists(self.file_path):
            return None
        
        return os.path.getsize(self.file_path)
    
    def get_type(self):
        """
        Get image format type.
        
        Returns:
            Image format (PNG, JPEG, etc.) or None if no image loaded
        """
        if self.original_format is None:
            return None
        
        return self.original_format
    
    def get_mode(self):
        """
        Get image mode (RGB, RGBA, L, etc.).
        
        Returns:
            Image mode or None if no image loaded
        """
        if self.image is None:
            return None
        
        return self.image.mode
    
    def get_color_channels(self):
        """
        Get number of color channels.
        
        Returns:
            Number of channels or None if no image loaded
        """
        if self.image_array is None:
            return None
        
        if self.image_array.ndim == 2:
            return 1  # Grayscale
        else:
            return self.image_array.shape[2]  # RGB or RGBA
    
    def get_bit_depth(self):
        """
        Get bit depth of image.
        
        Returns:
            Bit depth or None if no image loaded
        """
        if self.image_array is None:
            return None
        
        dtype = self.image_array.dtype
        if dtype == np.uint8:
            return 8
        elif dtype == np.uint16:
            return 16
        elif dtype == np.float32 or dtype == np.float64:
            return 32
        else:
            return None
    
    def get_all_info(self):
        """
        Get all image information as a dictionary.
        
        Returns:
            Dictionary containing all image information
        """
        if self.image_array is None:
            return None
        
        resolution = self.get_resolution()
        
        info = {
            'resolution': f"{resolution[0]} x {resolution[1]}" if resolution else "N/A",
            'width': resolution[0] if resolution else 0,
            'height': resolution[1] if resolution else 0,
            'size': self.get_size() or "N/A",
            'size_bytes': self.get_size_bytes() or 0,
            'type': self.get_type() or "N/A",
            'mode': self.get_mode() or "N/A",
            'channels': self.get_color_channels() or 0,
            'bit_depth': self.get_bit_depth() or 0,
            'file_path': self.file_path or "N/A"
        }
        
        return info
    
    def validate_image(self):
        """
        Validate if image is properly loaded.
        
        Returns:
            Boolean indicating if image is valid
        """
        return self.image_array is not None and self.image_array.size > 0
    
    def get_image_array(self):
        """
        Get the loaded image as numpy array.
        
        Returns:
            Numpy array of image or None
        """
        return self.image_array
    
    def get_image_copy(self):
        """
        Get a copy of the loaded image array.
        
        Returns:
            Copy of numpy array or None
        """
        if self.image_array is None:
            return None
        
        return np.copy(self.image_array)