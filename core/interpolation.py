"""
Image interpolation methods for resolution enhancement.
"""

import numpy as np
from utils.validators import validate_interpolation_size, validate_image_array
from utils.math_operations import cubic_weight
from core.prog_upd import prog_upd


class Interpolator:
    """Handle image interpolation operations."""
    
    @staticmethod
    def nearest_neighbor(image, new_size):
        """
        Resize image using nearest neighbor interpolation.
        
        Args:
            image: Input image
            new_size: Tuple of (new_width, new_height)
            
        Returns:
            Resized image
        """
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        is_valid, message = validate_interpolation_size(new_size)
        if not is_valid:
            raise ValueError(message)
        
        new_width, new_height = new_size
        
        # Handle multi-channel images
        if image.ndim == 3:
            channels = image.shape[2]
            result_channels = []
            for c in range(channels):
                prog_upd.setchnl(c,channels)
                resized = Interpolator.nearest_neighbor(image[:, :, c], new_size)
                result_channels.append(resized)
            prog_upd.setchnl(0,1)
            return np.stack(result_channels, axis=2)
        
        # Get original dimensions
        old_height, old_width = image.shape
        
        # Create output image
        output = np.zeros((new_height, new_width), dtype=image.dtype)
        
        # Calculate scaling factors
        scale_x = old_width / new_width
        scale_y = old_height / new_height
        
        # Perform nearest neighbor interpolation
        for new_y in range(new_height):
            for new_x in range(new_width):
                prog_upd.setprog((new_y*new_width+new_x)/(new_height*new_width))
                # Find corresponding pixel in original image
                old_x = int(new_x * scale_x)
                old_y = int(new_y * scale_y)
                
                # Ensure within bounds
                old_x = min(old_x, old_width - 1)
                old_y = min(old_y, old_height - 1)
                
                # Copy pixel value
                output[new_y, new_x] = image[old_y, old_x]
        
        return output
    
    @staticmethod
    def bilinear(image, new_size):
        """
        Resize image using bilinear interpolation.
        
        Args:
            image: Input image
            new_size: Tuple of (new_width, new_height)
            
        Returns:
            Resized image
        """
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        is_valid, message = validate_interpolation_size(new_size)
        if not is_valid:
            raise ValueError(message)
        
        new_width, new_height = new_size
        
        # Handle multi-channel images
        if image.ndim == 3:
            channels = image.shape[2]
            result_channels = []
            for c in range(channels):
                prog_upd.setchnl(c,channels)
                resized = Interpolator.bilinear(image[:, :, c], new_size)
                result_channels.append(resized)
            prog_upd.setchnl(0,1)
            return np.stack(result_channels, axis=2)
        
        # Get original dimensions
        old_height, old_width = image.shape
        
        # Create output image
        output = np.zeros((new_height, new_width), dtype=np.float64)
        
        # Calculate scaling factors
        scale_x = (old_width - 1) / (new_width - 1) if new_width > 1 else 0
        scale_y = (old_height - 1) / (new_height - 1) if new_height > 1 else 0
        
        # Perform bilinear interpolation
        for new_y in range(new_height):
            for new_x in range(new_width):
                prog_upd.setprog((new_y*new_width+new_x)/(new_height*new_width))
                # Find corresponding position in original image
                old_x = new_x * scale_x
                old_y = new_y * scale_y
                
                # Get integer and fractional parts
                x0 = int(np.floor(old_x))
                y0 = int(np.floor(old_y))
                x1 = min(x0 + 1, old_width - 1)
                y1 = min(y0 + 1, old_height - 1)
                
                # Calculate interpolation weights
                wx = old_x - x0
                wy = old_y - y0
                
                # Bilinear interpolation
                value = (1 - wx) * (1 - wy) * image[y0, x0] + \
                        wx * (1 - wy) * image[y0, x1] + \
                        (1 - wx) * wy * image[y1, x0] + \
                        wx * wy * image[y1, x1]
                
                output[new_y, new_x] = value
        
        # Clip and convert to original dtype
        output = np.clip(output, 0, 255)
        return output.astype(image.dtype)
    
    @staticmethod
    def bicubic(image, new_size):
        """
        Resize image using bicubic interpolation.
        
        Args:
            image: Input image
            new_size: Tuple of (new_width, new_height)
            
        Returns:
            Resized image
        """
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        is_valid, message = validate_interpolation_size(new_size)
        if not is_valid:
            raise ValueError(message)
        
        new_width, new_height = new_size
        
        # Handle multi-channel images
        if image.ndim == 3:
            channels = image.shape[2]
            result_channels = []
            for c in range(channels):
                prog_upd.setchnl(c,channels)
                resized = Interpolator.bicubic(image[:, :, c], new_size)
                result_channels.append(resized)
            prog_upd.setchnl(0,1)
            return np.stack(result_channels, axis=2)
        
        # Get original dimensions
        old_height, old_width = image.shape
        
        # Create output image
        output = np.zeros((new_height, new_width), dtype=np.float64)
        
        # Calculate scaling factors
        scale_x = (old_width - 1) / (new_width - 1) if new_width > 1 else 0
        scale_y = (old_height - 1) / (new_height - 1) if new_height > 1 else 0
        
        # Perform bicubic interpolation
        for new_y in range(new_height):
            for new_x in range(new_width):
                prog_upd.setprog((new_y*new_width+new_x)/(new_height*new_width))
                # Find corresponding position in original image
                old_x = new_x * scale_x
                old_y = new_y * scale_y
                
                # Get integer part
                x_int = int(np.floor(old_x))
                y_int = int(np.floor(old_y))
                
                # Calculate interpolation over 4x4 neighborhood
                value = 0.0
                
                for j in range(-1, 3):
                    for i in range(-1, 3):
                        # Get coordinates with boundary checking
                        sample_x = np.clip(x_int + i, 0, old_width - 1)
                        sample_y = np.clip(y_int + j, 0, old_height - 1)
                        
                        # Calculate cubic weights
                        weight_x = cubic_weight(old_x - (x_int + i))
                        weight_y = cubic_weight(old_y - (y_int + j))
                        weight = weight_x * weight_y
                        
                        # Accumulate weighted value
                        value += weight * image[sample_y, sample_x]
                
                output[new_y, new_x] = value
        
        # Clip and convert to original dtype
        output = np.clip(output, 0, 255)
        return output.astype(image.dtype)
    
    @staticmethod
    def get_pixel_value(image, x, y):
        """
        Get pixel value with boundary checking.
        
        Args:
            image: Input image
            x, y: Coordinates (can be float)
            
        Returns:
            Pixel value
        """
        height, width = image.shape
        
        # Clip coordinates to valid range
        x = np.clip(int(round(x)), 0, width - 1)
        y = np.clip(int(round(y)), 0, height - 1)
        
        return image[y, x]
    
    @staticmethod
    def compare_methods(image, new_size):
        """
        Compare all interpolation methods.
        
        Args:
            image: Input image
            new_size: Tuple of (new_width, new_height)
            
        Returns:
            Dictionary with results from all methods
        """
        results = {
            'nearest': Interpolator.nearest_neighbor(image, new_size),
            'bilinear': Interpolator.bilinear(image, new_size),
            'bicubic': Interpolator.bicubic(image, new_size)
        }
        
        return results