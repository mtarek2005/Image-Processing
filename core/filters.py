"""
Low-pass and high-pass filtering operations.
"""

import numpy as np
from utils.math_operations import custom_convolution, custom_median
from utils.validators import validate_kernel_size, validate_image_array
from utils.helpers import ensure_grayscale, normalize_image


class FilterOperations:
    """Handle image filtering operations."""
    
    @staticmethod
    def create_gaussian_kernel(size, sigma):
        """
        Create Gaussian kernel for filtering.
        
        Args:
            size: Kernel size (must be odd)
            sigma: Standard deviation
            
        Returns:
            Gaussian kernel
        """
        is_valid, message = validate_kernel_size(size)
        if not is_valid:
            raise ValueError(message)
        
        # Create coordinate grids
        center = size // 2
        kernel = np.zeros((size, size), dtype=np.float64)
        
        # Calculate Gaussian values
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    @staticmethod
    def gaussian_filter(image, kernel_size=19, sigma=3):
        """
        Apply Gaussian filter (low-pass filter).
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (default 19x19)
            sigma: Standard deviation (default 3)
            
        Returns:
            Filtered image
        """
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        # Handle multi-channel images
        if image.ndim == 3:
            channels = image.shape[2]
            result_channels = []
            for c in range(channels):
                filtered = FilterOperations.gaussian_filter(
                    image[:, :, c], kernel_size, sigma
                )
                result_channels.append(filtered)
            return np.stack(result_channels, axis=2)
        
        # Ensure grayscale for single channel
        gray_image = ensure_grayscale(image)
        
        # Create Gaussian kernel
        kernel = FilterOperations.create_gaussian_kernel(kernel_size, sigma)
        
        # Apply convolution
        filtered = custom_convolution(gray_image.astype(np.float64), kernel)
        
        # Clip values
        filtered = np.clip(filtered, 0, 255)
        
        return filtered.astype(np.uint8)
    
    @staticmethod
    def median_filter(image, kernel_size=7):
        """
        Apply median filter (low-pass filter, good for salt-and-pepper noise).
        
        Args:
            image: Input image
            kernel_size: Size of median filter window (default 7x7)
            
        Returns:
            Filtered image
        """
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        is_valid, message = validate_kernel_size(kernel_size)
        if not is_valid:
            raise ValueError(message)
        
        # Handle multi-channel images
        if image.ndim == 3:
            channels = image.shape[2]
            result_channels = []
            for c in range(channels):
                filtered = FilterOperations.median_filter(
                    image[:, :, c], kernel_size
                )
                result_channels.append(filtered)
            return np.stack(result_channels, axis=2)
        
        # Ensure grayscale
        gray_image = ensure_grayscale(image)
        height, width = gray_image.shape
        
        # Pad image
        pad = kernel_size // 2
        padded = np.pad(gray_image, pad, mode='edge')
        
        # Create output image
        output = np.zeros_like(gray_image, dtype=np.uint8)
        
        # Apply median filter
        for i in range(height):
            for j in range(width):
                # Extract window
                window = padded[i:i+kernel_size, j:j+kernel_size]
                
                # Calculate median
                output[i, j] = custom_median(window)
        
        return output
    
    @staticmethod
    def laplacian_filter(image):
        """
        Apply Laplacian filter (high-pass, second derivative).
        Detects edges and rapid intensity changes.
        
        Args:
            image: Input image
            
        Returns:
            Filtered image
        """
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        # Handle multi-channel images
        if image.ndim == 3:
            channels = image.shape[2]
            result_channels = []
            for c in range(channels):
                filtered = FilterOperations.laplacian_filter(image[:, :, c])
                result_channels.append(filtered)
            return np.stack(result_channels, axis=2)
        
        # Ensure grayscale
        gray_image = ensure_grayscale(image)
        
        # Laplacian kernel (detects edges in all directions)
        # Using the standard 3x3 Laplacian kernel
        kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float64)
        
        # Alternative: Laplacian kernel with diagonals
        # kernel = np.array([
        #     [1,  1, 1],
        #     [1, -8, 1],
        #     [1,  1, 1]
        # ], dtype=np.float64)
        
        # Apply convolution
        filtered = custom_convolution(gray_image.astype(np.float64), kernel)
        
        # Normalize to 0-255 range
        filtered = normalize_image(filtered, 0, 255)
        
        return filtered.astype(np.uint8)
    
    @staticmethod
    def sobel_filter(image):
        """
        Apply Sobel filter (high-pass, gradient magnitude).
        Detects edges using gradient approximation.
        
        Args:
            image: Input image
            
        Returns:
            Filtered image (gradient magnitude)
        """
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        # Handle multi-channel images
        if image.ndim == 3:
            channels = image.shape[2]
            result_channels = []
            for c in range(channels):
                filtered = FilterOperations.sobel_filter(image[:, :, c])
                result_channels.append(filtered)
            return np.stack(result_channels, axis=2)
        
        # Ensure grayscale
        gray_image = ensure_grayscale(image)
        
        # Sobel kernels for x and y directions
        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float64)
        
        sobel_y = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float64)
        
        # Apply convolution
        gradient_x = custom_convolution(gray_image.astype(np.float64), sobel_x)
        gradient_y = custom_convolution(gray_image.astype(np.float64), sobel_y)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize to 0-255 range
        gradient_magnitude = normalize_image(gradient_magnitude, 0, 255)
        
        return gradient_magnitude.astype(np.uint8)
    
    @staticmethod
    def gradient_filter(image):
        """
        Apply gradient filter (high-pass, first derivatives).
        Simple gradient approximation using forward differences.
        
        Args:
            image: Input image
            
        Returns:
            Filtered image (gradient magnitude)
        """
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        # Handle multi-channel images
        if image.ndim == 3:
            channels = image.shape[2]
            result_channels = []
            for c in range(channels):
                filtered = FilterOperations.gradient_filter(image[:, :, c])
                result_channels.append(filtered)
            return np.stack(result_channels, axis=2)
        
        # Ensure grayscale
        gray_image = ensure_grayscale(image)
        
        # Simple gradient kernels (first derivatives)
        gradient_x_kernel = np.array([
            [-1, 1]
        ], dtype=np.float64)
        
        gradient_y_kernel = np.array([
            [-1],
            [ 1]
        ], dtype=np.float64)
        
        # Calculate gradients using convolution
        height, width = gray_image.shape
        gradient_x = np.zeros_like(gray_image, dtype=np.float64)
        gradient_y = np.zeros_like(gray_image, dtype=np.float64)
        
        # X gradient (horizontal)
        for i in range(height):
            for j in range(width - 1):
                gradient_x[i, j] = float(gray_image[i, j+1]) - float(gray_image[i, j])
        
        # Y gradient (vertical)
        for i in range(height - 1):
            for j in range(width):
                gradient_y[i, j] = float(gray_image[i+1, j]) - float(gray_image[i, j])
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize to 0-255 range
        gradient_magnitude = normalize_image(gradient_magnitude, 0, 255)
        
        return gradient_magnitude.astype(np.uint8)
    
    @staticmethod
    def prewitt_filter(image):
        """
        Apply Prewitt filter (alternative edge detection).
        
        Args:
            image: Input image
            
        Returns:
            Filtered image
        """
        gray_image = ensure_grayscale(image)
        
        # Prewitt kernels
        prewitt_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float64)
        
        prewitt_y = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ], dtype=np.float64)
        
        # Apply convolution
        gradient_x = custom_convolution(gray_image.astype(np.float64), prewitt_x)
        gradient_y = custom_convolution(gray_image.astype(np.float64), prewitt_y)
        
        # Calculate magnitude
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        magnitude = normalize_image(magnitude, 0, 255)
        
        return magnitude.astype(np.uint8)
    
    @staticmethod
    def roberts_filter(image):
        """
        Apply Roberts cross filter (simple edge detection).
        
        Args:
            image: Input image
            
        Returns:
            Filtered image
        """
        gray_image = ensure_grayscale(image)
        
        # Roberts kernels
        roberts_x = np.array([
            [1,  0],
            [0, -1]
        ], dtype=np.float64)
        
        roberts_y = np.array([
            [0,  1],
            [-1, 0]
        ], dtype=np.float64)
        
        # Apply convolution
        gradient_x = custom_convolution(gray_image.astype(np.float64), roberts_x)
        gradient_y = custom_convolution(gray_image.astype(np.float64), roberts_y)
        
        # Calculate magnitude
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        magnitude = normalize_image(magnitude, 0, 255)
        
        return magnitude.astype(np.uint8)
    
    @staticmethod
    def sharpen_filter(image):
        """
        Apply sharpening filter to enhance edges.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        gray_image = ensure_grayscale(image)
        
        # Sharpening kernel
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float64)
        
        # Apply convolution
        sharpened = custom_convolution(gray_image.astype(np.float64), kernel)
        sharpened = np.clip(sharpened, 0, 255)
        
        return sharpened.astype(np.uint8)
    
    @staticmethod
    def unsharp_mask(image, amount=1.5):
        """
        Apply unsharp masking for edge enhancement.
        
        Args:
            image: Input image
            amount: Enhancement amount
            
        Returns:
            Enhanced image
        """
        gray_image = ensure_grayscale(image)
        
        # Apply Gaussian blur
        blurred = FilterOperations.gaussian_filter(gray_image, kernel_size=5, sigma=1.0)
        
        # Calculate mask
        mask = gray_image.astype(np.float64) - blurred.astype(np.float64)
        
        # Add weighted mask to original
        enhanced = gray_image.astype(np.float64) + amount * mask
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)