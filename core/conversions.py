import numpy as np
from utils.math_operations import custom_mean, custom_std
from utils.validators import validate_image_array, validate_threshold
from utils.helpers import ensure_grayscale


class ImageConverter:
    
    @staticmethod
    def rgb_to_grayscale(image):
        """
        Convert RGB image to grayscale using luminosity method.
        Uses standard weights: 0.299R + 0.587G + 0.114B
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Grayscale image as numpy array
        """
        # Validate input
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        # If already grayscale, return as is
        if image.ndim == 2:
            return image
        
        # Convert RGB to grayscale using custom weights
        # Standard luminosity method: Y = 0.299R + 0.587G + 0.114B
        if image.shape[2] >= 3:
            r_channel = image[:, :, 0].astype(np.float64)
            g_channel = image[:, :, 1].astype(np.float64)
            b_channel = image[:, :, 2].astype(np.float64)
            
            # Apply weights manually
            grayscale = (0.299 * r_channel + 
                        0.587 * g_channel + 
                        0.114 * b_channel)
            
            # Clip values to valid range
            grayscale = np.clip(grayscale, 0, 255)
            
            return grayscale.astype(np.uint8)
        
        return image
    
    @staticmethod
    def calculate_optimal_threshold(image):
        """
        Calculate threshold using mean of pixel intensities.
        
        Args:
            image: Grayscale image as numpy array
            
        Returns:
            Tuple of (threshold_value, is_optimal, explanation)
        """
        # Ensure image is grayscale
        gray_image = ensure_grayscale(image)
        
        # Calculate mean as threshold
        threshold = custom_mean(gray_image)
        
        # Evaluate if threshold is optimal
        # Calculate standard deviation to assess distribution
        std_dev = custom_std(gray_image)
        
        # Calculate how many pixels are above and below threshold
        pixels_below = np.sum(gray_image < threshold)
        pixels_above = np.sum(gray_image >= threshold)
        total_pixels = gray_image.size
        
        ratio_below = pixels_below / total_pixels
        ratio_above = pixels_above / total_pixels
        
        # Assess optimality
        # A good threshold should:
        # 1. Not be too extreme (too close to 0 or 255)
        # 2. Have reasonable distribution on both sides
        # 3. Standard deviation should be reasonable
        
        is_optimal = True
        explanations = []
        
        # Check if threshold is too extreme
        if threshold < 30:
            is_optimal = False
            explanations.append("Threshold is very low (< 30), image may be too dark")
        elif threshold > 225:
            is_optimal = False
            explanations.append("Threshold is very high (> 225), image may be too bright")
        
        # Check distribution balance
        if ratio_below < 0.1 or ratio_above < 0.1:
            is_optimal = False
            explanations.append(f"Unbalanced distribution: {ratio_below*100:.1f}% below, {ratio_above*100:.1f}% above threshold")
        
        # Check standard deviation
        if std_dev < 20:
            is_optimal = False
            explanations.append(f"Low standard deviation ({std_dev:.2f}), image has low contrast")
        
        if is_optimal:
            explanations.append(f"Mean-based threshold is reasonable for this image")
            explanations.append(f"Threshold value: {threshold:.2f}")
            explanations.append(f"Distribution: {ratio_below*100:.1f}% below, {ratio_above*100:.1f}% above")
            explanations.append(f"Standard deviation: {std_dev:.2f}")
        
        explanation = "\n".join(explanations)
        
        return threshold, is_optimal, explanation
    
    @staticmethod
    def grayscale_to_binary(image, threshold=None):
        """
        Convert grayscale image to binary using threshold.
        If threshold is None, calculates optimal threshold automatically.
        
        Args:
            image: Grayscale image as numpy array
            threshold: Threshold value (0-255), if None uses mean
            
        Returns:
            Binary image (0 or 255) as numpy array
        """
        # Ensure image is grayscale
        gray_image = ensure_grayscale(image)
        
        # Calculate threshold if not provided
        if threshold is None:
            threshold, _, _ = ImageConverter.calculate_optimal_threshold(gray_image)
        else:
            # Validate threshold
            is_valid, message = validate_threshold(threshold)
            if not is_valid:
                raise ValueError(message)
        
        # Apply thresholding
        # Pixels >= threshold become white (255)
        # Pixels < threshold become black (0)
        binary_image = np.zeros_like(gray_image, dtype=np.uint8)
        binary_image[gray_image >= threshold] = 255
        binary_image[gray_image < threshold] = 0
        
        return binary_image
    
    @staticmethod
    def evaluate_threshold(image, threshold):
        """
        Evaluate how good a specific threshold is for the image.
        
        Args:
            image: Grayscale image
            threshold: Threshold value to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        gray_image = ensure_grayscale(image)
        
        # Apply threshold
        binary = ImageConverter.grayscale_to_binary(gray_image, threshold)
        
        # Calculate metrics
        total_pixels = gray_image.size
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)
        
        white_ratio = white_pixels / total_pixels
        black_ratio = black_pixels / total_pixels
        
        # Calculate mean intensities of each group
        white_group = gray_image[gray_image >= threshold]
        black_group = gray_image[gray_image < threshold]
        
        mean_white = custom_mean(white_group) if white_group.size > 0 else 0
        mean_black = custom_mean(black_group) if black_group.size > 0 else 0
        
        # Calculate separation (higher is better)
        separation = abs(mean_white - mean_black)
        
        # Evaluate balance
        balance_score = 1 - abs(white_ratio - black_ratio)
        
        # Overall quality score (0-100)
        quality_score = (separation / 255 * 70) + (balance_score * 30)
        
        evaluation = {
            'threshold': threshold,
            'white_pixels': white_pixels,
            'black_pixels': black_pixels,
            'white_ratio': white_ratio * 100,
            'black_ratio': black_ratio * 100,
            'mean_white_group': mean_white,
            'mean_black_group': mean_black,
            'separation': separation,
            'balance_score': balance_score * 100,
            'quality_score': quality_score
        }
        
        return evaluation
    
    @staticmethod
    def otsu_threshold(image):
        """
        Calculate optimal threshold using Otsu's method.
        This is implemented as an alternative to mean-based threshold.
        
        Args:
            image: Grayscale image
            
        Returns:
            Optimal threshold value
        """
        gray_image = ensure_grayscale(image)
        
        # Calculate histogram
        histogram = np.zeros(256)
        for pixel_value in range(256):
            histogram[pixel_value] = np.sum(gray_image == pixel_value)
        
        # Normalize histogram
        total_pixels = gray_image.size
        histogram = histogram / total_pixels
        
        # Find optimal threshold
        best_threshold = 0
        max_variance = 0
        
        for t in range(256):
            # Calculate weights
            w0 = np.sum(histogram[:t]) if t > 0 else 0
            w1 = np.sum(histogram[t:])
            
            if w0 == 0 or w1 == 0:
                continue
            
            # Calculate means
            mu0 = np.sum(np.arange(t) * histogram[:t]) / w0 if w0 > 0 else 0
            mu1 = np.sum(np.arange(t, 256) * histogram[t:]) / w1 if w1 > 0 else 0
            
            # Calculate between-class variance
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > max_variance:
                max_variance = variance
                best_threshold = t
        
        return best_threshold