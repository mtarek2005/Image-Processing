"""
Histogram computation and equalization operations.
"""

import numpy as np
from utils.math_operations import custom_mean, custom_std, calculate_entropy
from utils.helpers import ensure_grayscale
from utils.validators import validate_image_array


class HistogramProcessor:
    """Handle histogram operations."""
    
    @staticmethod
    def compute_histogram(image):
        """
        Compute histogram of grayscale image.
        
        Args:
            image: Grayscale image
            
        Returns:
            256-element array representing histogram
        """
        gray_image = ensure_grayscale(image)
        
        is_valid, message = validate_image_array(gray_image)
        if not is_valid:
            raise ValueError(message)
        
        # Initialize histogram array
        histogram = np.zeros(256, dtype=np.int32)
        
        # Count occurrences of each intensity value
        for intensity in range(256):
            histogram[intensity] = np.sum(gray_image == intensity)
        
        return histogram
    
    @staticmethod
    def compute_normalized_histogram(image):
        """
        Compute normalized histogram (probabilities).
        
        Args:
            image: Grayscale image
            
        Returns:
            256-element array of probabilities
        """
        histogram = HistogramProcessor.compute_histogram(image)
        total_pixels = np.sum(histogram)
        
        if total_pixels == 0:
            return histogram
        
        return histogram / total_pixels
    
    @staticmethod
    def compute_cdf(histogram):
        """
        Compute cumulative distribution function from histogram.
        
        Args:
            histogram: Histogram array
            
        Returns:
            Cumulative distribution function
        """
        cdf = np.zeros_like(histogram, dtype=np.float64)
        cdf[0] = histogram[0]
        
        for i in range(1, len(histogram)):
            cdf[i] = cdf[i-1] + histogram[i]
        
        return cdf
    
    @staticmethod
    def assess_histogram_quality(image):
        """
        Assess whether histogram is good or not.
        
        Args:
            image: Grayscale image
            
        Returns:
            Tuple of (quality_score, is_good, explanation)
        """
        gray_image = ensure_grayscale(image)
        histogram = HistogramProcessor.compute_histogram(gray_image)
        
        # Normalize histogram
        total_pixels = gray_image.size
        normalized_hist = histogram / total_pixels
        
        # Criteria for good histogram:
        # 1. Wide distribution (uses most of the dynamic range)
        # 2. Good contrast (standard deviation)
        # 3. Not heavily skewed to one side
        # 4. High entropy
        
        # Calculate metrics
        mean_intensity = custom_mean(gray_image)
        std_intensity = custom_std(gray_image)
        entropy = calculate_entropy(gray_image)
        
        # Find intensity range being used
        non_zero_bins = np.where(histogram > 0)[0]
        if len(non_zero_bins) > 0:
            min_intensity = non_zero_bins[0]
            max_intensity = non_zero_bins[-1]
            intensity_range = max_intensity - min_intensity
        else:
            intensity_range = 0
        
        # Calculate scores for each criterion (0-100)
        
        # 1. Dynamic range score (should use most of 0-255 range)
        range_score = (intensity_range / 255) * 100
        
        # 2. Contrast score (based on standard deviation)
        # Good contrast: std > 50
        contrast_score = min((std_intensity / 50) * 100, 100)
        
        # 3. Balance score (mean should be around 127)
        balance_score = (1 - abs(mean_intensity - 127.5) / 127.5) * 100
        
        # 4. Entropy score (max entropy is 8 for 8-bit image)
        entropy_score = (entropy / 8) * 100
        
        # Overall quality score (weighted average)
        quality_score = (range_score * 0.3 + 
                        contrast_score * 0.3 + 
                        balance_score * 0.2 + 
                        entropy_score * 0.2)
        
        # Determine if histogram is good
        is_good = quality_score >= 60
        
        # Generate explanation
        explanations = []
        explanations.append(f"Overall Quality Score: {quality_score:.2f}/100")
        explanations.append(f"\nDetailed Analysis:")
        explanations.append(f"- Dynamic Range: {intensity_range}/255 (Score: {range_score:.2f}/100)")
        
        if range_score < 50:
            explanations.append("  ⚠ Image uses limited intensity range")
        else:
            explanations.append("  ✓ Good dynamic range")
        
        explanations.append(f"- Contrast (Std Dev): {std_intensity:.2f} (Score: {contrast_score:.2f}/100)")
        if contrast_score < 50:
            explanations.append("  ⚠ Low contrast image")
        else:
            explanations.append("  ✓ Good contrast")
        
        explanations.append(f"- Balance (Mean): {mean_intensity:.2f} (Score: {balance_score:.2f}/100)")
        if mean_intensity < 50:
            explanations.append("  ⚠ Image is too dark")
        elif mean_intensity > 200:
            explanations.append("  ⚠ Image is too bright")
        else:
            explanations.append("  ✓ Well-balanced brightness")
        
        explanations.append(f"- Entropy: {entropy:.2f}/8 (Score: {entropy_score:.2f}/100)")
        if entropy_score < 50:
            explanations.append("  ⚠ Low information content")
        else:
            explanations.append("  ✓ Good information distribution")
        
        if is_good:
            explanations.append(f"\n✓ The histogram is GOOD - image has acceptable quality")
        else:
            explanations.append(f"\n✗ The histogram is POOR - consider histogram equalization")
        
        explanation = "\n".join(explanations)
        
        return quality_score, is_good, explanation
    
    @staticmethod
    def histogram_equalization(image):
        """
        Apply histogram equalization to enhance image contrast.
        
        Args:
            image: Grayscale image
            
        Returns:
            Equalized image
        """
        gray_image = ensure_grayscale(image)
        
        # Compute histogram
        histogram = HistogramProcessor.compute_histogram(gray_image)
        
        # Compute CDF
        cdf = HistogramProcessor.compute_cdf(histogram)
        
        # Normalize CDF
        cdf_normalized = cdf / cdf[-1]
        
        # Create transformation function
        # Map each intensity to new intensity based on CDF
        transformation = np.round(cdf_normalized * 255).astype(np.uint8)
        
        # Apply transformation
        equalized = transformation[gray_image.astype(np.uint8)]
        
        return equalized
    
    @staticmethod
    def histogram_matching(image, target_histogram):
        """
        Match histogram of image to target histogram.
        
        Args:
            image: Input grayscale image
            target_histogram: Target histogram to match
            
        Returns:
            Image with matched histogram
        """
        gray_image = ensure_grayscale(image)
        
        # Compute CDFs
        source_hist = HistogramProcessor.compute_histogram(gray_image)
        source_cdf = HistogramProcessor.compute_cdf(source_hist)
        target_cdf = HistogramProcessor.compute_cdf(target_histogram)
        
        # Normalize CDFs
        source_cdf_normalized = source_cdf / source_cdf[-1]
        target_cdf_normalized = target_cdf / target_cdf[-1]
        
        # Create mapping
        mapping = np.zeros(256, dtype=np.uint8)
        
        for i in range(256):
            # Find closest match in target CDF
            diff = np.abs(target_cdf_normalized - source_cdf_normalized[i])
            mapping[i] = np.argmin(diff)
        
        # Apply mapping
        matched = mapping[gray_image.astype(np.uint8)]
        
        return matched
    
    @staticmethod
    def adaptive_histogram_equalization(image, tile_size=8):
        """
        Apply adaptive histogram equalization (CLAHE-like).
        
        Args:
            image: Grayscale image
            tile_size: Size of local region for equalization
            
        Returns:
            Adaptively equalized image
        """
        gray_image = ensure_grayscale(image)
        height, width = gray_image.shape
        
        # Create output image
        output = np.zeros_like(gray_image, dtype=np.float64)
        
        # Process each tile
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Extract tile
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                tile = gray_image[y:y_end, x:x_end]
                
                # Equalize tile
                equalized_tile = HistogramProcessor.histogram_equalization(tile)
                
                # Store result
                output[y:y_end, x:x_end] = equalized_tile
        
        return output.astype(np.uint8)