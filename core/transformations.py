"""
Affine transformation operations: Translation, Scaling, Rotation, Shearing.
"""

import numpy as np
from utils.validators import validate_transformation_params, validate_image_array
from utils.helpers import ensure_grayscale
from core.prog_upd import prog_upd


class AffineTransformations:
    """Handle affine transformation operations."""
    
    @staticmethod
    def create_transformation_matrix(transform_type, **params):
        """
        Create transformation matrix for specified transform.
        
        Args:
            transform_type: Type of transformation
            **params: Transformation parameters
            
        Returns:
            3x3 transformation matrix
        """
        if transform_type == 'translate':
            tx = params.get('tx', 0)
            ty = params.get('ty', 0)
            return np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ], dtype=np.float64)
        
        elif transform_type == 'scale':
            sx = params.get('sx', 1)
            sy = params.get('sy', 1)
            return np.array([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]
            ], dtype=np.float64)
        
        elif transform_type == 'rotate':
            angle = params.get('angle', 0)
            # Convert to radians
            theta = np.radians(angle)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ], dtype=np.float64)
        
        elif transform_type == 'shear_x':
            factor = params.get('factor', 0)
            return np.array([
                [1, factor, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float64)
        
        elif transform_type == 'shear_y':
            factor = params.get('factor', 0)
            return np.array([
                [1, 0, 0],
                [factor, 1, 0],
                [0, 0, 1]
            ], dtype=np.float64)
        
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")
    
    @staticmethod
    def apply_transformation_matrix(image, matrix, output_shape=None):
        """
        Apply transformation matrix to image using inverse mapping.
        
        Args:
            image: Input image
            matrix: 3x3 transformation matrix
            output_shape: Desired output shape (height, width), if None uses input shape
            
        Returns:
            Transformed image
        """
        is_valid, message = validate_image_array(image)
        if not is_valid:
            raise ValueError(message)
        
        # Handle multi-channel images
        if image.ndim == 3:
            channels = image.shape[2]
            result_channels = []
            for c in range(channels):
                prog_upd.setchnl(c,channels)
                transformed = AffineTransformations.apply_transformation_matrix(
                    image[:, :, c], matrix, output_shape
                )
                result_channels.append(transformed)
            prog_upd.setchnl(0,1)
            return np.stack(result_channels, axis=2)
        
        # Get dimensions
        height, width = image.shape
        
        if output_shape is None:
            output_shape = (height, width)
        
        out_height, out_width = output_shape
        
        # Create output image
        output = np.zeros((out_height, out_width), dtype=image.dtype)
        
        # Calculate inverse transformation matrix
        try:
            inv_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Transformation matrix is singular and cannot be inverted")
        
        # Apply inverse mapping
        for out_y in range(out_height):
            for out_x in range(out_width):
                prog_upd.setprog((out_y*out_width+out_x)/(out_height*out_width))
                # Create homogeneous coordinates
                out_coords = np.array([out_x, out_y, 1])
                
                # Apply inverse transformation
                src_coords = inv_matrix @ out_coords
                src_x = src_coords[0]
                src_y = src_coords[1]
                
                # Check if source coordinates are within bounds
                if 0 <= src_x < width - 1 and 0 <= src_y < height - 1:
                    # Bilinear interpolation
                    x0 = int(np.floor(src_x))
                    x1 = x0 + 1
                    y0 = int(np.floor(src_y))
                    y1 = y0 + 1
                    
                    # Calculate interpolation weights
                    wx = src_x - x0
                    wy = src_y - y0
                    
                    # Interpolate
                    value = (1 - wx) * (1 - wy) * image[y0, x0] + \
                            wx * (1 - wy) * image[y0, x1] + \
                            (1 - wx) * wy * image[y1, x0] + \
                            wx * wy * image[y1, x1]
                    
                    output[out_y, out_x] = np.clip(value, 0, 255)
        
        return output.astype(image.dtype)
    
    @staticmethod
    def translate(image, tx, ty):
        """
        Translate image by (tx, ty) pixels.
        
        Args:
            image: Input image
            tx: Translation in x direction (positive = right)
            ty: Translation in y direction (positive = down)
            
        Returns:
            Translated image
        """
        params = {'tx': tx, 'ty': ty}
        is_valid, message = validate_transformation_params(params, 'translate')
        if not is_valid:
            raise ValueError(message)
        
        matrix = AffineTransformations.create_transformation_matrix('translate', tx=tx, ty=ty)
        return AffineTransformations.apply_transformation_matrix(image, matrix)
    
    @staticmethod
    def scale(image, sx, sy):
        """
        Scale image by factors (sx, sy).
        
        Args:
            image: Input image
            sx: Scale factor in x direction
            sy: Scale factor in y direction
            
        Returns:
            Scaled image
        """
        params = {'sx': sx, 'sy': sy}
        is_valid, message = validate_transformation_params(params, 'scale')
        if not is_valid:
            raise ValueError(message)
        
        # Calculate new dimensions
        height, width = image.shape[:2]
        new_height = int(height * sy)
        new_width = int(width * sx)
        
        matrix = AffineTransformations.create_transformation_matrix('scale', sx=sx, sy=sy)
        return AffineTransformations.apply_transformation_matrix(
            image, matrix, output_shape=(new_height, new_width)
        )
    
    @staticmethod
    def rotate(image, angle, center=None):
        """
        Rotate image by angle (in degrees) around center.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counterclockwise)
            center: Center of rotation (x, y), if None uses image center
            
        Returns:
            Rotated image
        """
        is_valid, message = validate_transformation_params(angle, 'rotate')
        if not is_valid:
            raise ValueError(message)
        
        height, width = image.shape[:2]
        
        # Use image center if not specified
        if center is None:
            center = (width / 2, height / 2)
        
        cx, cy = center
        
        # Create rotation matrix around center
        # 1. Translate to origin
        # 2. Rotate
        # 3. Translate back
        T1 = AffineTransformations.create_transformation_matrix('translate', tx=-cx, ty=-cy)
        R = AffineTransformations.create_transformation_matrix('rotate', angle=angle)
        T2 = AffineTransformations.create_transformation_matrix('translate', tx=cx, ty=cy)
        
        # Combine transformations
        matrix = T2 @ R @ T1
        
        # Calculate output size to fit entire rotated image
        theta = np.radians(angle)
        cos_theta = abs(np.cos(theta))
        sin_theta = abs(np.sin(theta))
        
        new_width = int(width * cos_theta + height * sin_theta)
        new_height = int(width * sin_theta + height * cos_theta)
        
        # Adjust translation to center the rotated image
        matrix[0, 2] += (new_width - width) / 2
        matrix[1, 2] += (new_height - height) / 2
        
        return AffineTransformations.apply_transformation_matrix(
            image, matrix, output_shape=(new_height, new_width)
        )
    
    @staticmethod
    def shear_x(image, factor):
        """
        Apply shear transformation in x direction.
        
        Args:
            image: Input image
            factor: Shear factor
            
        Returns:
            Sheared image
        """
        is_valid, message = validate_transformation_params(factor, 'shear')
        if not is_valid:
            raise ValueError(message)
        
        height, width = image.shape[:2]
        
        # Calculate new width to accommodate shear
        new_width = int(width + abs(factor * height))
        
        matrix = AffineTransformations.create_transformation_matrix('shear_x', factor=factor)
        
        # Adjust translation if factor is negative
        if factor < 0:
            matrix[0, 2] = abs(factor * height)
        
        return AffineTransformations.apply_transformation_matrix(
            image, matrix, output_shape=(height, new_width)
        )
    
    @staticmethod
    def shear_y(image, factor):
        """
        Apply shear transformation in y direction.
        
        Args:
            image: Input image
            factor: Shear factor
            
        Returns:
            Sheared image
        """
        is_valid, message = validate_transformation_params(factor, 'shear')
        if not is_valid:
            raise ValueError(message)
        
        height, width = image.shape[:2]
        
        # Calculate new height to accommodate shear
        new_height = int(height + abs(factor * width))
        
        matrix = AffineTransformations.create_transformation_matrix('shear_y', factor=factor)
        
        # Adjust translation if factor is negative
        if factor < 0:
            matrix[1, 2] = abs(factor * width)
        
        return AffineTransformations.apply_transformation_matrix(
            image, matrix, output_shape=(new_height, width)
        )
    
    @staticmethod
    def crop(image, x1, y1, x2, y2):
        """
        Crop image to specified region.
        
        Args:
            image: Input image
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
            
        Returns:
            Cropped image
        """
        from utils.validators import validate_crop_region
        
        is_valid, message = validate_crop_region((x1, y1, x2, y2), image.shape)
        if not is_valid:
            raise ValueError(message)
        
        # Perform crop
        if image.ndim == 2:
            return image[y1:y2, x1:x2]
        else:
            return image[y1:y2, x1:x2, :]