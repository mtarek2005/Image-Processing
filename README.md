# Image Processing Suite

A comprehensive desktop application for image processing operations built with Python and Tkinter. This application provides a user-friendly GUI for performing various image manipulation, analysis, and compression techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Description

Image Processing Suite is an educational and practical tool that implements various image processing algorithms from scratch (without relying on high-level built-in functions). It features a modern dark-themed GUI with side-by-side image comparison and real-time processing feedback.

##  Features

- **Custom Implementation**: All algorithms implemented using basic operations (no reliance on complete built-in functions)
- **Visual Comparison**: Side-by-side display of original and processed images
- **Real-time Feedback**: Processing time and image information display
- **10 Compression Methods**: Compare various compression techniques
- **Multiple Filter Options**: Low-pass and high-pass filtering
- **Histogram Analysis**: Complete histogram tools with quality assessment

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Clone the Repository
```bash
git clone https://github.com/yourusername/image-processing.git
cd image-processing
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy==1.24.3` - Numerical operations and array manipulation
- `Pillow==10.0.0` - Image I/O operations
- `matplotlib==3.7.2` - Histogram visualization

##  Usage

Run the application:
```bash
python main.py
```

##  Project Structure
```
image_processing_suite/
│
├── main.py                          # Application entry point
├── requirements.txt                 # Project dependencies
├── README.md                        # This file
│
├── gui/                            # GUI components
│   ├── __init__.py
│   ├── main_window.py              # Main application window
│   ├── image_display.py            # Image display widgets
│   ├── control_panel.py            # Control buttons
│   └── styles.py                   # Styling constants
│
├── core/                           # Core processing modules
│   ├── __init__.py
│   ├── image_loader.py             # Image loading
│   ├── conversions.py              # Color conversions
│   ├── transformations.py          # Affine transformations
│   ├── interpolation.py            # Image interpolation
│   ├── histogram.py                # Histogram operations
│   ├── filters.py                  # Image filters
│   └── compression.py              # Compression algorithms
│
└── utils/                          # Utility functions
    ├── __init__.py
    ├── math_operations.py          # Mathematical operations
    ├── validators.py               # Input validation
    └── helpers.py                  # Helper functions
```

##  GUI Features Explained

###  File Operations

#### **Upload Image**
- Opens a file dialog to select an image from your computer
- Supported formats: PNG, JPEG, BMP, TIFF, GIF
- Displays image information: resolution, file size, and format
- Loads the image into both the original and current processing pipeline

#### **Save Image**
- Saves the processed image to your computer
- Choose output format (PNG or JPEG)
- Only available after processing an image

#### **Reset**
- Restores the current image to the original uploaded image
- Clears all applied processing operations
- Useful for starting over without re-uploading

---

###  Conversions

#### **Grayscale**
- Converts RGB color image to grayscale
- Uses luminosity method: `Y = 0.299R + 0.587G + 0.114B`
- Preserves image dimensions
- Essential for many other operations

#### **Binary (Auto Threshold)**
- Converts grayscale image to binary (black and white)
- Automatically calculates optimal threshold using mean pixel intensity
- Displays threshold evaluation including:
  - Calculated threshold value
  - Whether threshold is optimal
  - Distribution analysis
  - Standard deviation assessment

#### **Binary (Custom Threshold)**
- Converts grayscale image to binary with user-specified threshold
- Input: Threshold value (0-255)
- Pixels ≥ threshold become white (255)
- Pixels < threshold become black (0)

---

###  Transformations

All transformations use affine transformation matrices with inverse mapping.

#### **Translate**
- Shifts image by specified pixels in X and Y directions
- Input: `tx` (horizontal shift), `ty` (vertical shift)
- Positive values move right/down, negative values move left/up
- Example: tx=50, ty=30 moves image 50 pixels right and 30 pixels down

#### **Scale**
- Resizes image by scale factors
- Input: `sx` (horizontal scale), `sy` (vertical scale)
- Values > 1 enlarge, values < 1 shrink
- Example: sx=2, sy=1.5 doubles width and increases height by 50%
- Uses bilinear interpolation for smooth results

#### **Rotate**
- Rotates image around its center
- Input: Angle in degrees
- Positive angles rotate counterclockwise
- Output size adjusted to fit entire rotated image
- Example: 45° creates a diamond orientation

#### **Shear X**
- Applies horizontal shear transformation
- Input: Shear factor
- Slants image horizontally while keeping vertical lines straight
- Creates a "pushed" effect
- Example: factor=0.5 creates moderate horizontal slant

#### **Shear Y**
- Applies vertical shear transformation
- Input: Shear factor
- Slants image vertically while keeping horizontal lines straight
- Creates a "leaning" effect
- Example: factor=0.5 creates moderate vertical slant

---

###  Interpolation (Resolution Enhancement)

Methods for resizing images to new dimensions.

#### **Nearest Neighbor**
- Simplest and fastest interpolation method
- Input: New width and height in pixels
- Assigns each output pixel the value of nearest input pixel
- Best for: Pixel art, images requiring sharp edges
- Drawback: Can create blocky/pixelated results

#### **Bilinear**
- Smooth interpolation using 2×2 pixel neighborhood
- Input: New width and height in pixels
- Calculates weighted average of 4 nearest pixels
- Better quality than nearest neighbor
- Good balance of speed and quality

#### **Bicubic**
- High-quality interpolation using 4×4 pixel neighborhood
- Input: New width and height in pixels
- Uses cubic convolution for smooth results
- Best quality but slower processing
- Ideal for: Enlarging images, photography

---

###  Editing

#### **Crop**
- Extracts a rectangular region from the image
- Input: `x1, y1` (top-left corner), `x2, y2` (bottom-right corner)
- Coordinates start from (0,0) at top-left
- Validates that region is within image bounds
- Example: x1=50, y1=50, x2=200, y2=200 extracts a 150×150 region

---

###  Histogram

#### **Show Histogram**
- Displays distribution of pixel intensities (0-255)
- X-axis: Pixel intensity values
- Y-axis: Frequency (number of pixels)
- Opens in new window with matplotlib chart
- Useful for understanding image contrast and brightness

#### **Assess Histogram**
- Analyzes histogram quality with detailed metrics:
  - **Dynamic Range**: Measures intensity range usage (0-255)
  - **Contrast**: Based on standard deviation
  - **Balance**: How centered the distribution is
  - **Entropy**: Information content measure
  - **Overall Quality Score**: 0-100 rating
- Provides recommendations for improvement
- Explains whether histogram equalization would help

#### **Equalize Histogram**
- Enhances image contrast by redistributing intensities
- Uses cumulative distribution function (CDF)
- Makes histogram more uniform
- Benefits:
  - Improves contrast in low-contrast images
  - Reveals hidden details
  - Better utilizes full intensity range
- Best for: Underexposed or washed-out images

---

###  Filters

#### **Low-Pass Filters** (Smoothing/Blurring)

##### **Gaussian Filter (19×19, σ=3)**
- Smooths image using Gaussian kernel
- Reduces noise and detail
- Creates natural-looking blur
- Sigma (σ) controls blur amount
- Uses: Noise reduction, preprocessing for edge detection
- 19×19 kernel size provides strong smoothing

##### **Median Filter (7×7)**
- Replaces each pixel with median of surrounding pixels
- Excellent for removing salt-and-pepper noise
- Preserves edges better than Gaussian
- 7×7 window size balances noise removal and detail preservation
- Uses: Cleaning noisy images while keeping edges sharp

#### **High-Pass Filters** (Edge Detection)

##### **Laplacian Filter**
- Detects edges using second derivative
- Highlights regions of rapid intensity change
- Responds to edges in all directions
- Formula: Measures rate of rate of change
- Uses: Edge detection, image sharpening
- Output shows edges as bright lines on dark background

##### **Sobel Filter**
- Gradient-based edge detection
- Calculates horizontal and vertical gradients separately
- Combines them into gradient magnitude
- More robust to noise than simple gradient
- Emphasizes strong edges
- Uses: Edge detection, feature extraction

##### **Gradient Filter**
- Simplest edge detection using first derivatives
- Calculates difference between adjacent pixels
- Fast but sensitive to noise
- Shows rate of intensity change
- Uses: Quick edge detection, basic image analysis

---

###  Compression

All methods display: Original size, Compressed size, Compression ratio, Space saving percentage

#### **Huffman Coding**
- Variable-length encoding based on symbol frequency
- Most frequent pixels get shortest codes
- Lossless compression
- Optimal for data with non-uniform distribution
- **Best for**: Images with limited colors

#### **Golomb-Rice Coding**
- Specialized for geometrically distributed data
- Uses predictive coding (stores differences)
- Parameter m controls encoding
- Efficient for smooth images
- **Best for**: Images with gradual changes

#### **Arithmetic Coding**
- Encodes entire message into single floating-point number
- More efficient than Huffman for some data
- Uses probability ranges
- High compression ratio potential
- **Best for**: Images with known probability distributions

#### **LZW (Lempel-Ziv-Welch)**
- Dictionary-based compression
- Builds dictionary of repeating patterns
- Used in GIF format
- Adaptive algorithm (dictionary grows during compression)
- **Best for**: Images with repeating patterns

#### **RLE (Run-Length Encoding)**
- Encodes consecutive identical values as (value, count) pairs
- Very simple and fast
- Effective for images with large uniform areas
- Poor compression for complex images
- **Best for**: Simple graphics, logos, binary images

#### **Symbol-Based Coding**
- Creates dictionary of unique pixel values
- Assigns shorter codes to unique values
- Effective when image has few unique colors
- **Best for**: Images with limited color palette

#### **Bit-Plane Coding**
- Separates image into 8 bit planes
- Compresses each plane separately using RLE
- Higher bit planes contain more significant information
- Can achieve progressive transmission
- **Best for**: Images where some bit planes have patterns

#### **DCT (Discrete Cosine Transform)**
- Transforms image into frequency domain
- 8×8 block-based compression (JPEG-like)
- Quantizes coefficients to reduce data
- Lossy compression with quality control
- Removes high-frequency details
- **Best for**: Photographs, natural images

#### **Predictive Coding (DPCM)**
- Stores differences between predicted and actual values
- Predicts pixel from neighbors
- Uses Huffman on differences
- Exploits spatial correlation
- **Best for**: Images with spatial correlation

#### **Wavelet Coding**
- Multi-resolution analysis using Haar wavelets
- Decomposes image into multiple levels
- Thresholds small coefficients
- Better than DCT for some images
- **Best for**: Images with varying detail levels

#### **Compare All Methods**
- Runs all 10 compression methods on current image
- Displays comparison table with:
  - Compression ratios
  - Space savings
  - Relative performance
- Helps choose best method for your image type
- Processing time shown for each method

---

##  Technical Details

### Custom Implementation

All algorithms are implemented from scratch using basic operations:
- **Convolution**: Manual 2D convolution without built-in functions
- **Interpolation**: Custom pixel mapping calculations
- **Filters**: Hand-coded kernel operations
- **Transformations**: Matrix multiplication and inverse mapping
- **Compression**: Complete algorithm implementations

### Performance Considerations

- Processing time displayed for each operation
- Larger images take longer to process
- Complex operations (bicubic, wavelet) are slower
- Compression comparison can take several seconds

---

## 💡 Tips

1. **For best results**: Use high-quality source images
2. **Processing speed**: Smaller images process faster
3. **Binary conversion**: Try both auto and custom threshold to see differences
4. **Histogram**: Always assess before equalizing
5. **Compression**: Compare all methods to find best for your image type
6. **Filters**: Apply Gaussian before edge detection for better results

---

**⭐ If you found this project helpful, please consider giving it a star!**
