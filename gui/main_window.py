import tkinter as tk
from tkinter import filedialog, messagebox
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.styles import AppStyles
from gui.image_display import ImageDisplay
from gui.control_panel import ControlPanel, ask_params

from core.image_loader import ImageLoader
from core.conversions import ImageConverter
from core.transformations import AffineTransformations
from core.interpolation import Interpolator
from core.histogram import HistogramProcessor
from core.filters import FilterOperations
from core.compression import CompressionEngine


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Suite")
        self.root.geometry(f"{AppStyles.WINDOW_WIDTH}x{AppStyles.WINDOW_HEIGHT}")
        self.root.configure(bg=AppStyles.BG_DARK)
        
        self.loader = ImageLoader()
        self.original = None
        self.current = None
        
        self._setup_ui()
        self._setup_callbacks()
    
    def _setup_ui(self):
        # Main layout
        main = tk.Frame(self.root, bg=AppStyles.BG_DARK)
        main.pack(fill="both", expand=True)
        
        # Control panel
        ctrl_frame = tk.Frame(main, bg=AppStyles.BG_DARK, width=250)
        ctrl_frame.pack(side="left", fill="y")
        ctrl_frame.pack_propagate(False)
        
        # Images
        self.orig_display = ImageDisplay(main, "Original")
        self.orig_display.frame.pack(side="left", fill="both", expand=True)
        
        self.proc_display = ImageDisplay(main, "Processed")
        self.proc_display.frame.pack(side="left", fill="both", expand=True)
        
        # Status bar
        self.status = tk.Label(self.root, text="Ready", font=AppStyles.FONT_NORMAL,
                              fg=AppStyles.FG_TEXT, bg=AppStyles.BG_LIGHT, anchor="w")
        self.status.pack(side="bottom", fill="x")
        
        self.ctrl_frame_ref = ctrl_frame
    
    def _setup_callbacks(self):
        callbacks = {
            'upload': self.upload,
            'save': self.save,
            'reset': self.reset,
            'grayscale': lambda: self._process(ImageConverter.rgb_to_grayscale, self.current),
            'binary_auto': self.binary_auto,
            'binary_custom': self.binary_custom,
            'translate': self.translate,
            'scale': self.scale,
            'rotate': self.rotate,
            'shear_x': self.shear_x,
            'shear_y': self.shear_y,
            'nearest': self.nearest,
            'bilinear': self.bilinear,
            'bicubic': self.bicubic,
            'crop': self.crop,
            'show_hist': self.show_hist,
            'assess_hist': self.assess_hist,
            'equalize_hist': lambda: self._process(HistogramProcessor.histogram_equalization, self.current),
            'gaussian': lambda: self._process(FilterOperations.gaussian_filter, self.current, 19, 3),
            'median': lambda: self._process(FilterOperations.median_filter, self.current, 7),
            'laplacian': lambda: self._process(FilterOperations.laplacian_filter, self.current),
            'sobel': lambda: self._process(FilterOperations.sobel_filter, self.current),
            'gradient': lambda: self._process(FilterOperations.gradient_filter, self.current),
            'compress': self.compress,
            'compare_compress': self.compare_compress
        }
        
        self.panel = ControlPanel(self.ctrl_frame_ref, callbacks)
        self.panel.frame.pack(fill="both", expand=True)
    
    def _process(self, func, *args):
        if self.current is None:
            messagebox.showwarning("Warning", "Upload image first")
            return
        
        try:
            start = time.time()
            result = func(*args)
            elapsed = time.time() - start
            
            self.current = result
            self.proc_display.display(result, f"Time: {elapsed:.3f}s")
            self.status.config(text=f"Done in {elapsed:.3f}s")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def upload(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if not path:
            return
        
        img, msg = self.loader.load_image(path)
        if img is None:
            messagebox.showerror("Error", msg)
            return
        
        self.original = img
        self.current = np.copy(img)
        info = self.loader.get_all_info()
        self.orig_display.display(img, f"{info['resolution']} | {info['size']}")
        self.proc_display.clear()
        self.status.config(text=f"Loaded: {info['resolution']}")
    
    def save(self):
        if self.proc_display.current_image is None:
            messagebox.showwarning("Warning", "No processed image")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self.proc_display.current_image.save(path)
            messagebox.showinfo("Success", "Saved!")
    
    def reset(self):
        if self.original is not None:
            self.current = np.copy(self.original)
            self.proc_display.clear()
            self.status.config(text="Reset to original")
    
    def binary_auto(self):
        if self.current is None:
            messagebox.showwarning("Warning", "Upload image first")
            return
        
        result = ImageConverter.grayscale_to_binary(self.current, None)
        thresh, optimal, exp = ImageConverter.calculate_optimal_threshold(self.current)
        self.current = result
        self.proc_display.display(result, f"Binary (thresh={thresh:.1f})")
        messagebox.showinfo("Threshold", exp)
    
    def binary_custom(self):
        params = ask_params(self.root, "Binary", [("Threshold", 128, "0-255")])
        if params:
            self._process(ImageConverter.grayscale_to_binary, self.current, params['Threshold'])
    
    def translate(self):
        p = ask_params(self.root, "Translate", [("tx", 50, "X"), ("ty", 50, "Y")])
        if p:
            self._process(AffineTransformations.translate, self.current, p['tx'], p['ty'])
    
    def scale(self):
        p = ask_params(self.root, "Scale", [("sx", 1.5, "X factor"), ("sy", 1.5, "Y factor")])
        if p:
            self._process(AffineTransformations.scale, self.current, p['sx'], p['sy'])
    
    def rotate(self):
        p = ask_params(self.root, "Rotate", [("angle", 45, "degrees")])
        if p:
            self._process(AffineTransformations.rotate, self.current, p['angle'])
    
    def shear_x(self):
        p = ask_params(self.root, "Shear X", [("factor", 0.5, "factor")])
        if p:
            self._process(AffineTransformations.shear_x, self.current, p['factor'])
    
    def shear_y(self):
        p = ask_params(self.root, "Shear Y", [("factor", 0.5, "factor")])
        if p:
            self._process(AffineTransformations.shear_y, self.current, p['factor'])
    
    def nearest(self):
        p = ask_params(self.root, "Resize", [("width", 800, "px"), ("height", 600, "px")])
        if p:
            self._process(Interpolator.nearest_neighbor, self.current, (p['width'], p['height']))
    
    def bilinear(self):
        p = ask_params(self.root, "Resize", [("width", 800, "px"), ("height", 600, "px")])
        if p:
            self._process(Interpolator.bilinear, self.current, (p['width'], p['height']))
    
    def bicubic(self):
        p = ask_params(self.root, "Resize", [("width", 800, "px"), ("height", 600, "px")])
        if p:
            self._process(Interpolator.bicubic, self.current, (p['width'], p['height']))
    
    def crop(self):
        p = ask_params(self.root, "Crop", [("x1", 50, ""), ("y1", 50, ""), ("x2", 200, ""), ("y2", 200, "")])
        if p:
            self._process(AffineTransformations.crop, self.current, p['x1'], p['y1'], p['x2'], p['y2'])
    
    def show_hist(self):
        if self.current is None:
            return
        
        hist = HistogramProcessor.compute_histogram(self.current)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(256), hist, color='blue', alpha=0.7)
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram')
        
        win = tk.Toplevel(self.root)
        win.title("Histogram")
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack()
        plt.close(fig)
    
    def assess_hist(self):
        if self.current is None:
            return
        score, good, exp = HistogramProcessor.assess_histogram_quality(self.current)
        messagebox.showinfo("Histogram Assessment", exp)
    
    def compress(self, method):
        if self.current is None:
            return
        
        methods = {
            'huffman': CompressionEngine.huffman_encode,
            'golomb_rice': CompressionEngine.golomb_rice_encode,
            'arithmetic': CompressionEngine.arithmetic_encode,
            'lzw': CompressionEngine.lzw_encode,
            'rle': CompressionEngine.rle_encode,
            'symbol': CompressionEngine.symbol_based_encode,
            'bitplane': CompressionEngine.bitplane_encode,
            'dct': CompressionEngine.dct_encode,
            'predictive': CompressionEngine.predictive_encode,
            'wavelet': CompressionEngine.wavelet_encode
        }
        
        start = time.time()
        result = methods[method](self.current)
        elapsed = time.time() - start
        
        msg = f"""{result['method']}

Original: {result['original_size_bits']} bits
Compressed: {result['compressed_size_bits']} bits
Ratio: {result['compression_ratio']:.2f}:1
Savings: {result['space_saving_percent']:.2f}%
Time: {elapsed:.3f}s"""
        
        messagebox.showinfo("Compression", msg)
    
    def compare_compress(self):
        if self.current is None:
            return
        
        results = CompressionEngine.compare_all_methods(self.current)
        
        win = tk.Toplevel(self.root)
        win.title("Compression Comparison")
        win.geometry("700x500")
        
        text = tk.Text(win, font=AppStyles.FONT_NORMAL, bg=AppStyles.BG_LIGHT, fg=AppStyles.FG_TEXT)
        text.pack(fill="both", expand=True, padx=10, pady=10)
        
        text.insert("end", "COMPRESSION COMPARISON\n" + "="*70 + "\n\n")
        for key, r in results.items():
            text.insert("end", f"{r['method']}\n")
            text.insert("end", f"Ratio: {r['compression_ratio']:.2f}:1 | Savings: {r['space_saving_percent']:.2f}%\n\n")