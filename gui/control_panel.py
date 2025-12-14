# Replace the entire ControlPanel class in gui/control_panel.py with this:

import tkinter as tk
from tkinter import simpledialog
from gui.styles import AppStyles


class ControlPanel:
    def __init__(self, parent, callbacks):
        self.callbacks = callbacks
        
        # Main frame
        self.frame = tk.Frame(parent, bg=AppStyles.BG_DARK)
        
        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self.frame, bg=AppStyles.BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        
        # Scrollable frame inside canvas
        self.scroll_frame = tk.Frame(self.canvas, bg=AppStyles.BG_DARK)
        
        # Configure scrolling
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self._create_buttons()
    
    def _on_mousewheel(self, event):
        """Enable mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _btn(self, text, cmd, section=None):
        if section:
            tk.Label(self.scroll_frame, text=section, font=AppStyles.FONT_BOLD,
                    fg=AppStyles.ACCENT, bg=AppStyles.BG_DARK).pack(fill="x", pady=(10,5), padx=10)
        
        btn = tk.Button(self.scroll_frame, text=text, command=cmd,
                       bg=AppStyles.ACCENT, fg="white", font=AppStyles.FONT_NORMAL,
                       width=20)
        btn.pack(fill="x", padx=10, pady=2)
    
    def _create_buttons(self):
        # File Operations
        self._btn("Upload Image", self.callbacks['upload'], "📁 File")
        self._btn("Save Image", self.callbacks['save'])
        self._btn("Reset", self.callbacks['reset'])
        
        # Conversions
        self._btn("Grayscale", self.callbacks['grayscale'], "🎨 Conversions")
        self._btn("Binary (Auto)", self.callbacks['binary_auto'])
        self._btn("Binary (Custom)", self.callbacks['binary_custom'])
        
        # Transformations
        self._btn("Translate", self.callbacks['translate'], "🔄 Transforms")
        self._btn("Scale", self.callbacks['scale'])
        self._btn("Rotate", self.callbacks['rotate'])
        self._btn("Shear X", self.callbacks['shear_x'])
        self._btn("Shear Y", self.callbacks['shear_y'])
        
        # Interpolation
        self._btn("Nearest Neighbor", self.callbacks['nearest'], "📐 Interpolation")
        self._btn("Bilinear", self.callbacks['bilinear'])
        self._btn("Bicubic", self.callbacks['bicubic'])
        
        # Editing
        self._btn("Crop", self.callbacks['crop'], "✂️ Edit")
        
        # Histogram
        self._btn("Show Histogram", self.callbacks['show_hist'], "📊 Histogram")
        self._btn("Assess Histogram", self.callbacks['assess_hist'])
        self._btn("Equalize Histogram", self.callbacks['equalize_hist'])
        
        # Filters
        self._btn("Gaussian Filter", self.callbacks['gaussian'], "🔍 Filters")
        self._btn("Median Filter", self.callbacks['median'])
        self._btn("Laplacian Filter", self.callbacks['laplacian'])
        self._btn("Sobel Filter", self.callbacks['sobel'])
        self._btn("Gradient Filter", self.callbacks['gradient'])
        
        # Compression
        self._btn("Huffman", lambda: self.callbacks['compress']('huffman'), "🗜️ Compression")
        self._btn("Golomb-Rice", lambda: self.callbacks['compress']('golomb_rice'))
        self._btn("Arithmetic", lambda: self.callbacks['compress']('arithmetic'))
        self._btn("LZW", lambda: self.callbacks['compress']('lzw'))
        self._btn("RLE", lambda: self.callbacks['compress']('rle'))
        self._btn("Symbol-Based", lambda: self.callbacks['compress']('symbol'))
        self._btn("Bit-Plane", lambda: self.callbacks['compress']('bitplane'))
        self._btn("DCT", lambda: self.callbacks['compress']('dct'))
        self._btn("Predictive", lambda: self.callbacks['compress']('predictive'))
        self._btn("Wavelet", lambda: self.callbacks['compress']('wavelet'))
        self._btn("Compare All", self.callbacks['compare_compress'])
        
        # Add some bottom padding so last button is visible
        tk.Frame(self.scroll_frame, height=20, bg=AppStyles.BG_DARK).pack()


def ask_params(parent, title, params):
    """Simple parameter dialog."""
    result = {}
    for name, default, desc in params:
        val = simpledialog.askstring(title, f"{name} ({desc}):", 
                                     initialvalue=str(default), parent=parent)
        if val is None:
            return None
        try:
            result[name] = float(val) if '.' in val else int(val)
        except:
            result[name] = val
    return result