import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from gui.styles import AppStyles
from utils.helpers import numpy_to_pil


class ImageDisplay:
    def __init__(self, parent, title="Image"):
        self.frame = tk.Frame(parent, bg=AppStyles.BG_DARK)
        self.current_image = None
        self.photo = None
        
        tk.Label(self.frame, text=title, font=AppStyles.FONT_TITLE,
                fg=AppStyles.FG_TEXT, bg=AppStyles.BG_DARK).pack(pady=5)
        
        self.canvas = tk.Canvas(self.frame, width=500, height=500, 
                               bg=AppStyles.BG_LIGHT)
        self.canvas.pack(padx=10, pady=10)
        
        self.info = tk.Label(self.frame, text="", font=AppStyles.FONT_NORMAL,
                            fg=AppStyles.FG_TEXT, bg=AppStyles.BG_DARK)
        self.info.pack()
    
    def display(self, image, info=""):
        if isinstance(image, np.ndarray):
            image = numpy_to_pil(image)
        
        self.current_image = image
        img = image.copy()
        img.thumbnail((480, 480), Image.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(250, 250, image=self.photo)
        self.info.config(text=info)
    
    def clear(self):
        self.canvas.delete("all")
        self.info.config(text="")
        self.current_image = None