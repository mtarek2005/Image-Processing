import sys
import tkinter as tk
from tkinter import messagebox
from gui.main_window import MainWindow

def main():
    try:
        root = tk.Tk()
        app = MainWindow(root)
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()