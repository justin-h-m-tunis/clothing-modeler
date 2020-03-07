from gui_config import *
import gui_model
import gui_view
from pubsub import pub
from ctypes import windll
import tkinter as tk
from tkinter import messagebox

class GuiController(object):
    """GUI controller that handles model and view"""

    def __init__(self, parent):
        self.init_model()
        self.init_menu()
        self.init_view(parent)

    def init_model(self):
        self.model = gui_model.GuiModel()

    def init_view(self, parent):
        self.view = gui_view.GuiView(parent)
        self.bind_intro_text()
        
    def init_menu(self):
        self.menubar = tk.Menu(window)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="New", command=self.do_nothing)
        self.file_menu.add_command(label="Open", command=self.do_nothing)
        self.file_menu.add_command(label="Save", command=self.do_nothing)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.help_menu.add_command(label="Help Index", command=self.do_nothing)
        self.help_menu.add_command(label="About...", command=self.do_nothing)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        window.config(menu=self.menubar)

    def bind_intro_text(self):
        self.view.intro_text.bind("<Configure>", self.scale_font)

    '''Actions on key binding'''
    def do_nothing(self):
        pass

    def scale_font(self, event):
        if (window.winfo_width() < 900):
            self.view.intro_text.config(font="DengXian 16")
        else:
            self.view.intro_text.config(font="DengXian 20")

def on_closing():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

if __name__ == "__main__":
    windll.shcore.SetProcessDpiAwareness(1)
    window = tk.Tk()
    window.geometry("%sx%s" % (WIN_WIDTH, WIN_HEIGHT))
    window.title("Clothing Auto-Modeler")
    # window.iconbitmap(r"C:\Users\shuyo\OneDrive\Scripts\chinese_book\images\Papirus-Team-Papirus-Places-Folder-blue-download.ico")
    window.configure(background = WIN_BG_COLOR)
    window.minsize(WIN_MIN_WIDTH, WIN_MIN_HEIGHT)
    view = GuiController(window)
    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()
