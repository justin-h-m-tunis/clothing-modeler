from gui_config import *
import gui_model
import gui_view
from pubsub import pub
from ctypes import windll
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import open3d as o3d
import threading
import multiprocessing, pickle

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
        self.bind_q_start()
        self.bind_adv_option()
        
    def init_menu(self):
        self.menubar = tk.Menu(window)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="New", command=self.do_nothing)
        # self.file_menu.add_command(label="Open", command=self.do_nothing)
        # self.file_menu.add_command(label="Save", command=self.do_nothing)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=on_closing)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        self.tool_menu = tk.Menu(self.menubar, tearoff=0)
        self.tool_menu.add_command(label="View Model", command=self.open_ply)
        self.menubar.add_cascade(label="Tools", menu=self.tool_menu)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.help_menu.add_command(label="Help Index", command=self.do_nothing)
        self.help_menu.add_command(label="About...", command=self.do_nothing)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        window.config(menu=self.menubar)

    def bind_intro_text(self):
        self.view.intro_text.bind("<Configure>", self.scale_font)

    def bind_q_start(self):
        self.view.q_start_button.bind("<Enter>",
            lambda e: self.view.q_start_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.q_start_button.bind("<Leave>",
            lambda e: self.view.q_start_button.configure(bg = BUTTON_COLOR))
        self.view.q_start_button.bind("<ButtonRelease-1>", self.run_system)

    def bind_adv_option(self):
        self.view.adv_option.bind("<Enter>",
            lambda e: self.view.opt_font.configure(underline = True))
        self.view.adv_option.bind("<Leave>",
            lambda e: self.view.opt_font.configure(underline = False))
        self.view.adv_option.bind("<ButtonRelease-1>", self.run_adv_option)

    '''Actions on key binding'''
    def do_nothing(self):
        pass

    '''Main logic execution'''
    def run_system(self, event):
        print("3D scanning system start with default settings")
        self.model.blink_led()

    '''Place holder for advanced options'''
    def run_adv_option(self, event):
        print("Opening advanced menu")

    def scale_font(self, event):
        if (window.winfo_width() < 900):
            self.view.intro_text.config(font="Ubuntu 16")
        else:
            self.view.intro_text.config(font="Ubuntu 20")

    def open_ply(self):
        self.path_name = filedialog.askopenfilename(initialdir = "/", title = "Select ply file", filetypes = (("ply files","*.ply"),))
        print(self.path_name)
        if (self.path_name == ""):
            return
        
        self.p = multiprocessing.Process(target=gui_model.view_ply, args=(self.path_name,))
        self.p.start()

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
