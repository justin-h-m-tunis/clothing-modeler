import os, gui_controller
import tkinter as tk
from PIL import ImageTk, Image
from pathlib import Path
from gui_config import *

class GuiViewSettings(object):

    def __init__(self, parent):
        self.dirname = os.path.dirname(__file__)
        self.window = parent
        self.window_width = self.window.winfo_width()
        self.window_height = self.window.winfo_height()
        self.create_widgets()
        # self.setup_layout()

    def create_title(self):
        self.title_text = tk.Text(self.settings_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 20", state="normal", wrap="word", height=2, width=12,
            relief="flat")
        self.title_text.tag_configure("center", justify='center')
        self.title_text.insert("1.0", "Settings")
        self.title_text.tag_add("center", "1.0", "end")
        self.title_text.config(state="disabled")

    def create_widgets(self):    
        self.settings_frame = tk.Frame(self.window, bg=RIGHT_FRAME_COLOR)
        self.create_title()

    def setup_layout(self):
        self.settings_frame.place(rely=0, relx=WIN_SPLIT, relheight=1, relwidth=1-WIN_SPLIT)
        self.title_text.place(anchor="center", relx=0.2, rely=0.1)

    def forget_layout(self):
        self.settings_frame.place_forget()

