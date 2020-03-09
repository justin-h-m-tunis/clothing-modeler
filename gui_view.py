from gui_config import *
import tkinter as tk
import gui_controller
from pubsub import pub
from PIL import ImageTk, Image
from pathlib import Path
import os
#from ctypes import windll

class GuiView(object):
    """description of class"""

    def __init__(self, parent):
        self.dirname = os.path.dirname(__file__)
        self.window = parent # window
        self.window_width = self.window.winfo_width()
        self.window_height = self.window.winfo_height()
        self.create_widgets()
        self.setup_layout()

    def create_quick_start(self):
        self.q_start_button = tk.Button(self.left_frame, text="Quick Start", 
            fg="white", bg=BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=BUTTON_FOCUS_COLOR,
            activeforeground="white")

    def create_app_name(self):
        self.intro_text = tk.Text(self.left_frame, fg="white", bg=WIN_BG_COLOR,
            font="Ubuntu 20", state="normal", wrap="word", height=2, width=12,
            relief="flat")
        self.intro_text.tag_configure("center", justify='center')
        self.intro_text.insert("1.0", APP_NAME)
        self.intro_text.tag_add("center", "1.0", "end")
        self.intro_text.config(state="disabled")

    def create_logo(self):
        data_folder = Path(LOGO_PATH)
        path = self.dirname / data_folder
        self.logo_img = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.logo_label = tk.Label(self.right_frame, image = self.logo_img, bg=RIGHT_FRAME_COLOR)

    def create_widgets(self):    
        self.left_frame = tk.Frame(self.window, bg=LEFT_FRAME_COLOR)
        self.right_frame = tk.Frame(self.window, bg=RIGHT_FRAME_COLOR)
        self.create_app_name()
        self.create_logo()
        # self.q_start_button = tk.Button(self.left_frame, text="Quick Start",
        #     font="Ubuntu 12", relief="flat", bg = "#4070f5",
        #     fg="white", wraplength=5)
        self.create_quick_start()

    def setup_layout(self):
        self.left_frame.place(rely=0, relx=0, relheight=1, relwidth=WIN_SPLIT)
        self.right_frame.place(rely=0, relx=WIN_SPLIT, relheight=1, relwidth=1-WIN_SPLIT)
        self.intro_text.place(anchor="center", relx=0.5, rely=0.1)
        self.logo_label.place(anchor="center", relx=0.5, rely=0.5)
        self.q_start_button.place(anchor="center", relx=0.5, rely=0.8)


