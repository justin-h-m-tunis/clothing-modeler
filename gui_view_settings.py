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
        self.config_setting_frame_row_col()
        self.config_setting_title_frame_row_col()

    def create_title(self):
        self.title_text = tk.Label(self.settings_title_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 16", state="normal", 
            relief="flat", text="Settings")
        
    def create_motor_title(self):
        self.title_text = tk.Label(self.settings_frame, bg="blue", fg=WIN_BG_COLOR,
            font="Ubuntu 20", state="normal", height=1, width=12,
            relief="flat", text="Settings")
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(0, weight=1)

    def create_camera_title(self):
        self.title_text = tk.Label(self.settings_frame, bg="blue", fg=WIN_BG_COLOR,
            font="Ubuntu 20", state="normal", height=1, width=12,
            relief="flat", text="Settings")
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(0, weight=1)

    def create_thres_title(self):
        self.title_text = tk.Label(self.settings_frame, bg="blue", fg=WIN_BG_COLOR,
            font="Ubuntu 20", state="normal", height=1, width=12,
            relief="flat", text="Settings")
        # self.title_text.tag_configure("center", justify='center')
        # self.title_text.insert("1.0", "Settings")
        # self.title_text.tag_add("center", "1.0", "end")
        # self.title_text.config(state="disabled")
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(0, weight=1)

    def create_widgets(self):    
        self.settings_frame = tk.Frame(self.window, bg=RIGHT_FRAME_COLOR)
        self.settings_title_frame = tk.Frame(self.settings_frame, bg="yellow")
        self.motor_frame = tk.Frame(self.settings_frame, bg="red")
        self.camera_frame = tk.Frame(self.settings_frame, bg="green")
        self.thres_frame = tk.Frame(self.settings_frame, bg="blue")
        self.create_title()

    def setup_layout(self):
        # setup frames
        self.settings_frame.place(rely=0, relx=WIN_SPLIT, relheight=1, relwidth=1-WIN_SPLIT)
        self.settings_title_frame.grid(row=0, column=0, columnspan=2, sticky="snwe")
        self.motor_frame.grid(row=1, column=0, sticky="snwe")
        self.camera_frame.grid(row=1, column=1, sticky="snwe")
        self.thres_frame.grid(row=2, column=0, columnspan=2, sticky="snwe")

        self.title_text.grid(row=0, column=0, pady=0, ipady=0)

    def config_setting_frame_row_col(self):
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_columnconfigure(1, weight=1)

        self.settings_frame.grid_rowconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(1, weight=60)
        self.settings_frame.grid_rowconfigure(2, weight=90)
    
    def config_setting_title_frame_row_col(self):
        self.settings_title_frame.grid_columnconfigure(0, weight=1)
        self.settings_title_frame.grid_rowconfigure(0, weight=1)

    def forget_layout(self):
        self.settings_frame.place_forget()

