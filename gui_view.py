from gui_config import *
from tkinter import DoubleVar, font, ttk
import tkinter as tk
import gui_controller, gui_view_settings
from PIL import ImageTk, Image
from pathlib import Path
import os, random
#from ctypes import windll

class GuiView(object):
    """description of class"""

    def __init__(self, parent):
        self.dirname = os.path.dirname(__file__)
        self.window = parent # window
        self.window_width = self.window.winfo_width()
        self.window_height = self.window.winfo_height()
        self.progress = None
        self.settings_open = True
        self.create_widgets()
        self.setup_layout()
        self.settings_panel = gui_view_settings.GuiViewSettings(parent)

    def create_capture_bg(self):
        self.capture_bg_button = tk.Button(self.left_frame, text="Capture BG", 
            fg="white", bg=BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=BUTTON_FOCUS_COLOR,
            activeforeground="white")

    def create_quick_start(self):
        self.q_start_button = tk.Button(self.left_frame, text="Quick Start", 
            fg="white", bg=BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=BUTTON_FOCUS_COLOR,
            activeforeground="white")

    def create_stop(self):
        self.stop_button = tk.Button(self.left_frame, text="Stop", 
            fg="white", bg=STOP_BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=STOP_BUTTON_FOCUS_COLOR,
            activeforeground="white")

    def create_adv_options(self):
        self.adv_option = tk.Label(self.left_frame, text="Settings", fg="white",
            bg=WIN_BG_COLOR, font="Ubuntu 10", cursor="hand2")
        self.opt_font = font.Font(self.adv_option, self.adv_option.cget("font"))
        self.adv_option.configure(font=self.opt_font)

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

    def create_progress_bar(self):
        self.progress_style = ttk.Style()
        self.progress_style.theme_use('clam')
        TROUGH_COLOR = PROGRESS_BAR_BG
        BAR_COLOR = PROGRESS_BAR_COLOR
        self.progress_style.configure("bar.Horizontal.TProgressbar", troughcolor=TROUGH_COLOR,
            bordercolor=TROUGH_COLOR, background=BAR_COLOR,
            lightcolor=BAR_COLOR, darkcolor=BAR_COLOR)
        self.progress_style.layout('bar.Horizontal.TProgressbar', 
             [('Horizontal.Progressbar.trough',
               {'children': [('Horizontal.Progressbar.pbar',
                              {'side': 'left', 'sticky': 'ns'})],
                'sticky': 'nswe'}), 
              ('Horizontal.Progressbar.label', {'sticky': ''})])
        self.progress_var = DoubleVar()
        self.progress_var.set(0)
        self.progress = tk.ttk.Progressbar(self.right_frame, orient = "horizontal", variable=self.progress_var,
            length = WIN_MIN_WIDTH/2, mode = 'determinate', style="bar.Horizontal.TProgressbar")
        self.progress["maximum"] = 100

    def update_progress_bar(self, curr_step, max_step):
        print("Progress bar updated")
        progress_percent = int(curr_step / max_step * 100)
        self.progress_var.set(progress_percent)
        if (progress_percent > 50):
            self.progress_style.configure('bar.Horizontal.TProgressbar', foreground="white",
                    text='{:g} %'.format(int(curr_step / max_step * 100))) 
            if (progress_percent == 100):
                self.progress_style.configure('bar.Horizontal.TProgressbar', foreground="white",
                    text="Done!") 
        else:
            self.progress_style.configure('bar.Horizontal.TProgressbar', foreground=WIN_BG_COLOR,
                    text='{:g} %'.format(int(curr_step / max_step * 100))) 
        self.progress.update()

    def manage_settings(self, parent, action=None):
        if (action == True or (self.settings_open and action == None)): # open settings panel
            print("Settings panel open")
            self.settings_panel.setup_layout()
            self.settings_open = False
        elif (action == False or (not self.settings_open and action == None)):
            print("Settings panel close")
            self.settings_panel.forget_layout()
            self.settings_open = True

    def process_settings(self, parent):
        print("Settings panel close")
        self.settings_panel.forget_layout()
        self.settings_open = True 

    def create_widgets(self):    
        self.left_frame = tk.Frame(self.window, bg=LEFT_FRAME_COLOR,
            highlightthickness=0)
        self.right_frame = tk.Frame(self.window, bg=RIGHT_FRAME_COLOR,
            highlightthickness=0)
        self.create_app_name()
        self.create_logo()
        # self.q_start_button = tk.Button(self.left_frame, text="Quick Start",
        #     font="Ubuntu 12", relief="flat", bg = "#4070f5",
        #     fg="white", wraplength=5)
        self.create_quick_start()
        self.create_adv_options()
        self.create_progress_bar()
        self.create_capture_bg()
        self.create_stop()

    def setup_layout(self):
        self.left_frame.place(rely=0, relx=0, relheight=1, relwidth=WIN_SPLIT)
        self.right_frame.place(rely=0, relx=WIN_SPLIT, relheight=1, relwidth=1-WIN_SPLIT)
        self.intro_text.place(anchor="center", relx=0.5, rely=0.1)
        self.logo_label.place(anchor="center", relx=0.5, rely=0.5)
        self.q_start_button.place(anchor="center", relx=0.5, rely=0.7)
        self.capture_bg_button.place(anchor="center", relx=0.5, rely=0.8)
        self.adv_option.place(anchor="center", relx=0.5, rely=0.87)
        self.progress.place(anchor="center", relx=0.5, rely=0.87)

    def place_q_start(self):
        self.q_start_button.place(anchor="center", relx=0.5, rely=0.7)

    def forget_q_start(self):
        self.q_start_button.place_forget()

    def place_stop(self):
        self.stop_button.place(anchor="center", relx=0.5, rely=0.7)

    def forget_stop(self):
        self.stop_button.place_forget()


