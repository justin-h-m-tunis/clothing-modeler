import os, gui_controller
import tkinter as tk
from PIL import ImageTk, Image
from pathlib import Path
from gui_config import *
# from ttkwidgets import TickScale
from tkinter import ttk, font

class GuiViewSettings(object):

    def __init__(self, parent):
        self.dirname = os.path.dirname(__file__)
        self.window = parent
        self.window_width = self.window.winfo_width()
        self.window_height = self.window.winfo_height()
        self.create_widgets()
        self.config_setting_frame_row_col()
        self.config_setting_title_frame_row_col()
        self.config_motor_frame_row_col()
        self.config_camera_frame_row_col()
        self.config_thres_frame_row_col()

    def create_title(self):
        self.title_text = tk.Label(self.settings_title_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 16", state="normal", relief="flat", text="Settings")


    '''Motor frame implementations'''

    def create_motor_title(self):
        self.motor_title_text = tk.Label(self.motor_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 14 underline", state="normal", relief="flat", text="Motor")
    
    def create_param_speed_slider(self):
        self.param_speed_slider = tk.Scale(self.motor_frame, relief="flat", sliderrelief="flat",
            orient="horizontal", bg="#FFFFFF", bd = 1, showvalue=1, highlightthickness=0, 
            label="Speed", from_=0, to=1, digits=2, resolution=0.1, font="Ubuntu 12")
        self.param_speed_slider.set(0.5)
        # style = ttk.Style()
        # style.theme_use('clam')
        # style.configure('my.Horizontal.TScale', sliderlength=50, background='#FFFFFF',
        #                 foreground=WIN_BG_COLOR)
        # self.param_speed_slider = TickScale(self.motor_frame, orient='horizontal', style='my.Horizontal.TScale',
        #     tickinterval=0.2, from_=0, to=1, showvalue=True, digits=2,
        #     length=100, labelpos='e')

    def create_motor_adv(self):
        self.motor_adv = tk.Label(self.motor_frame, text="Advanced", fg=WIN_BG_COLOR,
            bg="white", font="Ubuntu 12", cursor="hand2")
        self.motor_adv_font = font.Font(self.motor_adv, self.motor_adv.cget("font"))
        self.motor_adv.configure(font=self.motor_adv_font)

    def create_test_spin_button(self):
        self.test_spin_button = tk.Button(self.motor_frame, text="Test Spin", 
            fg="white", bg=BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=BUTTON_FOCUS_COLOR,
            activeforeground="white")




    '''Camera frame implementations'''

    def create_camera_title(self):
        self.camera_title_text = tk.Label(self.camera_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 14 underline", state="normal", relief="flat", text="Camera")

    def create_param_distance_label(self):
        self.param_distance_label = tk.Label(self.camera_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", relief="flat", text="Distance")

    def create_param_distance_entry(self):
        self.param_distance_entry = tk.Entry(self.camera_frame, fg=WIN_BG_COLOR,
            font="Ubuntu 14", state="normal", bg=ENTRY_COLOR, width=11)

    def create_camera_adv(self):
        self.camera_adv = tk.Label(self.camera_frame, text="Advanced", fg=WIN_BG_COLOR,
            bg="white", font="Ubuntu 12", cursor="hand2")
        self.camera_adv_font = font.Font(self.camera_adv, self.camera_adv.cget("font"))
        self.camera_adv.configure(font=self.camera_adv_font)

    def create_test_picture_button(self):
        self.test_picture_button = tk.Button(self.camera_frame, text="Test Picture", 
            fg="white", bg=BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=BUTTON_FOCUS_COLOR,
            activeforeground="white")




    '''Thresholding frame implementations'''

    def create_thres_title(self):
        self.thres_title_text = tk.Label(self.thres_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 14 underline", state="normal", relief="flat", text="Thresholding")

    def create_widgets(self):    
        # create frames
        self.settings_frame = tk.Frame(self.window, bg=RIGHT_FRAME_COLOR)
        self.settings_title_frame = tk.Frame(self.settings_frame, bg="white")
        self.motor_frame = tk.Frame(self.settings_frame, bg="white",
            highlightthickness=2, highlightcolor=FRAME_BORDER_COLOR)
        self.camera_frame = tk.Frame(self.settings_frame, bg="white",
            highlightthickness=2, highlightcolor=FRAME_BORDER_COLOR)
        self.thres_frame = tk.Frame(self.settings_frame, bg="white",
            highlightthickness=2, highlightcolor=FRAME_BORDER_COLOR)
        self.create_title()
        self.create_motor_title()
        self.create_camera_title()
        self.create_thres_title()

        # create motor frame widgets
        self.create_param_speed_slider()
        self.create_test_spin_button()
        self.create_motor_adv()

        # create camera frame widgets
        self.create_param_distance_label()
        self.create_param_distance_entry()
        self.create_camera_adv()
        self.create_test_picture_button()

    def setup_layout(self):
        # setup frames
        self.settings_frame.place(rely=0, relx=WIN_SPLIT, relheight=1, relwidth=1-WIN_SPLIT)
        self.settings_title_frame.grid(row=0, column=0, columnspan=2, sticky="snwe")
        self.motor_frame.grid(row=1, column=0, sticky="snwe")
        self.camera_frame.grid(row=1, column=1, sticky="snwe")
        self.thres_frame.grid(row=2, column=0, columnspan=2, sticky="snwe")

        # setup subframes
        self.title_text.grid(row=0, column=0, pady=0, ipady=0)
        self.motor_title_text.grid(row=0, column=0, columnspan=4, pady=0, ipady=0)
        self.camera_title_text.grid(row=0, column=0, columnspan=4, pady=0, ipady=0)
        self.thres_title_text.grid(row=0, column=0, pady=0, ipady=0)

        # setup widgets in motor frame
        self.param_speed_slider.grid(row=1, column=0, columnspan=4, padx=22, pady=0, ipady=0, sticky="we")
        self.test_spin_button.grid(row=2, column=2, columnspan=2, padx=10, pady=0, ipady=0, sticky="")
        self.motor_adv.grid(row=2, column=0, columnspan=2, padx=10, pady=0, ipady=0, sticky="")

        # setup widgets in camera frame
        self.param_distance_label.grid(row=1, column=0, columnspan=2, pady=0, padx=0)
        self.param_distance_entry.grid(row=1, column=2, columnspan=2, pady=0, padx=0)
        self.test_picture_button.grid(row=2, column=2, columnspan=2, padx=10, pady=0, ipady=0, sticky="")
        self.camera_adv.grid(row=2, column=0, columnspan=2, padx=10, pady=0, ipady=0, sticky="")


    def config_setting_frame_row_col(self):
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_columnconfigure(1, weight=1)

        self.settings_frame.grid_rowconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(1, weight=10)
        self.settings_frame.grid_rowconfigure(2, weight=100)
    
    def config_setting_title_frame_row_col(self):
        self.settings_title_frame.grid_columnconfigure(0, weight=1)
        self.settings_title_frame.grid_rowconfigure(0, weight=1)

    def config_motor_frame_row_col(self):
        self.motor_frame.grid_columnconfigure(0, weight=1)
        self.motor_frame.grid_columnconfigure(1, weight=1)
        self.motor_frame.grid_columnconfigure(2, weight=1)
        self.motor_frame.grid_columnconfigure(3, weight=1)
        self.motor_frame.grid_rowconfigure(0, weight=1)
        self.motor_frame.grid_rowconfigure(1, weight=100)
        self.motor_frame.grid_rowconfigure(2, weight=1000)

    def config_camera_frame_row_col(self):
        self.camera_frame.grid_columnconfigure(0, weight=1)
        self.camera_frame.grid_columnconfigure(1, weight=1)
        self.camera_frame.grid_columnconfigure(2, weight=1)
        self.camera_frame.grid_columnconfigure(3, weight=1)
        self.camera_frame.grid_rowconfigure(0, weight=1)
        self.camera_frame.grid_rowconfigure(1, weight=1000)
        self.camera_frame.grid_rowconfigure(2, weight=1000)

    def config_thres_frame_row_col(self):
        self.thres_frame.grid_columnconfigure(0, weight=1)
        self.thres_frame.grid_rowconfigure(0, weight=1)
        self.thres_frame.grid_rowconfigure(1, weight=100)
        self.thres_frame.grid_rowconfigure(2, weight=100)
        self.thres_frame.grid_rowconfigure(3, weight=100)

    def forget_layout(self):
        self.settings_frame.place_forget()

