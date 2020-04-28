import os, gui_controller
import tkinter as tk
from PIL import ImageTk, Image
from pathlib import Path
from gui_config import *
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
        self.config_setting_action_frame_row_col()
        self.config_thres_preview_frame_row_col()
        self.config_thres_adv_frame_row_col()

    def create_title(self):
        self.title_text = tk.Label(self.settings_title_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 16", state="normal", relief="flat", text="Settings")
        return self.title_text

    def create_settings_apply_button(self):
        self.settings_apply_button = tk.Button(self.settings_action_frame, text="Apply", 
            fg="white", bg=BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=BUTTON_FOCUS_COLOR,
            activeforeground="white")
        return self.settings_apply_button

    def create_settings_cancel_button(self):
        self.settings_cancel_button = tk.Button(self.settings_action_frame, text="Cancel", 
            fg=WIN_BG_COLOR, bg=CANCEL_BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=CANCEL_BUTTON_FOCUS_COLOR,
            activeforeground=WIN_BG_COLOR)
        return self.settings_cancel_button


    '''Motor frame implementations'''

    def create_motor_title(self):
        self.motor_title_text = tk.Label(self.motor_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 14 underline", state="normal", relief="flat", text="Motor")
        return self.motor_title_text
    
    def create_param_speed_slider(self):
        self.param_speed_slider = tk.Scale(self.motor_frame, relief="flat", sliderrelief="flat",
            orient="horizontal", bg="#FFFFFF", bd = 1, showvalue=1, highlightthickness=0, 
            label="Speed", from_=0, to=1, digits=2, resolution=0.1, font="Ubuntu 12")
        self.param_speed_slider.set(0.5)
        return self.param_speed_slider

    def create_motor_adv(self):
        self.motor_adv = tk.Label(self.motor_frame, text="Advanced", fg=WIN_BG_COLOR,
            bg="white", font="Ubuntu 12", cursor="hand2")
        self.motor_adv_font = font.Font(self.motor_adv, self.motor_adv.cget("font"))
        self.motor_adv.configure(font=self.motor_adv_font)
        return self.motor_adv

    def create_test_spin_button(self):
        self.test_spin_button = tk.Button(self.motor_frame, text="Test Spin", 
            fg="white", bg=BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=BUTTON_FOCUS_COLOR,
            activeforeground="white")
        return self.test_spin_button




    '''Camera frame implementations'''

    def create_camera_title(self):
        self.camera_title_text = tk.Label(self.camera_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 14 underline", state="normal", relief="flat", text="Camera")
        return self.camera_title_text

    def create_param_distance_label(self):
        self.param_distance_label = tk.Label(self.camera_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", relief="flat", text="Distance")
        return self.param_distance_label

    def create_param_distance_entry(self):
        self.param_distance_entry = tk.Entry(self.camera_frame, fg=WIN_BG_COLOR,
            font="Ubuntu 14", state="normal", bg=ENTRY_COLOR, width=11, justify="center")
        return self.param_distance_entry

    def create_camera_adv(self):
        self.camera_adv = tk.Label(self.camera_frame, text="Advanced", fg=WIN_BG_COLOR,
            bg="white", font="Ubuntu 12", cursor="hand2")
        self.camera_adv_font = font.Font(self.camera_adv, self.camera_adv.cget("font"))
        self.camera_adv.configure(font=self.camera_adv_font)
        return self.camera_adv

    def create_test_picture_button(self):
        self.test_picture_button = tk.Button(self.camera_frame, text="Test Picture", 
            fg="white", bg=BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=BUTTON_FOCUS_COLOR,
            activeforeground="white")
        return self.test_picture_button



    '''Thresholding frame and preview implementations'''

    def create_thres_title(self):
        self.thres_title_text = tk.Label(self.thres_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 14 underline", state="normal", relief="flat", text="Thresholding")
        return self.thres_title_text

    def create_param_thres_slider(self):
        self.param_thres_slider = tk.Scale(self.thres_frame, relief="flat", sliderrelief="flat",
            orient="horizontal", bg="#FFFFFF", bd = 1, showvalue=1, highlightthickness=0, 
            label="Sensitivity", from_=0, to=1, digits=2, resolution=0.1, font="Ubuntu 12")
        self.param_thres_slider.set(0.5)
        return self.param_thres_slider

    def refresh_preview(self):
        self.thres_source_img_label.destroy()
        self.create_thres_preview()
        self.thres_source_img_label.grid(row=0, column=0, padx=0, pady=0, ipady=0, sticky="snwe")
        pass

    def create_thres_preview(self):
        data_folder = Path(PREVIEW_DIR_PATH)
        data_folder_path = self.dirname / data_folder
        image_path = DEFAULT_PREVIEW_IMG
        image = Image.open(image_path)
        if (len(os.listdir(data_folder_path)) != 0):
            # preview image exists
            image_path = data_folder_path / Path(os.listdir(data_folder_path)[0])
            image = Image.open(image_path)

        [image_size_width, image_size_height] = image.size
        scale_factor = max((image_size_width / IMAGE_WIDTH_CAP), (image_size_height / IMAGE_HEIGHT_CAP))
        scaled_width = int(image_size_width / scale_factor)
        scaled_height = int(image_size_height / scale_factor)
        if (scale_factor < 1):
            scaled_width = int(image_size_width * scale_factor)
            scaled_height = int(image_size_height * scale_factor)
        self.thres_source_img = ImageTk.PhotoImage(Image.open(image_path).resize((scaled_width, scaled_height)))
        self.thres_source_img_label = tk.Label(self.thres_preview_frame, image = self.thres_source_img, 
            bg=RIGHT_FRAME_COLOR, padx=0, pady=0)

    def create_param_xmin_label(self):
        self.param_xmin_label = tk.Label(self.thres_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", relief="flat", text="Left")
        return self.param_xmin_label

    def create_param_xmin_entry(self):
        self.param_xmin_entry = tk.Entry(self.thres_frame, fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", bg=ENTRY_COLOR, width=6, justify="center")
        return self.param_xmin_entry

    def create_param_xmax_label(self):
        self.param_xmax_label = tk.Label(self.thres_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", relief="flat", text="Right")
        return self.param_xmax_label

    def create_param_xmax_entry(self):
        self.param_xmax_entry = tk.Entry(self.thres_frame, fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", bg=ENTRY_COLOR, width=6, justify="center")
        return self.param_xmax_entry

    def create_param_ymin_label(self):
        self.param_ymin_label = tk.Label(self.thres_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", relief="flat", text="Top")
        return self.param_ymin_label

    def create_param_ymin_entry(self):
        self.param_ymin_entry = tk.Entry(self.thres_frame, fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", bg=ENTRY_COLOR, width=6, justify="center")
        return self.param_ymin_entry

    def create_param_ymax_label(self):
        self.param_ymax_label = tk.Label(self.thres_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", relief="flat", text="Bottom")
        return self.param_ymax_label

    def create_param_ymax_entry(self):
        self.param_ymax_entry = tk.Entry(self.thres_frame, fg=WIN_BG_COLOR,
            font="Ubuntu 12", state="normal", bg=ENTRY_COLOR, width=6, justify="center")
        return self.param_ymax_entry

    def create_thres_adv(self):
        self.thres_adv = tk.Label(self.thres_frame, text="Advanced", fg=WIN_BG_COLOR,
            bg="white", font="Ubuntu 12", cursor="hand2")
        self.thres_adv_font = font.Font(self.thres_adv, self.thres_adv.cget("font"))
        self.thres_adv.configure(font=self.thres_adv_font)
        return self.thres_adv

    def create_preview_thres_button(self):
        self.prev_thres_button = tk.Button(self.thres_frame, text="Preview", 
            fg="white", bg=BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=BUTTON_FOCUS_COLOR,
            activeforeground="white")
        return self.prev_thres_button

    '''Threashold advanced frame implementations'''
    
    def create_hue_weight_label(self):
        self.hue_weight_label = tk.Label(self.thres_adv_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 14", state="normal", relief="flat", text="Hue Weight")
        return self.hue_weight_label

    def create_hue_weight_slider(self):
        self.hue_weight_slider = tk.Scale(self.thres_adv_frame, relief="flat", sliderrelief="flat",
            orient="horizontal", bg="#FFFFFF", bd = 1, showvalue=1, highlightthickness=0, 
            from_=0, to=1, digits=2, resolution=0.1, font="Ubuntu 14")
        self.hue_weight_slider.set(0.5)
        return self.hue_weight_slider

    def create_color_dist_label(self):
        self.color_dist_label = tk.Label(self.thres_adv_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 14", state="normal", relief="flat", text="Color Dist")
        return self.color_dist_label

    def create_depth_dist_label(self):
        self.depth_dist_label = tk.Label(self.thres_adv_frame, bg="white", fg=WIN_BG_COLOR,
            font="Ubuntu 14", state="normal", relief="flat", text="Depth Dist")
        return self.depth_dist_label

    def create_color_dist_entry(self):
        self.param_color_dist = tk.Entry(self.thres_adv_frame, fg=WIN_BG_COLOR,
            font="Ubuntu 14", state="normal", bg=ENTRY_COLOR, width=11, justify="center")
        return self.param_color_dist

    def create_depth_dist_entry(self):
        self.param_depth_dist = tk.Entry(self.thres_adv_frame, fg=WIN_BG_COLOR,
            font="Ubuntu 14", state="normal", bg=ENTRY_COLOR, width=11, justify="center")
        return self.param_depth_dist

    def create_close_thres_button(self):
        self.close_thres_button = tk.Button(self.thres_adv_frame, text="Close", 
            fg=WIN_BG_COLOR, bg=CANCEL_BUTTON_COLOR, font="Ubuntu 12", width=12,
            relief="flat", pady=-1, activebackground=CANCEL_BUTTON_FOCUS_COLOR,
            activeforeground=WIN_BG_COLOR)
        return self.close_thres_button


    '''Create and setup'''

    def create_widgets(self):    
        # create frames
        self.settings_frame = tk.Frame(self.window, bg=RIGHT_FRAME_COLOR)
        self.settings_title_frame = tk.Frame(self.settings_frame, bg="white")
        self.motor_frame = tk.Frame(self.settings_frame, bg="white",
            highlightthickness=2, highlightcolor=FRAME_BORDER_COLOR)
        self.camera_frame = tk.Frame(self.settings_frame, bg="white",
            highlightthickness=2, highlightcolor=FRAME_BORDER_COLOR)
        self.thres_frame = tk.Frame(self.settings_frame, bg="white",
            highlightthickness=0)
        self.thres_preview_frame = tk.Frame(self.settings_frame, bg="white",
            highlightthickness=0)
        self.settings_action_frame = tk.Frame(self.settings_frame, bg="white",
            highlightthickness=2, highlightcolor=FRAME_BORDER_COLOR)
        
        # create advanced settings frame
        self.thres_adv_frame = tk.Frame(self.window, bg="white",
            highlightthickness=2, highlightcolor=FRAME_BORDER_COLOR)
        self.camera_adv_frame = tk.Frame(self.window, bg="white",
            highlightthickness=2, highlightcolor=FRAME_BORDER_COLOR)
        self.motor_adv_frame = tk.Frame(self.window, bg="white",
            highlightthickness=2, highlightcolor=FRAME_BORDER_COLOR)
        self.create_title()
        self.create_motor_title()
        self.create_camera_title()
        self.create_thres_title()
        self.create_settings_apply_button()
        self.create_settings_cancel_button()

        # create motor frame widgets
        self.motor_widgets = [self.create_param_speed_slider(),
                            self.create_test_spin_button(),
                            self.create_motor_adv()]

        # create camera frame widgets
        self.camera_widgets = [
                                self.create_param_distance_label(),
                                self.create_param_distance_entry(),
                                self.create_camera_adv(),
                                self.create_test_picture_button(),
                               ]

        # create thres frame widgets
        self.thres_widgets =    [
                                self.create_param_thres_slider(),
                                self.create_thres_preview(),
                                self.create_param_xmax_entry(),
                                self.create_param_xmax_label(),
                                self.create_param_xmin_entry(),
                                self.create_param_xmin_label(),
                                self.create_param_ymax_entry(),
                                self.create_param_ymax_label(),
                                self.create_param_ymin_entry(),
                                self.create_param_ymin_label(),
                                self.create_thres_adv(),
                                self.create_preview_thres_button(),
                                ]

        # create thres adv frame widgets
        self.thres_adv_widgets =    [
                                    self.create_hue_weight_label(),
                                    self.create_color_dist_label(),
                                    self.create_depth_dist_label(),
                                    self.create_color_dist_entry(),
                                    self.create_depth_dist_entry(),
                                    self.create_close_thres_button(),
                                    self.create_hue_weight_slider(),
                                    ]
        
    def setup_layout(self):
        # setup frames
        self.settings_frame.place(rely=0, relx=WIN_SPLIT, relheight=1, relwidth=1-WIN_SPLIT)
        self.settings_title_frame.grid(row=0, column=0, columnspan=2, sticky="snwe")
        self.motor_frame.grid(row=1, column=0, sticky="snwe")
        self.camera_frame.grid(row=1, column=1, sticky="snwe")
        self.thres_frame.grid(row=2, column=0, sticky="nswe")
        self.thres_preview_frame.grid(row=2, column=1, sticky="nswe")
        self.settings_action_frame.grid(row=3, column=0, columnspan=2, sticky="snwe")

        # setup subframes
        self.title_text.grid(row=0, column=0, pady=0, ipady=0)
        self.motor_title_text.grid(row=0, column=0, columnspan=4, pady=0, ipady=0)
        self.camera_title_text.grid(row=0, column=0, columnspan=4, pady=0, ipady=0)
        self.thres_title_text.grid(row=0, column=0, columnspan=4, pady=0, ipady=0)
        self.settings_apply_button.grid(row=0, column=3, pady=5, padx=5, sticky="e")
        self.settings_cancel_button.grid(row=0, column=2, pady=5, padx=5, sticky="e")

        # setup widgets in motor frame
        self.param_speed_slider.grid(row=1, column=0, columnspan=4, padx=22, pady=0, ipady=0, sticky="we")
        self.test_spin_button.grid(row=2, column=2, columnspan=2, padx=10, pady=0, ipady=0, sticky="")
        self.motor_adv.grid(row=2, column=0, columnspan=2, padx=10, pady=0, ipady=0, sticky="")

        # setup widgets in camera frame
        self.param_distance_label.grid(row=1, column=0, columnspan=2, pady=0, padx=0)
        self.param_distance_entry.grid(row=1, column=2, columnspan=2, pady=0, padx=0)
        self.test_picture_button.grid(row=2, column=2, columnspan=2, padx=10, pady=0, ipady=0, sticky="")
        self.camera_adv.grid(row=2, column=0, columnspan=2, padx=10, pady=0, ipady=0, sticky="")

        # setup widgets in thres frame
        self.param_thres_slider.grid(row=1, column=0, columnspan=4, padx=22, pady=0, ipady=0, sticky="we")
        self.thres_source_img_label.grid(row=0, column=0, padx=0, pady=0, ipady=0, sticky="snwe")
        self.param_xmin_label.grid(row=2, column=0, columnspan=1, pady=0, padx=(10, 0))
        self.param_xmin_entry.grid(row=2, column=1, columnspan=1, pady=0, padx=(0, 10))
        self.param_xmax_label.grid(row=2, column=2, columnspan=1, pady=0, padx=(10, 0))
        self.param_xmax_entry.grid(row=2, column=3, columnspan=1, pady=0, padx=(0, 18))
        self.param_ymin_label.grid(row=3, column=0, columnspan=1, pady=0, padx=(10, 0))
        self.param_ymin_entry.grid(row=3, column=1, columnspan=1, pady=0, padx=(0, 10))
        self.param_ymax_label.grid(row=3, column=2, columnspan=1, pady=0, padx=(10, 0))
        self.param_ymax_entry.grid(row=3, column=3, columnspan=1, pady=0, padx=(0, 18))
        self.prev_thres_button.grid(row=4, column=2, columnspan=2, padx=10, pady=0, ipady=0, sticky="")
        self.thres_adv.grid(row=4, column=0, columnspan=2, padx=10, pady=0, ipady=0, sticky="")

        # setup widgets in thres adv frame
        self.hue_weight_label.grid(row=0, column=0, columnspan=2, pady=(40,0), sticky="e")
        self.color_dist_label.grid(row=1, column=0, columnspan=2, sticky="e")
        self.depth_dist_label.grid(row=2, column=0, columnspan=2, sticky="e")
        self.param_color_dist.grid(row=1, column=2, columnspan=2, sticky="")
        self.param_depth_dist.grid(row=2, column=2, columnspan=2, sticky="")
        self.close_thres_button.grid(row=3, column=2, columnspan=2, sticky="se", padx=20, pady=20)
        self.hue_weight_slider.grid(row=0, column=2, columnspan=2, pady=(20,0), padx=0, sticky="")

    def config_setting_frame_row_col(self):
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_columnconfigure(1, weight=1)

        self.settings_frame.grid_rowconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(1, weight=35)
        self.settings_frame.grid_rowconfigure(2, weight=100)
        self.settings_frame.grid_rowconfigure(3, weight=1)
    
    def config_setting_title_frame_row_col(self):
        self.settings_title_frame.grid_columnconfigure(0, weight=1)
        self.settings_title_frame.grid_rowconfigure(0, weight=1)

    def config_setting_action_frame_row_col(self):
        self.settings_action_frame.grid_columnconfigure(0, weight=100)
        self.settings_action_frame.grid_columnconfigure(1, weight=100)
        self.settings_action_frame.grid_columnconfigure(2, weight=1)
        self.settings_action_frame.grid_columnconfigure(3, weight=1)
        self.settings_action_frame.grid_rowconfigure(0, weight=1)

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
        self.camera_frame.grid_rowconfigure(2, weight=750)

    def config_thres_frame_row_col(self):
        self.thres_frame.grid_columnconfigure(0, weight=1)
        self.thres_frame.grid_columnconfigure(1, weight=1)
        self.thres_frame.grid_columnconfigure(2, weight=1)
        self.thres_frame.grid_columnconfigure(3, weight=1)
        self.thres_frame.grid_rowconfigure(0, weight=1)
        self.thres_frame.grid_rowconfigure(1, weight=100)
        self.thres_frame.grid_rowconfigure(2, weight=1000)
        self.thres_frame.grid_rowconfigure(3, weight=1000)
        self.thres_frame.grid_rowconfigure(4, weight=1000)
    
    def config_thres_preview_frame_row_col(self):
        self.thres_preview_frame.grid_columnconfigure(0, weight=1)
        self.thres_preview_frame.grid_rowconfigure(0, weight=1)

    def config_thres_adv_frame_row_col(self):
        self.thres_adv_frame.grid_columnconfigure(0, weight=1)
        self.thres_adv_frame.grid_columnconfigure(1, weight=1)
        self.thres_adv_frame.grid_columnconfigure(2, weight=1)
        self.thres_adv_frame.grid_columnconfigure(3, weight=1)
        self.thres_adv_frame.grid_rowconfigure(0, weight=10)
        self.thres_adv_frame.grid_rowconfigure(1, weight=10)
        self.thres_adv_frame.grid_rowconfigure(2, weight=10)
        self.thres_adv_frame.grid_rowconfigure(3, weight=10)

    def forget_layout(self):
        self.settings_frame.place_forget()

    def open_thres_adv(self):
        self.thres_adv_frame.place(anchor="center", rely=0.5, relx=0.5, relheight=0.5, relwidth=0.5)
        self.thres_adv_frame.focus_set()

    def forget_thres_adv(self):
        self.thres_adv_frame.place_forget()

