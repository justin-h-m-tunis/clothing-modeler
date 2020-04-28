from gui_config import *
import gui_model
import gui_view
from ctypes import windll
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import open3d as o3d
from pathlib import Path
import multiprocessing, time, os, shutil
import numpy as np
from motor_camera import *
import cv2
from bkgThresh import *
from PIL import Image

class GuiController(object):
    """GUI controller that handles model and view"""

    def __init__(self, parent):
        self.init_model()
        self.init_menu()
        self.init_view(parent)
        self.parent = parent
        self.settings_path = "./settings.npz"

    def init_model(self):
        self.model = gui_model.GuiModel(updateFn=lambda n: self.update_progress(n,200),total_macrosteps=200)

    def init_view(self, parent):
        self.view = gui_view.GuiView(parent)
        parent.bind_all("<1>", lambda event:event.widget.focus_set())
        self.bind_intro_text()
        self.bind_q_start()
        self.bind_capture_bg()
        self.bind_adv_option()
        self.bind_motor_adv_option()
        self.bind_motor_test_spin()
        self.bind_camera_adv_option()
        self.bind_camera_test_picture()
        self.bind_settings_apply()
        self.bind_settings_cancel()
        self.bind_thres_adv_option()
        self.bind_preview_thres()
        self.bind_thres_adv_frame()
        
    def init_menu(self):
        self.menubar = tk.Menu(window)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)

        self.file_menu.add_command(label="Import...", command=self.import_settings)
        self.file_menu.add_command(label="Export...", command=self.export_settings)
        # self.file_menu.add_command(label="Save", command=self.do_nothing)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=on_closing)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        self.tool_menu = tk.Menu(self.menubar, tearoff=0)
        self.tool_menu.add_command(label="View Model", command=self.open_ply)
        self.menubar.add_cascade(label="Tools", menu=self.tool_menu)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        # self.help_menu.add_command(label="Help Index", command=lambda : None)
        self.help_menu.add_command(label="About", command=self.open_about)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        window.config(menu=self.menubar)

    def bind_thres_adv_frame(self):
        self.view.settings_panel.thres_adv_frame.bind("<FocusOut>", lambda e: self.view.settings_panel.forget_thres_adv())
        self.view.settings_panel.close_thres_button.bind("<ButtonRelease-1>", lambda e: self.view.settings_panel.forget_thres_adv())

    def bind_intro_text(self):
        self.view.intro_text.bind("<Configure>", self.scale_font)

    def bind_q_start(self):
        self.view.q_start_button.bind("<Enter>",
            lambda e: self.view.q_start_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.q_start_button.bind("<Leave>",
            lambda e: self.view.q_start_button.configure(bg = BUTTON_COLOR))
        self.view.q_start_button.bind("<ButtonRelease-1>", self.run_system)
    
    def bind_capture_bg(self):
        self.view.capture_bg_button.bind("<Enter>",
            lambda e: self.view.capture_bg_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.capture_bg_button.bind("<Leave>",
            lambda e: self.view.capture_bg_button.configure(bg = BUTTON_COLOR))
        self.view.capture_bg_button.bind("<ButtonRelease-1>", self.run_capture_bg)

    def bind_adv_option(self):
        self.view.adv_option.bind("<Enter>",
            lambda e: self.view.opt_font.configure(underline = True))
        self.view.adv_option.bind("<Leave>",
            lambda e: self.view.opt_font.configure(underline = False))
        self.view.adv_option.bind("<ButtonRelease-1>", self.run_settings)

    def bind_motor_adv_option(self):
        self.view.settings_panel.motor_adv.bind("<Enter>",
            lambda e: self.view.settings_panel.motor_adv_font.configure(underline = True))
        self.view.settings_panel.motor_adv.bind("<Leave>",
            lambda e: self.view.settings_panel.motor_adv_font.configure(underline = False))
        self.view.settings_panel.motor_adv.bind("<ButtonRelease-1>", lambda : None)

    def bind_motor_test_spin(self):
        self.view.settings_panel.test_spin_button.bind("<Enter>",
            lambda e: self.view.settings_panel.test_spin_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.settings_panel.test_spin_button.bind("<Leave>",
            lambda e: self.view.settings_panel.test_spin_button.configure(bg = BUTTON_COLOR))
        self.view.settings_panel.test_spin_button.bind("<ButtonRelease-1>", lambda : None)

    def bind_camera_adv_option(self):
        self.view.settings_panel.camera_adv.bind("<Enter>",
            lambda e: self.view.settings_panel.camera_adv_font.configure(underline = True))
        self.view.settings_panel.camera_adv.bind("<Leave>",
            lambda e: self.view.settings_panel.camera_adv_font.configure(underline = False))
        self.view.settings_panel.camera_adv.bind("<ButtonRelease-1>", lambda : None)

    def bind_camera_test_picture(self):
        self.view.settings_panel.test_picture_button.bind("<Enter>",
            lambda e: self.view.settings_panel.test_picture_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.settings_panel.test_picture_button.bind("<Leave>",
            lambda e: self.view.settings_panel.test_picture_button.configure(bg = BUTTON_COLOR))
        self.view.settings_panel.test_picture_button.bind("<ButtonRelease-1>", lambda : None)

    def bind_thres_adv_option(self):
        self.view.settings_panel.thres_adv.bind("<Enter>",
            lambda e: self.view.settings_panel.thres_adv_font.configure(underline = True))
        self.view.settings_panel.thres_adv.bind("<Leave>",
            lambda e: self.view.settings_panel.thres_adv_font.configure(underline = False))
        self.view.settings_panel.thres_adv.bind("<ButtonRelease-1>", lambda e : self.view.settings_panel.open_thres_adv())

    def bind_preview_thres(self):
        self.view.settings_panel.prev_thres_button.bind("<Enter>",
            lambda e: self.view.settings_panel.prev_thres_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.settings_panel.prev_thres_button.bind("<Leave>",
            lambda e: self.view.settings_panel.prev_thres_button.configure(bg = BUTTON_COLOR))
        self.view.settings_panel.prev_thres_button.bind("<ButtonRelease-1>", lambda event : self.img_thres(event,rgb_output_path=PREVIEW_DIR_PATH,img_ind=0))

    def bind_settings_apply(self):
        self.view.settings_panel.settings_apply_button.bind("<Enter>",
            lambda e: self.view.settings_panel.settings_apply_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.settings_panel.settings_apply_button.bind("<Leave>",
            lambda e: self.view.settings_panel.settings_apply_button.configure(bg = BUTTON_COLOR))

        self.view.settings_panel.settings_apply_button.bind("<ButtonRelease-1>", self.save_settings)

    def bind_settings_cancel(self):
        self.view.settings_panel.settings_cancel_button.bind("<Enter>",
            lambda e: self.view.settings_panel.settings_cancel_button.configure(bg = CANCEL_BUTTON_FOCUS_COLOR))
        self.view.settings_panel.settings_cancel_button.bind("<Leave>",
            lambda e: self.view.settings_panel.settings_cancel_button.configure(bg = CANCEL_BUTTON_COLOR))
        self.view.settings_panel.settings_cancel_button.bind("<ButtonRelease-1>", self.cancel_settings)

    def is_crop_error(self, crop_top, crop_left, crop_bottom, crop_right, im):
        width, height = im.size
        if (crop_top < 0 or crop_top > height):
            return True
        if (crop_bottom < 0 or crop_bottom > height):
            return True
        if (crop_left < 0 or crop_left > width):
            return True
        if (crop_right < 0 or crop_right > width):
            return True
        if (crop_left > crop_right or crop_top > crop_bottom):
            return True
        return False

    def img_thres(self, event, rgb_output_path=None, depth_output_path=None, img_ind=0):
        curr_dirname = os.path.dirname(__file__)
        img_path = 'data/color/'
        depth_path = 'data/depth/'
        bkg_path = 'data/color_bkg/'
        depth_bkg_path = 'data/depth_bkg/'
        dirs = [img_path, depth_path, bkg_path, depth_bkg_path]
        for dir in dirs:
            if (len(os.listdir(dir)) == 0): # exit when empty folder found
                return 
        print("Processing image " + str(img_ind))
        # temp, will pull from advanced settings panel later
        depth_thresh = int(self.view.settings_panel.param_depth_dist.get())
        rgb_thresh = int(self.view.settings_panel.param_color_dist.get())

        img_names = [os.listdir(img_path), os.listdir(depth_path)]
        f = ['color_' + str(img_ind) + '.png','Depth_' + str(img_ind) + '.png']
        img = cv2.imread(img_path + f[0])
        rgb_dist = getBkgDistRGB(img,cv2.imread(bkg_path + f[0]))
        depth = cv2.imread(depth_path + f[1])
        depth_dist = getBkgDistDepth(depth,cv2.imread(depth_bkg_path + f[1]))
        bkg_thresh_rgb = removeBkg(img,np.array([rgb_dist,depth_dist]),[rgb_thresh,depth_thresh],'or')
        bkg_thresh_depth = removeBkg(depth,np.array([rgb_dist,depth_dist]),[rgb_thresh,depth_thresh],'or')
        '''
        take bkg_thresh_rgb and bkg_depth_rgb and do static crop/color filtering
        '''
        crop_left = int(self.view.settings_panel.param_xmin_entry.get())
        crop_right = int(self.view.settings_panel.param_xmax_entry.get())
        crop_top = int(self.view.settings_panel.param_ymin_entry.get())
        crop_bottom = int(self.view.settings_panel.param_ymax_entry.get())

        if rgb_output_path is not None:
            cv2.imwrite(rgb_output_path + f[0], bkg_thresh_rgb)
            im = Image.open(rgb_output_path + f[0])
            if (not self.is_crop_error(crop_top, crop_left, crop_bottom, crop_right, im)):
                im2 = im.crop((crop_left, crop_top, crop_right, crop_bottom))
                im2.save(rgb_output_path + f[0])
            else:
                print("crop dimension error! showing original image")
        if depth_output_path is not None:
            cv2.imwrite(depth_output_path + f[0], bkg_thresh_depth)
            im = Image.open(depth_output_path + f[0])
            if (not self.is_crop_error(crop_top, crop_left, crop_bottom, crop_right, im)):
                im2 = im.crop((crop_left, crop_top, crop_right, crop_bottom))
                im2.save(depth_output_path + f[0])
            else:
                print("crop dimension error! showing original image")
        self.view.settings_panel.refresh_preview()
        print("done!")

    '''Main logic execution'''
    def run_system(self, event):
        print("3D scanning system start")
        self.view.manage_settings(self.parent, False)
        self.model.run_motor_camera()

    def run_capture_bg(self, event):
        print("3D scanning system start: capture bg")
        self.view.manage_settings(self.parent, False)
        pass

    '''export settings file to a disk location'''
    def export_settings(self):
        path_name = filedialog.askdirectory(initialdir = "/", title = "Select save directory")
        print(path_name)
        if (path_name == ""):
            return
        else:
            shutil.copy2(self.settings_path, path_name)

    '''import settings from npz file'''
    def import_settings(self):
        path_name = filedialog.askopenfilename(initialdir = "/", title = "Select settings file", filetypes = (("npz files","*.npz"),))
        print(path_name)
        if (path_name == ""):
            return
        else:
            self.settings_path = path_name
            self.run_settings("<1>", True)  

    '''load settings from a given npz file'''
    def load_settings(self, path_name):
        data = np.load(path_name)
        self.view.settings_panel.param_speed_slider.set(data["speed"])
        self.view.settings_panel.param_distance_entry.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_distance_entry.insert(0, data["distance"])
        self.view.settings_panel.param_thres_slider.set(data["sensitivity"])
        self.view.settings_panel.param_xmin_entry.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_xmin_entry.insert(0, data["xmin_left"])
        self.view.settings_panel.param_xmax_entry.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_xmax_entry.insert(0, data["xmax_right"])
        self.view.settings_panel.param_ymin_entry.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_ymin_entry.insert(0, data["ymin_top"])
        self.view.settings_panel.param_ymax_entry.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_ymax_entry.insert(0, data["ymax_bottom"])

        self.view.settings_panel.param_color_dist.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_color_dist.insert(0, data["color_dist"])
        self.view.settings_panel.param_depth_dist.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_depth_dist.insert(0, data["depth_dist"])
        self.view.settings_panel.hue_weight_slider.set(data["hue_weight"])

    '''Open settings panel'''
    def run_settings(self, event, action=None):
        if (os.path.exists(self.settings_path)):
            self.load_settings(self.settings_path)
        self.view.manage_settings(self.parent, action)

    '''Save params to npz file on disk'''
    def save_settings(self, event):
        speed = self.view.settings_panel.param_speed_slider.get()
        distance = self.view.settings_panel.param_distance_entry.get()
        sensitivity = self.view.settings_panel.param_thres_slider.get()
        xmin_left = self.view.settings_panel.param_xmin_entry.get()
        xmax_right = self.view.settings_panel.param_xmax_entry.get()
        ymin_top = self.view.settings_panel.param_ymin_entry.get()
        ymax_bottom = self.view.settings_panel.param_ymax_entry.get()
        color_dist = self.view.settings_panel.param_color_dist.get()
        depth_dist = self.view.settings_panel.param_depth_dist.get()
        hue_weight = self.view.settings_panel.hue_weight_slider.get()
        
        np.savez(self.settings_path, speed=speed, distance=distance,
            sensitivity=sensitivity, xmin_left=xmin_left, xmax_right=xmax_right,
            ymin_top=ymin_top, ymax_bottom=ymax_bottom, color_dist=color_dist,
            depth_dist=depth_dist, hue_weight=hue_weight)
        self.view.manage_settings(self.parent, False)
        # run system
        # self.run_system(event)

    def cancel_settings(self, event):
        self.view.manage_settings(self.parent)

    def scale_font(self, event):
        if (window.winfo_width() < 900):
            self.view.intro_text.config(font="Ubuntu 16")
        else:
            self.view.intro_text.config(font="Ubuntu 20")

    def open_about(self):
        tk.messagebox.showinfo("About", ABOUT_TEXT)

    def open_ply(self):
        self.path_name = filedialog.askopenfilename(initialdir = "/", title = "Select ply file", filetypes = (("ply files","*.ply"),))
        print(self.path_name)
        if (self.path_name == ""):
            return
        
        self.p = multiprocessing.Process(target=gui_model.view_ply, args=(self.path_name,))
        self.p.start()

    def update_progress(self, curr_step, max_step):
        print(curr_step,"/",max_step)
        self.view.update_progress_bar(curr_step, max_step)

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
