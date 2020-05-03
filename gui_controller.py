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
import pickle
import os
import threading

class GuiController(object):
    """GUI controller that handles model and view"""

    def __init__(self, parent):
        self.init_menu()
        self.settings_path = "./settings.npz"
        try:
            self.init_model(settings=np.load(self.settings_path))
        except IOError:
            self.init_model(settings=None)
        self.init_view(parent)
        self.parent = parent
        self.optimizer_weights=[]
        self.optimizer_biases=[]

    def init_model(self, settings=None, onSerialFail=lambda: print("Motor not found!")):
        self.model = gui_model.GuiModel(updateFn=lambda n: self.update_progress(n,200),settings=settings, onSerialFail=onSerialFail)

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
        self.bind_motor_adv_frame()
        self.bind_camera_adv_frame()
        self.bind_stop()
        self.bind_optimize()
        self.bind_optimize_revert()
        
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

    def bind_motor_adv_frame(self):
        self.view.settings_panel.motor_adv_frame.bind("<FocusOut>", lambda e: self.view.settings_panel.forget_motor_adv())
        self.view.settings_panel.close_motor_button.bind("<ButtonRelease-1>", lambda e: self.view.settings_panel.forget_motor_adv())

    def bind_camera_adv_frame(self):
        self.view.settings_panel.camera_adv_frame.bind("<FocusOut>", lambda e: self.view.settings_panel.forget_camera_adv())
        self.view.settings_panel.close_camera_button.bind("<ButtonRelease-1>", lambda e: self.view.settings_panel.forget_camera_adv())

    def bind_intro_text(self):
        self.view.intro_text.bind("<Configure>", self.scale_font)

    def bind_stop(self):
        self.view.stop_button.bind("<Enter>",
            lambda e: self.view.stop_button.configure(bg = STOP_BUTTON_FOCUS_COLOR))
        self.view.stop_button.bind("<Leave>",
            lambda e: self.view.stop_button.configure(bg = STOP_BUTTON_COLOR))
        self.view.stop_button.bind("<ButtonRelease-1>", lambda event: self.stop_system())

    def bind_q_start(self):
        self.view.q_start_button.bind("<Enter>",
            lambda e: self.view.q_start_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.q_start_button.bind("<Leave>",
            lambda e: self.view.q_start_button.configure(bg = BUTTON_COLOR))
        self.view.q_start_button.bind("<ButtonRelease-1>", lambda event: self.run_system())
    
    def bind_capture_bg(self):
        self.view.capture_bg_button.bind("<Enter>",
            lambda e: self.view.capture_bg_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.capture_bg_button.bind("<Leave>",
            lambda e: self.view.capture_bg_button.configure(bg = BUTTON_COLOR))
        self.view.capture_bg_button.bind("<ButtonRelease-1>", lambda event : self.model.run_motor_camera(img_path='data/bkg/'))

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
        self.view.settings_panel.motor_adv.bind("<ButtonRelease-1>", lambda e : self.view.settings_panel.open_motor_adv())

    def bind_motor_test_spin(self):
        self.view.settings_panel.test_spin_button.bind("<Enter>",
            lambda e: self.view.settings_panel.test_spin_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.settings_panel.test_spin_button.bind("<Leave>",
            lambda e: self.view.settings_panel.test_spin_button.configure(bg = BUTTON_COLOR))
        self.view.settings_panel.test_spin_button.bind("<ButtonRelease-1>", lambda event: self.model.motor.fullRotation())

    def bind_camera_adv_option(self):
        self.view.settings_panel.camera_adv.bind("<Enter>",
            lambda e: self.view.settings_panel.camera_adv_font.configure(underline = True))
        self.view.settings_panel.camera_adv.bind("<Leave>",
            lambda e: self.view.settings_panel.camera_adv_font.configure(underline = False))
        self.view.settings_panel.camera_adv.bind("<ButtonRelease-1>", lambda e : self.view.settings_panel.open_camera_adv())

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
        self.view.settings_panel.prev_thres_button.bind("<ButtonRelease-1>", lambda event : self.img_thres(rgb_output_path=PREVIEW_DIR_PATH,img_ind=50))

    def bind_optimize(self):
        self.view.settings_panel.optimize_button.bind("<Enter>",
            lambda e: self.view.settings_panel.optimize_button.configure(bg=BUTTON_FOCUS_COLOR))
        self.view.settings_panel.optimize_button.bind("<Leave>",
            lambda e: self.view.settings_panel.optimize_button.configure(bg=BUTTON_COLOR))
        self.view.settings_panel.optimize_button.bind("<ButtonRelease-1>",
            lambda event: self.get_optimized_weights(W0=self.optimizer_weights,b0=self.optimizer_biases,
                                                      learning_speed=self.view.settings_panel.optimize_speed_slider.get(),
                                                        optimize_quality=self.view.settings_panel.optimize_quality_slider.get()))

    def bind_optimize_revert(self):
        self.view.settings_panel.optimize_revert.bind("<Enter>",
            lambda e: self.view.settings_panel.optimize_revert.configure(bg=BUTTON_FOCUS_COLOR))
        self.view.settings_panel.optimize_revert.bind("<Leave>",
            lambda e: self.view.settings_panel.optimize_revert.configure(bg=BUTTON_COLOR))
        def reset_optimizer():
            self.optimizer_biases = []
            self.optimizer_weights = []
            print("optimizer reset")
        self.view.settings_panel.optimize_revert.bind("<ButtonRelease-1>", lambda event: reset_optimizer())

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
        if (crop_left < 0 or crop_left >= width/2 - 5):
            return True
        if (crop_right < 0 or crop_right <= width/2 - 5):
            return True
        if (crop_left >= crop_right or crop_top >= crop_bottom):
            return True
        return False

    '''checks thresholding preview entry errors'''
    def parse_thres_depth_rgb_dist(self):
        depth_str = self.view.settings_panel.param_depth_dist.get()
        rgb_str = self.view.settings_panel.param_color_dist.get()
        settings = np.load(self.settings_path)
        
        if (depth_str == ''):
            if settings["depth_dist"] == '':
                depth_result = 50
            else:
                depth_result = float(settings["depth_dist"])
        else:
            depth_result = float(depth_str)
        
        if (rgb_str == ''):
            if settings["color_dist"] == '':
                rgb_result = 30
            else:
                rgb_result = float(settings["color_dist"])
        else:
            rgb_result = float(rgb_str)
        return (depth_result, rgb_result)

    '''checks thresholding preview entry errors'''
    def parse_crop_params2(self, im):
        crop_horizontal_str = self.view.settings_panel.param_xmin_entry.get()
        crop_top_str = self.view.settings_panel.param_ymin_entry.get()
        crop_bottom_str = self.view.settings_panel.param_ymax_entry.get()
        settings = np.load(self.settings_path)
        width, height = im.size

        crop_left = -1
        crop_right = -1
        if (crop_horizontal_str == ''):
            if settings["xmin_left"] == '':
                crop_horizontal_result = -1
            else:
                crop_horizontal_result = int(settings["xmin_left"])
                to_crop = int(width * ((crop_horizontal_result/100)/2))
                crop_left = to_crop
                crop_right = width - to_crop
        else:
            crop_horizontal_result = int(float(crop_horizontal_str))
            to_crop = int(width * ((crop_horizontal_result/100)/2))
            crop_left = to_crop
            crop_right = width - to_crop
        
        if (crop_top_str == ''):
            if settings["ymin_top"] == '':
                crop_top_result = -1
            else:
                crop_top_result = int(settings["ymin_top"])
        else:
            crop_top_result = int(crop_top_str)
        if (crop_bottom_str == ''):
            if settings["ymax_bottom"] == '':
                crop_bottom_result = -1
            else:
                crop_bottom_result = int(settings["ymax_bottom"])
        else:
            crop_bottom_result = int(crop_bottom_str)
            
        return [crop_left, crop_right, crop_top_result, crop_bottom_result]

    def get_optimized_weights(self,W0,b0,optimize_quality,learning_speed):
        print("optimizing!")
        img_path = 'data/color/'
        depth_path = 'data/depth/'
        bkg_path = 'data/bkg/color/'
        depth_bkg_path = 'data/bkg/depth/'
        sample_num = int(50*optimize_quality)
        im = Image.open(img_path + 'color_1.png')
        [crop_left, crop_right, crop_top, crop_bottom] = self.parse_crop_params2(im)
        w = crop_right - crop_left
        h = crop_bottom - crop_top
        ims = np.zeros((h, w, 3, sample_num))
        depths = np.zeros((h, w, sample_num))
        bkgs = np.zeros((h, w, 3, sample_num))
        bkgdepths = np.zeros((h, w, sample_num))
        for i in range(sample_num):
            f = ['color_' + str(int(200 / sample_num * i)) + '.png',
                'Depth_' + str(int(200 / sample_num * i)) + '.png']
            print(i, img_path + f[0])
            mask = getThresholdMask(
                cv2.imread(img_path + f[0])[crop_top:crop_bottom, crop_left:crop_right, :],
                cv2.imread(depth_path + f[1], cv2.IMREAD_ANYDEPTH)[crop_top:crop_bottom, crop_left:crop_right],
                "backdrop.png",
                "mannequin.png",
                depth_range=(700, 1270),
                bk_confidence=self.view.settings_panel.similarity_to_backdrop_slider.get(),
                man_confidence=self.view.settings_panel.similarity_to_mannequin_slider.get(),
                apply_mask=False)
            ims[:, :, :, i] = applyMask( cv2.imread(img_path + f[0])[crop_top:crop_bottom, crop_left:crop_right, :], mask,3)
            cv2.waitKey(1)
            depths[:, :, i] = applyMask(cv2.imread(depth_path + f[1], cv2.IMREAD_ANYDEPTH)[crop_top:crop_bottom, crop_left:crop_right], mask,1)
            bkgs[:, :, :, i] = applyMask(cv2.imread(bkg_path + f[0])[crop_top:crop_bottom, crop_left:crop_right, :], mask,3)
            bkgdepths[:, :, i] = applyMask(cv2.imread(depth_bkg_path + f[1], cv2.IMREAD_ANYDEPTH)[crop_top:crop_bottom, crop_left:crop_right], mask,1)
        self.optimizer_weights, self.optimizer_biases = optimizeWeights(ims.astype(np.uint8), depths, bkgs.astype(np.uint8),
                        bkgdepths, epochs=3,learning_rate=.25*learning_speed, starting_weights=self.optimizer_weights, starting_biases=self.optimizer_biases)

    def img_thres(self, rgb_output_path=None, depth_output_path=None, img_ind=50):
        curr_dirname = os.path.dirname(__file__)
        img_path = 'data/color/'
        depth_path = 'data/depth/'
        bkg_path = 'data/bkg/color/'
        depth_bkg_path = 'data/bkg/depth/'
        dirs = [img_path, depth_path, bkg_path, depth_bkg_path]
        for dir in dirs:
            if (len(os.listdir(dir)) == 0): # exit when empty folder found
                return 
        print("Processing image " + str(img_ind))
        # pull parameters from settings/preview panel
        (depth_thresh, rgb_thresh) = self.parse_thres_depth_rgb_dist()
        im = Image.open(img_path + 'color_1.png')
        [crop_left, crop_right, crop_top, crop_bottom] = self.parse_crop_params2(im)
        print([crop_left, crop_right, crop_top, crop_bottom])
        w = crop_right - crop_left
        h = crop_bottom - crop_top

        f = ['color_' + str(img_ind) + '.png', 'Depth_' + str(img_ind) + '.png']
        im = cv2.imread(img_path + f[0])[crop_top:crop_bottom,crop_left:crop_right, :]
        depth = cv2.imread(depth_path + f[1], cv2.IMREAD_ANYDEPTH)[crop_top:crop_bottom,crop_left:crop_right]
        im_bk = cv2.imread(bkg_path + f[0])[crop_top:crop_bottom,crop_left:crop_right, :]
        depth_bk = cv2.imread(depth_bkg_path + f[1], cv2.IMREAD_ANYDEPTH)[crop_top:crop_bottom,crop_left:crop_right]

        bkg_thresh_rgb, bkg_thresh_depth = removeBackgroundThreshold(
                                            im.astype(np.uint8),
                                            depth,
                                            im_bk,
                                            depth_bk,
                                            depth_range = (700,1270),
                                            bk_path='backdrop.png',
                                            man_path='mannequin.png',
                                            bk_confidence=self.view.settings_panel.similarity_to_backdrop_slider.get(),
                                            man_confidence=self.view.settings_panel.similarity_to_mannequin_slider.get(),
                                            bk_weights=self.optimizer_weights,
                                            bk_biases=self.optimizer_biases,
                                            show_images=False)
        '''
        take bkg_thresh_rgb and bkg_depth_rgb and do static crop/color filtering
        '''
        if rgb_output_path is not None:
            cv2.imwrite(rgb_output_path + f[0], bkg_thresh_rgb)
            '''im = Image.open(rgb_output_path + f[0])
            [crop_left, crop_right, crop_top, crop_bottom] = self.parse_crop_params2(im)
            if (not self.is_crop_error(crop_top, crop_left, crop_bottom, crop_right, im)):
                im2 = im.crop((crop_left, crop_top, crop_right, crop_bottom))
                im2.save(rgb_output_path + f[0])
            else:
                print("crop dimension error! showing original image")'''
        if depth_output_path is not None:
            cv2.imwrite(depth_output_path + f[1], bkg_thresh_depth)
        '''im = Image.open(depth_output_path + f[1])
            [crop_left, crop_right, crop_top, crop_bottom] = self.parse_crop_params2(im)
            if (not self.is_crop_error(crop_top, crop_left, crop_bottom, crop_right, im)):
                im2 = im.crop((crop_left, crop_top, crop_right, crop_bottom))
                im2.save(depth_output_path + f[1])
            else:
                print("crop dimension error! showing original image")'''
        self.view.settings_panel.refresh_preview()
        print("done!")

    '''Stop the process'''
    def stop_system(self):
        print("System stop")
        self.view.forget_stop()
        self.view.place_q_start()

    '''Main logic execution'''
    def run_system(self, img_path='data/', get_images=False, Threshold_images=True,Stitch_images=False):
        print("3D scanning system start with default settings")
        self.view.forget_q_start()
        self.view.place_stop()
        self.view.manage_settings(self.parent, False)
        self.load_settings(self.settings_path)
        settings = np.load(self.settings_path)
        if get_images:
            # init model
            try:

                def raise_serial_exception():
                    raise Exception("Motor disconnected, aborting!")

                self.model.run_motor_camera(img_path=img_path)
            except Exception as E:
                print(E)
                return
        if Threshold_images:
            threads = []
            '''for i in range(settings['macrosteps']):
                threads.append(threading.Thread(target=self.img_thres,
                                                    args=('data/color_thresholded/','data/depth_thresholded/',i)))
                threads[-1].start()
                if len(threads) == n:
                    print(i)
                    for t in threads:
                        t.join()
                    print("done")
                    threads = []'''

            self.img_thres(rgb_output_path='data/color_thresholded/',depth_output_path='data/depth_thresholded/',img_ind=i)
        if Stitch_images:
            pass


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
        # self.view.settings_panel.param_xmax_entry.delete(0, MAX_CHAR_LEN)
        # self.view.settings_panel.param_xmax_entry.insert(0, data["xmax_right"])
        self.view.settings_panel.param_ymin_entry.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_ymin_entry.insert(0, data["ymin_top"])
        self.view.settings_panel.param_ymax_entry.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_ymax_entry.insert(0, data["ymax_bottom"])

        self.view.settings_panel.param_color_dist.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_color_dist.insert(0, data["color_dist"])
        self.view.settings_panel.param_depth_dist.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_depth_dist.insert(0, data["depth_dist"])
        self.view.settings_panel.hue_weight_slider.set(data["hue_weight"])

        # modified thresh adv panel
        self.view.settings_panel.similarity_to_backdrop_slider.set(data["similarity_to_backdrop"])
        self.view.settings_panel.similarity_to_mannequin_slider.set(data["similarity_to_mannequin"])
        self.view.settings_panel.optimize_speed_slider.set(data["optimize_speed"])
        self.view.settings_panel.optimize_quality_slider.set(data["optimize_quality"])

        self.view.settings_panel.param_rise_time.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_rise_time.insert(0, data["rise_time"])
        self.view.settings_panel.param_fall_time.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_fall_time.insert(0, data["fall_time"])
        self.view.settings_panel.param_delay_time.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_delay_time.insert(0, data["delay_time"])
        self.view.settings_panel.param_angular_velocity.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_angular_velocity.insert(0, data["angular_velocity"])

        self.view.settings_panel.param_fx.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_fx.insert(0, data["fx"])
        self.view.settings_panel.param_fy.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_fy.insert(0, data["fy"])
        self.view.settings_panel.param_ux.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_ux.insert(0, data["ux"])
        self.view.settings_panel.param_vy.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_vy.insert(0, data["vy"])

        self.view.settings_panel.param_r1_11.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r1_11.insert(0, data["r1_11"])
        self.view.settings_panel.param_r1_12.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r1_12.insert(0, data["r1_12"])
        self.view.settings_panel.param_r1_13.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r1_13.insert(0, data["r1_13"])
        self.view.settings_panel.param_r1_21.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r1_21.insert(0, data["r1_21"])
        self.view.settings_panel.param_r1_22.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r1_22.insert(0, data["r1_22"])
        self.view.settings_panel.param_r1_23.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r1_23.insert(0, data["r1_23"])
        self.view.settings_panel.param_r1_31.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r1_31.insert(0, data["r1_31"])
        self.view.settings_panel.param_r1_32.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r1_32.insert(0, data["r1_32"])
        self.view.settings_panel.param_r1_33.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r1_33.insert(0, data["r1_33"])

        self.view.settings_panel.param_r2_11.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r2_11.insert(0, data["r2_11"])
        self.view.settings_panel.param_r2_12.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r2_12.insert(0, data["r2_12"])
        self.view.settings_panel.param_r2_13.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r2_13.insert(0, data["r2_13"])
        self.view.settings_panel.param_r2_21.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r2_21.insert(0, data["r2_21"])
        self.view.settings_panel.param_r2_22.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r2_22.insert(0, data["r2_22"])
        self.view.settings_panel.param_r2_23.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r2_23.insert(0, data["r2_23"])
        self.view.settings_panel.param_r2_31.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r2_31.insert(0, data["r2_31"])
        self.view.settings_panel.param_r2_32.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r2_32.insert(0, data["r2_32"])
        self.view.settings_panel.param_r2_33.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_r2_33.insert(0, data["r2_33"])

        self.view.settings_panel.param_t1_11.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_t1_11.insert(0, data["t1_11"])
        self.view.settings_panel.param_t1_21.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_t1_21.insert(0, data["t1_21"])
        self.view.settings_panel.param_t1_31.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_t1_31.insert(0, data["t1_21"])

        self.view.settings_panel.param_t2_11.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_t2_11.insert(0, data["t2_11"])
        self.view.settings_panel.param_t2_21.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_t2_21.insert(0, data["t2_21"])
        self.view.settings_panel.param_t2_31.delete(0, MAX_CHAR_LEN)
        self.view.settings_panel.param_t2_31.insert(0, data["t2_21"])

        self.optimizer_weights = data["optimizer_weights"]
        self.optimizer_biases = data["optimizer_biases"]

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
        # xmax_right = self.view.settings_panel.param_xmax_entry.get()
        ymin_top = self.view.settings_panel.param_ymin_entry.get()
        ymax_bottom = self.view.settings_panel.param_ymax_entry.get()
        baud = 9600
        com = 'COM3'
        w_max = np.power(MAXIMUM_OMEGA/MINIMUM_OMEGA,speed)*MINIMUM_OMEGA
        tr = speed*(MAXIMUM_TR- MINIMUM_TR) + MINIMUM_TR
        delay = speed*(MAXIMUM_DELAY - MINIMUM_DELAY) + MINIMUM_DELAY
        macrostep_time = (360/200)/w_max + tr + delay + DELAY_TOL
        macrosteps=200

        color_dist = self.view.settings_panel.param_color_dist.get()
        depth_dist = self.view.settings_panel.param_depth_dist.get()
        hue_weight = self.view.settings_panel.hue_weight_slider.get()

        # modified thresh adv panel
        similarity_to_backdrop = self.view.settings_panel.similarity_to_backdrop_slider.get()
        similarity_to_mannequin = self.view.settings_panel.similarity_to_mannequin_slider.get()
        optimize_speed = self.view.settings_panel.optimize_speed_slider.get()
        optimize_quality = self.view.settings_panel.optimize_quality_slider.get()

        rise_time = self.view.settings_panel.param_rise_time.get()
        fall_time = self.view.settings_panel.param_fall_time.get()
        delay_time = self.view.settings_panel.param_delay_time.get()
        angular_velocity = self.view.settings_panel.param_angular_velocity.get()

        fx = self.view.settings_panel.param_fx.get()
        fy = self.view.settings_panel.param_fy.get()
        ux = self.view.settings_panel.param_ux.get()
        vy = self.view.settings_panel.param_vy.get()

        r1_11 = self.view.settings_panel.param_r1_11.get()
        r1_12 = self.view.settings_panel.param_r1_12.get()
        r1_13 = self.view.settings_panel.param_r1_13.get()
        r1_21 = self.view.settings_panel.param_r1_21.get()
        r1_22 = self.view.settings_panel.param_r1_22.get()
        r1_23 = self.view.settings_panel.param_r1_23.get()
        r1_31 = self.view.settings_panel.param_r1_31.get()
        r1_32 = self.view.settings_panel.param_r1_32.get()
        r1_33 = self.view.settings_panel.param_r1_33.get()

        r2_11 = self.view.settings_panel.param_r2_11.get()
        r2_12 = self.view.settings_panel.param_r2_12.get()
        r2_13 = self.view.settings_panel.param_r2_13.get()
        r2_21 = self.view.settings_panel.param_r2_21.get()
        r2_22 = self.view.settings_panel.param_r2_22.get()
        r2_23 = self.view.settings_panel.param_r2_23.get()
        r2_31 = self.view.settings_panel.param_r2_31.get()
        r2_32 = self.view.settings_panel.param_r2_32.get()
        r2_33 = self.view.settings_panel.param_r2_33.get()

        t1_11 = self.view.settings_panel.param_t1_11.get()
        t1_21 = self.view.settings_panel.param_t1_21.get()
        t1_31 = self.view.settings_panel.param_t1_31.get()

        t2_11 = self.view.settings_panel.param_t2_11.get()
        t2_21 = self.view.settings_panel.param_t2_21.get()
        t2_31 = self.view.settings_panel.param_t2_31.get()
        
        np.savez(self.settings_path,
                    speed=speed,
                    distance=distance,
                    sensitivity=sensitivity,
                    xmin_left=xmin_left,
                    # xmax_right=xmax_right,
                    ymin_top=ymin_top,
                    ymax_bottom=ymax_bottom,
                    macrostep_time=macrostep_time,
                    baud=baud,
                    com=com,
                    macrosteps=macrosteps,
                    color_dist=color_dist,
                    depth_dist=depth_dist,
                    hue_weight=hue_weight,
                    rise_time=rise_time,
                    fall_time=fall_time,
                    delay_time=delay_time,
                    angular_velocity=angular_velocity,
                    fx=fx,
                    fy=fy,
                    ux=ux,
                    vy=vy,
                    r1_11=r1_11,
                    r1_12=r1_12,
                    r1_13=r1_13,
                    r1_21=r1_21,
                    r1_22=r1_22,
                    r1_23=r1_23,
                    r1_31=r1_31,
                    r1_32=r1_32,
                    r1_33=r1_33,
                    r2_11=r2_11,
                    r2_12=r2_12,
                    r2_13=r2_13,
                    r2_21=r2_21,
                    r2_22=r2_22,
                    r2_23=r2_23,
                    r2_31=r2_31,
                    r2_32=r2_32,
                    r2_33=r2_33,
                    t1_11=t1_11,
                    t1_21=t1_21,
                    t1_31=t1_31,
                    t2_11=t2_11,
                    t2_21=t2_21,
                    t2_31=t2_31,
                    similarity_to_backdrop=similarity_to_backdrop,
                    similarity_to_mannequin=similarity_to_mannequin,
                    optimize_speed=optimize_speed,
                    optimize_quality=optimize_quality,
                    optimizer_weights=self.optimizer_weights,
                    optimizer_biases=self.optimizer_biases
                )
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
