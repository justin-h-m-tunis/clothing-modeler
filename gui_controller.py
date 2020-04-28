from gui_config import *
import gui_model
import gui_view
from ctypes import windll
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import open3d as o3d
import multiprocessing, time
from motor_camera import *

class GuiController(object):
    """GUI controller that handles model and view"""

    def __init__(self, parent):
        self.init_menu()
        self.init_view(parent)
        self.parent = parent
        self.settings_path = "./settings.npz"
        self.init_model()


    def init_model(self, settings=None, onSerialFail=print("Motor not found!")):
        self.model = gui_model.GuiModel(updateFn=lambda n: self.update_progress(n,200),settings=settings, onSerialFail=onSerialFail)

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
        self.help_menu.add_command(label="About", command=self.open_about)
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
        self.view.settings_panel.test_spin_button.bind("<ButtonRelease-1>", self.model.run_motor_camera(img_path='data/bkg'))

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
        self.view.settings_panel.thres_adv.bind("<ButtonRelease-1>", lambda : None)

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

    def img_thres(self, event=None, rgb_output_path=None, depth_output_path=None, img_ind=0):
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
        # temp, will pull from advanced settings panel later
        depth_thresh = 30
        rgb_thresh = 50

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
        if rgb_output_path is not None:
            cv2.imwrite(rgb_output_path + f[0], bkg_thresh_rgb)
        if depth_output_path is not None:
            cv2.imwrite(depth_output_path + f[0], bkg_thresh_depth)
        self.view.settings_panel.refresh_preview()
        print("done!")

    '''Main logic execution'''
    def run_system(self, event, img_path='data', get_images=True, Threshold_images=True,Stitch_images=True):
        print("3D scanning system start with default settings")
        self.view.manage_settings(self.parent, False)
        settings = np.load(self.settings_path)
        if get_images:
            # init model
            try:

                def raise_serial_exception():
                    raise Exception("Motor disconnected, aborting!")

                self.init_model(onSerialFail=raise_serial_exception())
                self.model.run_motor_camera(img_path=img_path, settings=settings, onSerialFail=raise_serial_exception())
            except Exception as E:
                print(E)
        if Threshold_images:
            for i in range(settings['macrosteps']):
                img_thresh(rgb_output_path='data/color',depth_output_path='data/depth',img_ind=i)
        if Stitch_images:
            pass

    '''Place holder for advanced options'''
    def run_adv_option(self, event):
        print("Opening advanced menu")

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
