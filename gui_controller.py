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
        self.init_model()
        self.init_menu()
        self.init_view(parent)
        self.parent = parent

    def init_model(self):
        self.model = gui_model.GuiModel(updateFn=lambda n: self.update_progress(n,200),total_macrosteps=200)

    def init_view(self, parent):
        self.view = gui_view.GuiView(parent)
        parent.bind_all("<1>", lambda event:event.widget.focus_set())
        self.bind_intro_text()
        self.bind_q_start()
        self.bind_adv_option()
        self.bind_motor_adv_option()
        self.bind_motor_test_spin()
        self.bind_camera_adv_option()
        self.bind_camera_test_picture()
        self.bind_settings_apply()
        self.bind_settings_cancel()
        self.bind_thres_adv_option()
        self.bind_preview_thres()
        
    def init_menu(self):
        self.menubar = tk.Menu(window)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Import...", command=lambda : None)
        self.file_menu.add_command(label="Export...", command=lambda : None)
        # self.file_menu.add_command(label="Save", command=lambda : None)
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
        self.view.settings_panel.thres_adv.bind("<ButtonRelease-1>", lambda : None)

    def bind_preview_thres(self):
        self.view.settings_panel.prev_thres_button.bind("<Enter>",
            lambda e: self.view.settings_panel.prev_thres_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.settings_panel.prev_thres_button.bind("<Leave>",
            lambda e: self.view.settings_panel.prev_thres_button.configure(bg = BUTTON_COLOR))
        self.view.settings_panel.prev_thres_button.bind("<ButtonRelease-1>", lambda : None)

    def bind_settings_apply(self):
        self.view.settings_panel.settings_apply_button.bind("<Enter>",
            lambda e: self.view.settings_panel.settings_apply_button.configure(bg = BUTTON_FOCUS_COLOR))
        self.view.settings_panel.settings_apply_button.bind("<Leave>",
            lambda e: self.view.settings_panel.settings_apply_button.configure(bg = BUTTON_COLOR))
        self.view.settings_panel.settings_apply_button.bind("<ButtonRelease-1>", lambda : None)

    def bind_settings_cancel(self):
        self.view.settings_panel.settings_cancel_button.bind("<Enter>",
            lambda e: self.view.settings_panel.settings_cancel_button.configure(bg = CANCEL_BUTTON_FOCUS_COLOR))
        self.view.settings_panel.settings_cancel_button.bind("<Leave>",
            lambda e: self.view.settings_panel.settings_cancel_button.configure(bg = CANCEL_BUTTON_COLOR))
        self.view.settings_panel.settings_cancel_button.bind("<ButtonRelease-1>", self.cancel_settings)

    '''Main logic execution'''
    def run_system(self, event):
        print("3D scanning system start with default settings")
        self.view.process_settings(self.parent)
        self.model.run_motor_camera()

    '''Place holder for advanced options'''
    def run_settings(self, event):
        self.view.manage_settings(self.parent)

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
