from gui_config import *
import open3d as o3d
import serial
from motor_camera.Motor import *
from motor_camera.Camera import *


class GuiModel(object):
    """description of class"""

    def __init__(self, updateFn=lambda n: None, additional_conds= lambda n: True, onSerialFail=lambda : print("Error! Please check hardware connectivity"), settings=None):
        if settings is None:
            macrostep_time=550
            baudrate=9600
            com='COM3'
        else:
            macrostep_time = int(settings['macrostep_time'])
            print(macrostep_time)
            baudrate = int(settings['baud'])
            com = settings['com']
        self.pathname = "data/"
        self.motor = Motor(macrostep_time=macrostep_time,total_macrosteps=200,baudrate=baudrate,com=str(com),onSerialFail=onSerialFail)
        self.camera = Camera()
        self.updateFn=updateFn
        self.cond = lambda num: self.camera.captureRGBD(num, show_image=False, path=self.pathname) and additional_conds

    def set_image_path(self, img_path):
        self.pathname = img_path

    def run_motor_camera(self, img_path=None):
        if img_path is None:
            img_path = self.pathname
        self.motor.fullRotation(cond=lambda num: self.camera.captureRGBD(num, show_image=True, path=img_path), updateFn=self.updateFn)


'''
View ply 3D models using given path
''' 
def view_ply(path_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.io.read_point_cloud(path_name)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

