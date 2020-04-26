from gui_config import *
import open3d as o3d
import serial
from motor_camera.Motor import *
from motor_camera.Camera import *
# from call_back import *


class GuiModel(object):
    """description of class"""

    def __init__(self):
        self.pathname = "data/"
        self.motor = Motor(macrostep_time=160,total_macrosteps=200,baudrate=9600,com='COM3',onSerialFail=lambda : print("Error! Please check hardware connectivity"))
        self.camera = Camera()

    def set_image_path(self, img_path):
        self.pathname = img_path

    def run_motor_camera(self, control_instance):
        self.motor.fullRotation(control_instance, cond=lambda num: self.camera.captureRGBD(num, show_image=False, path=self.pathname))

    # def run_call_back(self, control_instance):
    #     cb = CallBack(control_instance)

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

