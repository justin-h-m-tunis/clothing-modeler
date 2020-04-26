from gui_config import *
import open3d as o3d
import serial
from motor_camera import Motor, Camera

class GuiModel(object):
    """description of class"""

    def __init__(self):
        try:
            self.motor = Motor(macrostep_time=160,total_macrosteps=200,baudrate=9600,com='COM3',onSerialFail=lambda : print("Cannot find motor!"))
            self.camera = Camera()
        except:
            print("Error! Please check hardware connectivity")
        
    def run_motor_camera(self):
        self.motor.fullRotation(cond=lambda num: camera.captureRGBD(num, show_image=False))

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

