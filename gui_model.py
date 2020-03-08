from gui_config import *
import open3d as o3d

class GuiModel(object):
    """description of class"""

    def __init__(self):
        pass

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

