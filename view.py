import open3d as o3d

def custom_draw_geometry(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)
    
pcd = o3d.io.read_point_cloud("C:/Users/shuyo/OneDrive/HW/24671/open3d_test/examples/Python/ReconstructionSystem/dataset/realsense/scene/integrated.ply")

custom_draw_geometry(pcd)
# custom_draw_geometry_with_rotation(pcd)


