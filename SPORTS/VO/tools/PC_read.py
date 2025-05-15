import open3d as o3d

pcd = o3d.io.read_point_cloud('/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/PC/cloudpoint/rgb.ply')

o3d.visualization.draw_geometries([pcd])