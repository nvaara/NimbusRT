import open3d as o3d
import numpy as np
from plyfile import PlyElement, PlyData

def create_ris_mesh(center, size, u, v):
    vertices = np.array([
        center - size[0] * u - size[1] * v,
        center + size[0] * u - size[1] * v,
        center + size[0] * u + size[1] * v,
        center - size[0] * u + size[1] * v
    ], dtype=np.float32)

    triangles = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def create_pcd_geom(cloud):
    pass

def create_trimesh_geom():
    pass

def render():
    pass

def is_trimesh(ply_data):
    return "vertex_indices" in ply_data

def main(args):
    ply_data = PlyData.read(args["model_path"])

    ris_pos = np.array([-1.4, 5.79,   1.0])
    ris_size = np.array([1.9986163, 1.9986163])*0.5
    ris_normal = np.array([1., 0., 0.])
    ris_u = np.array([0., 1., 0.])
    ris_v = np.array([0., 0., 1.])

    ris_mesh = create_ris_mesh(ris_pos, ris_size, ris_u, ris_v)

    if is_trimesh(ply_data):
        pass
        #Tri mesh model
    else: #PCD
        vertex = ply_data["vertex"]
        pos = np.array((vertex["x"], vertex["y"], vertex["z"])).T
        #normal = np.stack((vertex["nx"], vertex["ny"], vertex["nz"]))
        pc = o3d.geometry.PointCloud()        
        pc.points = o3d.utility.Vector3dVector(pos)
        #Load edge data
        
        o3d.visualization.draw_geometries([pc, ris_mesh])

if __name__ == "__main__":
    #Temp for testing
    args = {
        "model_path": "Data/SyntheticCorridor2Edges.ply",
        "radio_path_dir": None
    }
    main(args)