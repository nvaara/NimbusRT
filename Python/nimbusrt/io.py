from .edge import Edge, EdgeHelper
import json
from plyfile import PlyData


def read_edges_from_json(filename):
    with open(filename) as f:
        return [Edge(**item) for item in json.loads(f.read())]


def write_edges_to_json(filename, edges):
    json_string = json.dumps(
        [
            EdgeHelper(ob.start, ob.end, ob.normal0, ob.normal1, 0, 0).__dict__
            for ob in edges
        ]
    )
    with open(filename, "w") as f:
        f.write(json_string)


def load_point_cloud(filename, required_data_types=None):
    ply_data = PlyData.read(filename)
    v = ply_data["vertex"]

    if required_data_types is None:
        return ply_data

    for t in required_data_types:
        assert (
            t[0] in v
        ), f"Data type {t[0]} does not exist in point cloud file {filename}."
        if t[0] not in v:
            return None

    return ply_data
