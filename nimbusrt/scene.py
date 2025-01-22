import tensorflow as tf
from sionna import rt as srt
from sionna.rt.solver_paths import PathsTmpData as SionnaPathsTmpData
from .solver_paths import NimbusSolverPaths
from sionna.constants import PI

from plyfile import PlyData
import numpy as np
from .params import RTParams
from ._C import NativeScene
from . import itu


class Scene():
    _instance = None
    _native_scene = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            instance = object.__new__(cls)
            instance._native_scene = NativeScene()
            cls._instance = instance
        return cls._instance

    def __init__(self, dtype=tf.complex64):
        self._sionna_scene = srt.Scene("__empty__", dtype=dtype)
        self._sionna_scene._clear()
        #self._sionna_scene._solver_paths = NimbusSolverPaths(self._sionna_scene, dtype=dtype)
        self._itu_materials = itu.get_materials(dtype)
        self._sionna_scene.radio_material_callable = self

    def set_triangle_mesh(self, mesh, voxel_size=0.0625, use_face_normals=False):
        if isinstance(mesh, str):
            ply_data = PlyData.read(mesh)
            self._geometry_path = mesh
        elif isinstance(PlyData):
            ply_data = mesh
            self._geometry_path = None
        else:
            raise Exception("Input should be path to point cloud or 'plyfile.PlyData'.")
        
        vertices, normals, vertex_indices, face_properties = self._get_triangle_mesh(ply_data)
        self._init_materials(np.max(face_properties["material"]) + 1)
        return self._native_scene._set_triangle_mesh(vertices, normals, vertex_indices, face_properties, voxel_size, use_face_normals)

    def set_point_cloud(self, cloud, voxel_size=0.0625, aabb_bias=0.01):
        if isinstance(cloud, str):
            ply_data = PlyData.read(cloud)
            self._geometry_path = cloud
        elif isinstance(PlyData):
            ply_data = cloud
            self._geometry_path = None
        else:
            raise Exception("Input should be path to point cloud or 'plyfile.PlyData'.")
        
        point_cloud = self._get_point_cloud(ply_data)
        self._init_materials(np.max(point_cloud["material"]) + 1)
        self._native_scene._set_point_cloud(point_cloud, voxel_size, aabb_bias)

    @property
    def cameras(self):
        return dict(self._sionna_scene.cameras)
    
    @property
    def frequency(self):
        return self._sionna_scene.frequency

    @frequency.setter
    def frequency(self, f):
        self._sionna_scene.frequency = f

    @property
    def temperature(self):
        raise NotImplementedError()

    @temperature.setter
    def temperature(self, v):
        raise NotImplementedError()

    @property
    def bandwidth(self):
        raise NotImplementedError()

    @bandwidth.setter
    def bandwidth(self, v):
        raise NotImplementedError()

    @property
    def thermal_noise_power(self):
        raise NotImplementedError()

    @property
    def transmitters(self):
        return self._sionna_scene.transmitters

    @property
    def receivers(self):
        return self._sionna_scene.receivers

    @property
    def wavelength(self):
        return self._sionna_scene.wavelength

    @property
    def wavenumber(self):
        return self._sionna_scene.wavenumber

    @property
    def synthetic_array(self):
        return self._sionna_scene.synthetic_array

    @synthetic_array.setter
    def synthetic_array(self, value):
        self._sionna_scene.synthetic_array = value

    @property
    def dtype(self):
        return self._sionna_scene.dtype

    @property
    def transmitters(self):
        return self._sionna_scene.transmitters

    @property
    def receivers(self):
        return self._sionna_scene.receivers

    @property
    def ris(self):
        return self._sionna_scene.ris

    @property
    def radio_materials(self):
        return self._sionna_scene.radio_materials

    def set_itu_material_for_label(self, label_index, itu_name, scattering_coefficient=0.0, xpd_coefficient=0.0, scattering_pattern=None):
        itu_mat = self._itu_materials[itu_name]
        label_mat = self._sionna_scene.radio_materials[str(label_index)]
        label_mat.scattering_coefficient = scattering_coefficient
        label_mat.xpd_coefficient = xpd_coefficient
        label_mat.scattering_pattern = srt.LambertianPattern(dtype=self.dtype) if scattering_pattern is None else scattering_pattern
        label_mat.frequency_update_callback = itu_mat.frequency_update_callback
        label_mat.frequency_update()

    def get_label_material(self, label_index):
        return self._sionna_scene.radio_materials[str(label_index)]

    @property
    def objects(self):
        return self._sionna_scene.objects

    @property
    def tx_array(self):
        return self._sionna_scene.tx_array

    @tx_array.setter
    def tx_array(self, array):
        self._sionna_scene.tx_array = array

    @property
    def rx_array(self):
        return self._sionna_scene.rx_array

    @rx_array.setter
    def rx_array(self, array):
        self._sionna_scene.rx_array = array

    @property
    def size(self):
        raise NotImplementedError()
        
    @property
    def center(self):
        raise NotImplementedError()

    def get(self, name):
       return self._sionna_scene.get(name)

    def add(self, item):
        self._sionna_scene.add(item)

    def remove(self, name):
        self._sionna_scene.remove(name)

    def compute_paths(self, params: RTParams):
        return self.compute_fields(self.trace_paths(params))

    def trace_paths(self, params: RTParams):
        txs = list(self.transmitters.values())
        rxs = list(self.receivers.values())
        tx_positions = np.empty((len(txs), 3), dtype=np.float32)
        rx_positions = np.empty((len(rxs), 3), dtype=np.float32)
        for idx, v in enumerate(self.transmitters.values()):
            tx_positions[idx] = v.position
        for idx, v in enumerate(self.receivers.values()):
            rx_positions[idx] = v.position

        result = self._native_scene._compute_sionna_path_data(params,
                                                              tx_positions,
                                                              rx_positions)
        return self._convert_to_sionna(result)

    def compute_fields(self, path_tuple):
        return self._sionna_scene.compute_fields(*path_tuple)
        

    def __call__(self, objects, vertices):
        radio_mats = self._sionna_scene.radio_materials
        relative_permittivity  = tf.zeros(objects.shape, dtype=self.dtype)
        scattering_coefficient = tf.zeros(objects.shape, dtype=self.dtype.real_dtype)
        xpd_coefficient = tf.zeros(objects.shape, dtype=self.dtype.real_dtype)

        for rm_key, rm in radio_mats.items():
            if not rm_key.isdigit():
                continue
            indices_to_update = tf.where(objects == int(rm_key))
            relative_permittivity = tf.tensor_scatter_nd_update(relative_permittivity,
                                                                indices_to_update,
                                                                tf.fill([tf.shape(indices_to_update)[0]], rm.complex_relative_permittivity))
            scattering_coefficient = tf.tensor_scatter_nd_update(scattering_coefficient,
                                                                 indices_to_update,
                                                                 tf.fill([tf.shape(indices_to_update)[0]], rm.scattering_coefficient))
            xpd_coefficient = tf.tensor_scatter_nd_update(xpd_coefficient,
                                                          indices_to_update,
                                                          tf.fill([tf.shape(indices_to_update)[0]], rm.xpd_coefficient))

        return relative_permittivity, scattering_coefficient, xpd_coefficient

    def _init_materials(self, material_count):
        for i in range(material_count):
            str_i = str(i)
            self._sionna_scene.add(srt.RadioMaterial(str_i))
            self._sionna_scene._scene_objects[str(i)] = srt.SceneObject(str_i) #Workaround for object velocity.
            self._sionna_scene._scene_objects[str(i)].object_id = i
            self._sionna_scene._scene_objects[str(i)]._radio_material = self._sionna_scene.get(str_i)

    def _get_point_cloud(self, ply_data: PlyData):
            types = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("nx", "f4"),
                ("ny", "f4"),
                ("nz", "f4"),
                ("label", "u4"),
                ("material", "u4"),
            ]
            vertex = ply_data["vertex"]
            point_cloud = np.empty(vertex['x'].shape[0], dtype=types)
            
            for t in types:
                if not t[0] in vertex:
                    raise Exception(f"Field '{t[0]}' not found in point cloud.")
                point_cloud[t[0]] = vertex[t[0]]
            
            edge_result = None
            if "edge" in ply_data:
                edge = ply_data["edge"]
                edge_types = [
                    ('start_x', 'f4'), ('start_y', 'f4'), ('start_z', 'f4'),
                    ('end_x', 'f4'), ('end_y', 'f4'), ('end_z', 'f4'),
                    ('normal1_x', 'f4'), ('normal1_y', 'f4'), ('normal1_z', 'f4'),
                    ('normal2_x', 'f4'), ('normal2_y', 'f4'), ('normal2_z', 'f4'),
                    ('material1', 'u4'), ('material2', 'u4'),
                    ('edge_id')
                ]
                
                num_edges = edge['start_x'].shape[0]
                edge_result = np.empty(num_edges, dtype=types)
                edge_result['edge_id'] = np.arange(num_edges)
                for te in edge_types:
                    if not te[0] is 'edge_id' and not te[0] in edge:
                        raise Exception(f"Field '{te[0]}' not found in edge elements.")
                    edge_result[te[0]] = edge[te[0]]

            return point_cloud, edge_result

    def _get_triangle_mesh(self, ply_data: PlyData):
        ply_vertex_data = ply_data['vertex']
        vertices = np.empty(ply_vertex_data['x'].shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        normals = np.empty(ply_vertex_data['x'].shape[0], dtype=[("nx", "f4"), ("ny", "f4"), ("nz", "f4")])
        
        vertices['x'], vertices['y'], vertices['z'] = ply_vertex_data['x'], ply_vertex_data['y'], ply_vertex_data['z']
        normals['nx'], normals['ny'], normals['nz'] = ply_vertex_data['nx'], ply_vertex_data['ny'], ply_vertex_data['nz']

        face_data = ply_data['face']
        vertex_indices = face_data['vertex_indices']
        if vertex_indices[0].shape[0] == 4:
            n_quads = vertex_indices.shape[0]
            quad_indices = np.vstack(vertex_indices)
            triangle_indices = np.zeros(2 * n_quads, dtype=[('i0', 'u4'), ('i1', 'u4'), ('i2', 'u4')])
            tr = np.vstack((quad_indices[:, (0, 1, 2)], quad_indices[:, (2, 3, 0)]))
            triangle_indices['i0'] = tr[:, 0]
            triangle_indices['i1'] = tr[:, 1]
            triangle_indices['i2'] = tr[:, 2]
            a1 = np.arange(vertex_indices.shape[0], dtype=np.uint32)
            computed_labels = np.concatenate((a1, a1))
        elif vertex_indices[0].shape[0] == 3:
            triangle_indices = np.empty(vertex_indices.shape[0], dtype=[("i0", "u4"), ("i1", "u4"), ("i2", "u4")])
            stacked_data = np.vstack(vertex_indices)
            triangle_indices["i0"], triangle_indices["i1"], triangle_indices["i2"] = stacked_data[:, 0], stacked_data[:, 1], stacked_data[:, 2] 
            computed_labels = np.arange(triangle_indices.shape[0], dtype=np.uint32)

        v = np.stack((vertices['x'], vertices['y'], vertices['z']), axis=-1)
        f = np.stack((triangle_indices['i0'], triangle_indices['i1'], triangle_indices['i2']), axis=-1)
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)

        face_properties = np.empty(triangle_indices.shape[0], dtype=[("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("label", "u4"), ("material", "u4")])
        face_properties["nx"], face_properties["ny"], face_properties["nz"] = face_normals[:, 0], face_normals[:, 1], face_normals[:, 2]
        face_properties["label"] = face_data["label"] if "label" in face_data else computed_labels
        face_properties["material"] = face_data["material"] if "material" in face_data else face_properties["label"]

        return vertices, normals, triangle_indices, face_properties

    def _convert_to_sionna(self, sionna_path_data):
        sources = tf.convert_to_tensor(sionna_path_data.sources, dtype=tf.complex64.real_dtype)
        targets = tf.convert_to_tensor(sionna_path_data.targets, dtype=tf.complex64.real_dtype)

        ref_paths = srt.Paths(sources=sources, targets=targets, scene=self._sionna_scene)
        dif_paths = srt.Paths(sources=sources, targets=targets, scene=self._sionna_scene)
        sct_paths = srt.Paths(sources=sources, targets=targets, scene=self._sionna_scene)
        ris_paths = srt.Paths(sources=sources, targets=targets, scene=self._sionna_scene)

        tmp_ref_paths = SionnaPathsTmpData(sources=sources, targets=targets, dtype=self._sionna_scene.dtype)
        tmp_dif_paths = SionnaPathsTmpData(sources=sources, targets=targets, dtype=self._sionna_scene.dtype)
        tmp_sct_paths = SionnaPathsTmpData(sources=sources, targets=targets, dtype=self._sionna_scene.dtype)
        tmp_ris_paths = SionnaPathsTmpData(sources=sources, targets=targets, dtype=self._sionna_scene.dtype)

        ref_paths.types = srt.Paths.SPECULAR
        dif_paths.types = srt.Paths.DIFFRACTED
        sct_paths.types = srt.Paths.SCATTERED
        ris_paths.types = srt.Paths.RIS

        #Spec
        ref_paths.vertices = tf.convert_to_tensor(sionna_path_data.vertices(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        ref_paths.objects = tf.convert_to_tensor(sionna_path_data.objects(srt.Paths.SPECULAR), dtype=tf.int32)
        ref_paths.mask = tf.convert_to_tensor(sionna_path_data.mask(srt.Paths.SPECULAR), dtype=tf.bool)
        ref_paths.theta_t = tf.convert_to_tensor(sionna_path_data.theta_t(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        ref_paths.theta_r = tf.convert_to_tensor(sionna_path_data.theta_r(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        ref_paths.phi_t = tf.convert_to_tensor(sionna_path_data.phi_t(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        ref_paths.phi_r = tf.convert_to_tensor(sionna_path_data.phi_r(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        ref_paths.tau = tf.convert_to_tensor(sionna_path_data.tau(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)

        tmp_ref_paths.normals = tf.convert_to_tensor(sionna_path_data.normals(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        tmp_ref_paths.k_tx = tf.convert_to_tensor(sionna_path_data.k_tx(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        tmp_ref_paths.k_rx = tf.convert_to_tensor(sionna_path_data.k_rx(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        tmp_ref_paths.total_distance = tf.convert_to_tensor(sionna_path_data.total_distance(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        tmp_ref_paths.k_i = tf.convert_to_tensor(sionna_path_data.k_i(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)
        tmp_ref_paths.k_r = tf.convert_to_tensor(sionna_path_data.k_r(srt.Paths.SPECULAR), dtype=tf.complex64.real_dtype)

        #Scat
        sct_paths.vertices = tf.convert_to_tensor(sionna_path_data.vertices(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        sct_paths.objects = tf.convert_to_tensor(sionna_path_data.objects(srt.Paths.SCATTERED), dtype=tf.int32)
        sct_paths.mask = tf.convert_to_tensor(sionna_path_data.mask(srt.Paths.SCATTERED), dtype=tf.bool)
        sct_paths.theta_t = tf.convert_to_tensor(sionna_path_data.theta_t(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        sct_paths.theta_r = tf.convert_to_tensor(sionna_path_data.theta_r(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        sct_paths.phi_t = tf.convert_to_tensor(sionna_path_data.phi_t(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        sct_paths.phi_r = tf.convert_to_tensor(sionna_path_data.phi_r(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        sct_paths.tau = tf.convert_to_tensor(sionna_path_data.tau(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)

        tmp_sct_paths.normals = tf.convert_to_tensor(sionna_path_data.normals(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        tmp_sct_paths.k_tx = tf.convert_to_tensor(sionna_path_data.k_tx(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        tmp_sct_paths.k_rx = tf.convert_to_tensor(sionna_path_data.k_rx(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        tmp_sct_paths.total_distance = tf.convert_to_tensor(sionna_path_data.total_distance(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        tmp_sct_paths.k_i = tf.convert_to_tensor(sionna_path_data.k_i(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        tmp_sct_paths.k_r = tf.convert_to_tensor(sionna_path_data.k_r(srt.Paths.SCATTERED), dtype=tf.complex64.real_dtype)
        tmp_sct_paths.num_samples = 1.0 #To Nullify Sionnas Probability-based scaling
        tmp_sct_paths.scat_keep_prob = 4 * tf.cast(PI, self.dtype.real_dtype) #To Nullify Sionnas Probability-based scaling
        tmp_sct_paths.scat_last_objects = tf.convert_to_tensor(sionna_path_data.scat_last_objects(srt.Paths.SCATTERED))
        tmp_sct_paths.scat_last_vertices = tf.convert_to_tensor(sionna_path_data.scat_last_vertices(srt.Paths.SCATTERED))
        tmp_sct_paths.scat_last_k_i = tf.convert_to_tensor(sionna_path_data.scat_last_k_i(srt.Paths.SCATTERED))
        tmp_sct_paths.scat_k_s = tf.convert_to_tensor(sionna_path_data.scat_k_s(srt.Paths.SCATTERED))
        tmp_sct_paths.scat_last_normals = tf.convert_to_tensor(sionna_path_data.scat_last_normals(srt.Paths.SCATTERED))
        tmp_sct_paths.scat_src_2_last_int_dist = tf.convert_to_tensor(sionna_path_data.scat_src_2_last_int_dist(srt.Paths.SCATTERED))
        tmp_sct_paths.scat_2_target_dist = tf.convert_to_tensor(sionna_path_data.scat_2_target_dist(srt.Paths.SCATTERED))
        #Due to our discrete, deterministic area-based scattering, we apply the appropriate scaling cos(theta_i)*dA into scat_2_target_dist
        
        #Diffraction
        #dif_paths.objects = tf.range(0, dif_paths.mask.shape[2])
        #self._sionna_scene.solver_paths.wedges_e_hat
        '''
        _wedges_e_hat needs to be modified
        _wedges_normals is the normals

        objects [num_target, num_sources will be used to index 
        _wedges_objects will contain the object indices [num_targets, num_sources, 2]

        wedge_indices = path.objects

        normals = self._wedges
        '''
        #RIS

        return ref_paths, dif_paths, sct_paths, ris_paths, tmp_ref_paths, tmp_dif_paths, tmp_sct_paths, tmp_ris_paths