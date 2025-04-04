import tensorflow as tf
from sionna import rt as srt
from sionna.rt.solver_paths import PathsTmpData as SionnaPathsTmpData
from sionna.constants import PI
from plyfile import PlyData
import numpy as np
from .params import RTParams
from ._C import NativeScene, NativeRisData
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
        self._itu_materials = itu.get_materials(dtype)
        self._num_material_labels = 0

    def set_model(self, ply, params):
        ply_data = self._get_ply_data(ply)
        if 'face' in ply_data:
            self.set_triangle_mesh(ply_data,
                                    voxel_size=params["voxel_size"],
                                    use_face_normals=params["use_face_normals"])
        else:
            self.set_point_cloud(ply_data,
                                  voxel_size=params["voxel_size"],
                                  point_radius=params["point_radius"],
                                  sdf_threshold=params["sdf_threshold"],
                                  lambda_distance=params["lambda_distance"])
        
    def set_triangle_mesh(self, mesh, voxel_size=0.0625, use_face_normals=False):
        ply_data = self._get_ply_data(mesh)
        vertices, normals, vertex_indices, face_properties, edges = self._get_triangle_mesh(ply_data)
        self._init_materials(np.max(face_properties["material"]), edges)
        self._init_edges(edges)
        return self._native_scene._set_triangle_mesh(vertices, normals, vertex_indices, face_properties, edges, voxel_size, use_face_normals)

    def set_point_cloud(self, cloud, voxel_size=0.0625, point_radius=0.015, sdf_threshold=0.003, lambda_distance=1000):
        ply_data = self._get_ply_data(cloud)
        point_cloud, edges = self._get_point_cloud(ply_data)
        self._init_materials(np.max(point_cloud["material"]), edges)
        self._init_edges(edges)
        self._native_scene._set_point_cloud(point_cloud, edges, voxel_size, point_radius, sdf_threshold, lambda_distance)

    def _get_ply_data(self, ply):
        if isinstance(ply, str):
            ply_data = PlyData.read(ply)
            self._geometry_path = ply
        elif isinstance(ply, PlyData):
            ply_data = ply
            self._geometry_path = None
        else:
            raise Exception("Input should be path to point cloud or 'plyfile.PlyData'.")
        return ply_data
        
    @property
    def cameras(self):
        return self._sionna_scene.cameras
    
    @property
    def frequency(self):
        return self._sionna_scene.frequency

    @frequency.setter
    def frequency(self, f):
        self._sionna_scene.frequency = f

    @property
    def temperature(self):
        return self._sionna_scene.temperature

    @temperature.setter
    def temperature(self, temperature):
        self._sionna_scene.temperature = temperature

    @property
    def bandwidth(self):
        return self._sionna_scene.bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        self._sionna_scene.bandwidth = bandwidth

    @property
    def thermal_noise_power(self):
        return self._sionna_scene.thermal_noise_power

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

    @property
    def radio_material_callable(self):
        return self._sionna_scene.radio_material_callable

    @radio_material_callable.setter
    def radio_material_callable(self, rm_callable):
        self._sionna_scene.radio_material_callable = rm_callable

    @property
    def scattering_pattern_callable(self):
        return self._sionna_scene.scattering_pattern_callable

    @scattering_pattern_callable.setter
    def scattering_pattern_callable(self, sp_callable):
        self._sionna_scene.scattering_pattern_callable = sp_callable

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

    def set_label_material(self, label_index, material):
        material.name = str(label_index)
        self._sionna_scene.radio_materials[material.name] = material

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
        return tf.convert_to_tensor(self._native_scene._size, dtype=self.dtype.real_dtype)
        
    @property
    def center(self):
        return tf.convert_to_tensor(self._native_scene._center, dtype=self.dtype.real_dtype)

    @property
    def num_material_labels(self):
        return self._num_material_labels

    def get(self, name):
       return self._sionna_scene.get(name)

    def add(self, item):
        self._sionna_scene.add(item)

    def remove(self, name):
        self._sionna_scene.remove(name)

    def compute_paths(self, params: RTParams):
        return self.compute_fields(self.trace_paths(params))

    def trace_paths(self, params: RTParams):
        txs, rxs = self._build_txs_rxs()
        ris_data = self._build_ris_data()
        result = self._native_scene._compute_sionna_path_data(params,
                                                              txs,
                                                              rxs,
                                                              ris_data)
        return self._convert_to_sionna(result)

    def coverage_map(self, params: RTParams, cell_size, height, rx_polarization="V"):
        txs = np.asarray([tx.position for tx in self.transmitters.values()], dtype=np.float32)
        ris_data = self._build_ris_data()

        synthetic_array_cache = self.synthetic_array
        tx_array_cache = self.tx_array
        rx_array_cache = self.rx_array
        
        self.tx_array = srt.PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern=tx_array_cache.antenna.patterns)

        self.rx_array = srt.PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="iso",
                                        polarization=rx_polarization)
        self.synthetic_array = True
        cm_data = self._native_scene._sionna_coverage_map(params, txs, cell_size, height, ris_data)
        path_tuple = self._convert_to_sionna(cm_data)
        
        a, _ = self.compute_fields(path_tuple).cir()
        path_gains = tf.transpose(tf.squeeze(tf.reduce_sum(tf.abs(a)**2, axis=-2), axis=[0,2,4,5]))
        
        indices = tf.convert_to_tensor(cm_data.rx_coords_2d, dtype=tf.int32)
        indices_expanded = tf.tile(indices[None, :, :], [txs.shape[0], 1, 1])

        path_gain_map = tf.zeros((txs.shape[0], cm_data.shape[0], cm_data.shape[1]))

        tx_indices = tf.range(txs.shape[0])[:, None]
        tx_indices = tf.tile(tx_indices, [1, indices.shape[0]])
        scatter_indices = tf.stack([tx_indices, indices_expanded[:, :, 0], indices_expanded[:, :, 1]], axis=-1)
        path_gain_map = tf.tensor_scatter_nd_update(path_gain_map, scatter_indices, path_gains)

        result = srt.CoverageMap(center=cm_data.center,
                                 orientation=[0,0,0],
                                 size=cm_data.size,
                                 cell_size=cm_data.cell_size,
                                 path_gain=path_gain_map,
                                 scene=self._sionna_scene,
                                 dtype=self.dtype)

        self.synthetic_array = synthetic_array_cache
        self.tx_array = tx_array_cache
        self.rx_array = rx_array_cache

        return result

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

    def _init_materials(self, materials, edges):
        max_mat_index = np.max(materials)
        if edges is not None:
            max_mat_index = max(max(max_mat_index, np.max(edges["material1"])), np.max(edges["material2"]))

        self._num_material_labels = max_mat_index + 1

        for i in range(self._num_material_labels):
            str_i = str(i)
            str_obj = str_i + "_obj"
            radio_material = srt.RadioMaterial(str_i)
            self._sionna_scene._scene_objects[str_obj] = srt.SceneObject(str_obj) #Workaround for object velocity.
            self._sionna_scene._scene_objects[str_obj].object_id = i
            self._sionna_scene._scene_objects[str_obj].scene = self._sionna_scene
            self._sionna_scene._scene_objects[str_obj].radio_material = radio_material
        
        #For RIS
        self._sionna_scene.add(self._itu_materials["itu_metal"])
        

    @property
    def _solver_paths(self):
        return self._sionna_scene._solver_paths

    def _init_edges(self, edges):
        self._solver_paths._wedges_e_hat = None
        self._solver_paths._wedges_length = None
        self._solver_paths._wedges_normals = None
        self._solver_paths._wedges_objects = None

        if edges is not None:
            normals = np.array((edges["normal1_x"], edges["normal1_y"], edges["normal1_z"], edges["normal2_x"], edges["normal2_y"], edges["normal2_z"])).T.reshape(-1, 2, 3)
            wedge_objects = np.array((edges["material1"], edges["material2"])).T.reshape(-1, 2)
            self._solver_paths._wedges_normals = tf.convert_to_tensor(normals)
            e_hat, edge_l = srt.utils.normalize(srt.utils.cross(self._solver_paths._wedges_normals[...,0,:],self._solver_paths._wedges_normals[...,1,:]))
            self._solver_paths._wedges_e_hat, self._solver_paths._wedges_length = e_hat, edge_l
            self._solver_paths._wedges_objects = tf.convert_to_tensor(wedge_objects, dtype=tf.int32)

    def _build_ris_data(self):
        ris_world_positions = []
        ris_object_ids = []
        ris_cell_object_ids = []
        ris_normals = []
        ris_centers = []
        ris_sizes = []

        for ris in self._sionna_scene.ris.values():
            ris_world_positions.append(ris.cell_world_positions)
            ris_object_ids.append(ris.object_id)
            ris_cell_object_ids.append(np.full((ris_world_positions[-1].shape[0],), ris.object_id))
            ris_normals.append(ris.world_normal)
            ris_centers.append(ris.position)
            ris_sizes.append(ris.size)

        ris_data = NativeRisData()
        if len(self._sionna_scene.ris):
            ris_data.cell_world_positions = np.concatenate(ris_world_positions, axis=0, dtype=np.float32)
            ris_data.object_ids = np.asarray(ris_object_ids, dtype=np.int32)
            ris_data.cell_object_ids = np.concatenate(ris_cell_object_ids, axis=0, dtype=np.int32)
            ris_data.normals = np.stack(ris_normals, dtype=np.float32)
            ris_data.centers = np.stack(ris_centers, axis=0, dtype=np.float32)
            ris_data.size = np.stack(ris_sizes, dtype=np.float32)
        return ris_data

    def _build_txs_rxs(self):
        tx_pos = [tx.position for tx in self.transmitters.values()]
        tx_pos = tf.stack(tx_pos, axis=0)
        rx_pos = [rx.position for rx in self.receivers.values()]
        rx_pos = tf.stack(rx_pos, axis=0)

        if not self.synthetic_array:
            rx_rot_mat, tx_rot_mat = self._solver_paths._get_tx_rx_rotation_matrices()
            rx_rel_ant_pos, tx_rel_ant_pos = self._solver_paths._get_antennas_relative_positions(rx_rot_mat, tx_rot_mat)

        if self.synthetic_array:
            sources = tx_pos
            targets = rx_pos
        else:
            sources = tf.expand_dims(tx_pos, axis=1) + tx_rel_ant_pos
            sources = tf.reshape(sources, [-1, 3])
            targets = tf.expand_dims(rx_pos, axis=1) + rx_rel_ant_pos
            targets = tf.reshape(targets, [-1, 3])
        
        return sources.numpy().astype(np.float32), targets.numpy().astype(np.float32)

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
                    ('material1', 'u4'), ('material2', 'u4')
                ]
                
                num_edges = edge['start_x'].shape[0]
                edge_result = np.empty(num_edges, dtype=edge_types)
                for te in edge_types:
                    if not te[0] in edge:
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

        edge_result = None
        if "edge" in ply_data:
            edge = ply_data["edge"]
            edge_types = [
                ('start_x', 'f4'), ('start_y', 'f4'), ('start_z', 'f4'),
                ('end_x', 'f4'), ('end_y', 'f4'), ('end_z', 'f4'),
                ('normal1_x', 'f4'), ('normal1_y', 'f4'), ('normal1_z', 'f4'),
                ('normal2_x', 'f4'), ('normal2_y', 'f4'), ('normal2_z', 'f4'),
                ('material1', 'u4'), ('material2', 'u4')
            ]
            
            num_edges = edge['start_x'].shape[0]
            edge_result = np.empty(num_edges, dtype=edge_types)
            for te in edge_types:
                if not te[0] in edge:
                    raise Exception(f"Field '{te[0]}' not found in edge elements.")
                edge_result[te[0]] = edge[te[0]]        

        return vertices, normals, triangle_indices, face_properties, edge_result

    def _convert_to_sionna(self, sionna_path_data):
        real_dtype = self.dtype.real_dtype
        sources = tf.convert_to_tensor(sionna_path_data.sources, dtype=real_dtype)
        targets = tf.convert_to_tensor(sionna_path_data.targets, dtype=real_dtype)

        ref_paths = srt.Paths(sources=sources, targets=targets, scene=self._sionna_scene)
        dif_paths = srt.Paths(sources=sources, targets=targets, scene=self._sionna_scene)
        sct_paths = srt.Paths(sources=sources, targets=targets, scene=self._sionna_scene)
        ris_paths = srt.Paths(sources=sources, targets=targets, scene=self._sionna_scene)

        tmp_ref_paths = SionnaPathsTmpData(sources=sources, targets=targets, dtype=self.dtype)
        tmp_dif_paths = SionnaPathsTmpData(sources=sources, targets=targets, dtype=self.dtype)
        tmp_sct_paths = SionnaPathsTmpData(sources=sources, targets=targets, dtype=self.dtype)
        tmp_ris_paths = SionnaPathsTmpData(sources=sources, targets=targets, dtype=self.dtype)

        ref_paths.types = srt.Paths.SPECULAR
        dif_paths.types = srt.Paths.DIFFRACTED
        sct_paths.types = srt.Paths.SCATTERED
        ris_paths.types = srt.Paths.RIS

        #Spec
        if sionna_path_data.max_link_paths(srt.Paths.SPECULAR) > 0:
            ref_paths.vertices = tf.convert_to_tensor(sionna_path_data.vertices(srt.Paths.SPECULAR), dtype=real_dtype)
            ref_paths.objects = tf.convert_to_tensor(sionna_path_data.objects(srt.Paths.SPECULAR), dtype=tf.int32)
            ref_paths.mask = tf.convert_to_tensor(sionna_path_data.mask(srt.Paths.SPECULAR), dtype=tf.bool)

            ref_paths.theta_t = tf.convert_to_tensor(sionna_path_data.theta_t(srt.Paths.SPECULAR), dtype=real_dtype)
            ref_paths.theta_r = tf.convert_to_tensor(sionna_path_data.theta_r(srt.Paths.SPECULAR), dtype=real_dtype)
            ref_paths.phi_t = tf.convert_to_tensor(sionna_path_data.phi_t(srt.Paths.SPECULAR), dtype=real_dtype)
            ref_paths.phi_r = tf.convert_to_tensor(sionna_path_data.phi_r(srt.Paths.SPECULAR), dtype=real_dtype)
            ref_paths.tau = tf.convert_to_tensor(sionna_path_data.tau(srt.Paths.SPECULAR), dtype=real_dtype)

            tmp_ref_paths.normals = tf.convert_to_tensor(sionna_path_data.normals(srt.Paths.SPECULAR), dtype=real_dtype)
            tmp_ref_paths.k_tx = tf.convert_to_tensor(sionna_path_data.k_tx(srt.Paths.SPECULAR), dtype=real_dtype)
            tmp_ref_paths.k_rx = tf.convert_to_tensor(sionna_path_data.k_rx(srt.Paths.SPECULAR), dtype=real_dtype)
            tmp_ref_paths.total_distance = tf.convert_to_tensor(sionna_path_data.total_distance(srt.Paths.SPECULAR), dtype=real_dtype)
            tmp_ref_paths.k_i = tf.convert_to_tensor(sionna_path_data.k_i(srt.Paths.SPECULAR), dtype=real_dtype)
            tmp_ref_paths.k_r = tf.convert_to_tensor(sionna_path_data.k_r(srt.Paths.SPECULAR), dtype=real_dtype)

        #Scat
        if sionna_path_data.max_link_paths(srt.Paths.SCATTERED) > 0:
            sct_paths.vertices = tf.convert_to_tensor(sionna_path_data.vertices(srt.Paths.SCATTERED), dtype=real_dtype)
            sct_paths.objects = tf.convert_to_tensor(sionna_path_data.objects(srt.Paths.SCATTERED), dtype=tf.int32)
            sct_paths.mask = tf.convert_to_tensor(sionna_path_data.mask(srt.Paths.SCATTERED), dtype=tf.bool)
            sct_paths.theta_t = tf.convert_to_tensor(sionna_path_data.theta_t(srt.Paths.SCATTERED), dtype=real_dtype)
            sct_paths.theta_r = tf.convert_to_tensor(sionna_path_data.theta_r(srt.Paths.SCATTERED), dtype=real_dtype)
            sct_paths.phi_t = tf.convert_to_tensor(sionna_path_data.phi_t(srt.Paths.SCATTERED), dtype=real_dtype)
            sct_paths.phi_r = tf.convert_to_tensor(sionna_path_data.phi_r(srt.Paths.SCATTERED), dtype=real_dtype)
            sct_paths.tau = tf.convert_to_tensor(sionna_path_data.tau(srt.Paths.SCATTERED), dtype=real_dtype)

            tmp_sct_paths.normals = tf.convert_to_tensor(sionna_path_data.normals(srt.Paths.SCATTERED), dtype=real_dtype)
            tmp_sct_paths.k_tx = tf.convert_to_tensor(sionna_path_data.k_tx(srt.Paths.SCATTERED), dtype=real_dtype)
            tmp_sct_paths.k_rx = tf.convert_to_tensor(sionna_path_data.k_rx(srt.Paths.SCATTERED), dtype=real_dtype)
            tmp_sct_paths.total_distance = tf.convert_to_tensor(sionna_path_data.total_distance(srt.Paths.SCATTERED), dtype=real_dtype)
            tmp_sct_paths.k_i = tf.convert_to_tensor(sionna_path_data.k_i(srt.Paths.SCATTERED), dtype=real_dtype)
            tmp_sct_paths.k_r = tf.convert_to_tensor(sionna_path_data.k_r(srt.Paths.SCATTERED), dtype=real_dtype)
            tmp_sct_paths.num_samples = 1.0 #To Nullify Sionnas Probability-based scaling
            tmp_sct_paths.scat_keep_prob = 4 * tf.cast(PI, self.dtype.real_dtype) #To Nullify Sionnas Probability-based scaling
            tmp_sct_paths.scat_last_objects = tf.convert_to_tensor(sionna_path_data.scat_last_objects(srt.Paths.SCATTERED))
            tmp_sct_paths.scat_last_vertices = tf.convert_to_tensor(sionna_path_data.scat_last_vertices(srt.Paths.SCATTERED))
            tmp_sct_paths.scat_last_k_i = tf.convert_to_tensor(sionna_path_data.scat_last_k_i(srt.Paths.SCATTERED))
            tmp_sct_paths.scat_k_s = tf.convert_to_tensor(sionna_path_data.scat_k_s(srt.Paths.SCATTERED))
            tmp_sct_paths.scat_last_normals = tf.convert_to_tensor(sionna_path_data.scat_last_normals(srt.Paths.SCATTERED))
            tmp_sct_paths.scat_src_2_last_int_dist = tf.convert_to_tensor(sionna_path_data.scat_src_2_last_int_dist(srt.Paths.SCATTERED))
            tmp_sct_paths.scat_2_target_dist = tf.convert_to_tensor(sionna_path_data.scat_2_target_dist(srt.Paths.SCATTERED))
            #Due to our discrete deterministic area-based scattering, we apply the appropriate scaling cos(theta_i)*dA into scat_2_target_dist
        
        #Diffraction
        if sionna_path_data.max_link_paths(srt.Paths.DIFFRACTED) > 0:
            dif_paths.vertices = tf.convert_to_tensor(sionna_path_data.vertices(srt.Paths.DIFFRACTED), dtype=real_dtype)
            dif_paths.objects = tf.convert_to_tensor(sionna_path_data.objects(srt.Paths.DIFFRACTED), dtype=tf.int32)
            dif_paths.mask = tf.convert_to_tensor(sionna_path_data.mask(srt.Paths.DIFFRACTED), dtype=tf.bool)
            dif_paths.theta_t = tf.convert_to_tensor(sionna_path_data.theta_t(srt.Paths.DIFFRACTED), dtype=real_dtype)
            dif_paths.theta_r = tf.convert_to_tensor(sionna_path_data.theta_r(srt.Paths.DIFFRACTED), dtype=real_dtype)
            dif_paths.phi_t = tf.convert_to_tensor(sionna_path_data.phi_t(srt.Paths.DIFFRACTED), dtype=real_dtype)
            dif_paths.phi_r = tf.convert_to_tensor(sionna_path_data.phi_r(srt.Paths.DIFFRACTED), dtype=real_dtype)
            dif_paths.tau = tf.convert_to_tensor(sionna_path_data.tau(srt.Paths.DIFFRACTED), dtype=real_dtype)
            
            tmp_dif_paths.normals = tf.gather(self._solver_paths._wedges_normals, dif_paths.objects)
            tmp_dif_paths.k_tx = tf.convert_to_tensor(sionna_path_data.k_tx(srt.Paths.DIFFRACTED), dtype=real_dtype)
            tmp_dif_paths.k_rx = tf.convert_to_tensor(sionna_path_data.k_rx(srt.Paths.DIFFRACTED), dtype=real_dtype)
            tmp_dif_paths.total_distance = tf.convert_to_tensor(sionna_path_data.total_distance(srt.Paths.DIFFRACTED), dtype=real_dtype)
            tmp_dif_paths.k_i = tf.convert_to_tensor(sionna_path_data.k_i(srt.Paths.DIFFRACTED), dtype=real_dtype)
            tmp_dif_paths.k_r = tf.convert_to_tensor(sionna_path_data.k_r(srt.Paths.DIFFRACTED), dtype=real_dtype)
        
        #RIS
        if sionna_path_data.max_link_paths(srt.Paths.RIS) > 0:
            ris_paths.vertices = tf.convert_to_tensor(sionna_path_data.vertices(srt.Paths.RIS), dtype=real_dtype)
            ris_paths.objects = tf.convert_to_tensor(sionna_path_data.objects(srt.Paths.RIS), dtype=tf.int32)
            ris_paths.mask = tf.convert_to_tensor(sionna_path_data.mask(srt.Paths.RIS), dtype=tf.bool)
            ris_paths.theta_t = tf.convert_to_tensor(sionna_path_data.theta_t(srt.Paths.RIS), dtype=real_dtype)
            ris_paths.theta_r = tf.convert_to_tensor(sionna_path_data.theta_r(srt.Paths.RIS), dtype=real_dtype)
            ris_paths.phi_t = tf.convert_to_tensor(sionna_path_data.phi_t(srt.Paths.RIS), dtype=real_dtype)
            ris_paths.phi_r = tf.convert_to_tensor(sionna_path_data.phi_r(srt.Paths.RIS), dtype=real_dtype)
            ris_paths.tau = tf.convert_to_tensor(sionna_path_data.tau(srt.Paths.RIS), dtype=real_dtype)

            tmp_ris_paths.normals = tf.convert_to_tensor(sionna_path_data.normals(srt.Paths.RIS), dtype=real_dtype)
            tmp_ris_paths.k_tx = tf.convert_to_tensor(sionna_path_data.k_tx(srt.Paths.RIS), dtype=real_dtype)
            tmp_ris_paths.k_rx = tf.convert_to_tensor(sionna_path_data.k_rx(srt.Paths.RIS), dtype=real_dtype)
            tmp_ris_paths.total_distance = tf.convert_to_tensor(sionna_path_data.total_distance(srt.Paths.RIS), dtype=real_dtype)
            tmp_ris_paths.k_i = tf.convert_to_tensor(sionna_path_data.k_i(srt.Paths.RIS), dtype=real_dtype)
            tmp_ris_paths.k_r = tf.convert_to_tensor(sionna_path_data.k_r(srt.Paths.RIS), dtype=real_dtype)
            
            tmp_ris_paths.cos_theta_i = tf.convert_to_tensor(sionna_path_data.cos_theta_i(srt.Paths.RIS), dtype=real_dtype)
            tmp_ris_paths.cos_theta_m = tf.convert_to_tensor(sionna_path_data.cos_theta_m(srt.Paths.RIS), dtype=real_dtype)
            tmp_ris_paths.distances = tf.convert_to_tensor(np.stack((sionna_path_data.distance_tx_ris(srt.Paths.RIS),
                                                                     sionna_path_data.distance_rx_ris(srt.Paths.RIS)), axis=0),
                                                                     dtype=real_dtype)

        return ref_paths, dif_paths, sct_paths, ris_paths, tmp_ref_paths, tmp_dif_paths, tmp_sct_paths, tmp_ris_paths