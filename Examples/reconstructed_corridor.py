import nimbusrt as nrt

def reconstructed_corridor_input_params(num_interactions = 2):
    input_data = nrt.InputData()
    input_data.scene_settings.frequency = 60e9
    input_data.scene_settings.voxel_size = 0.5
    input_data.scene_settings.voxel_division_factor = 2
    input_data.scene_settings.subvoxel_division_factor = 4
    input_data.scene_settings.received_path_buffer_size = 25000
    input_data.scene_settings.propagation_path_buffer_size = 100000
    input_data.scene_settings.propagation_buffer_size_increase_factor = 2.0
    input_data.scene_settings.sample_radius_coarse = 0.015
    input_data.scene_settings.sample_radius_refine = 0.01
    input_data.scene_settings.variance_factor_coarse = 2.0
    input_data.scene_settings.variance_factor_refine = 2.0
    input_data.scene_settings.sdf_threshold_coarse = 0.0015
    input_data.scene_settings.sdf_threshold_refine = 0.001
    input_data.scene_settings.num_iterations = 2000
    input_data.scene_settings.delta = 1e-4
    input_data.scene_settings.alpha = 0.4
    input_data.scene_settings.beta = 0.4
    input_data.scene_settings.angle_threshold = 25.0
    input_data.scene_settings.distance_threshold = 0.02
    input_data.scene_settings.block_size = 32
    input_data.scene_settings.num_coarse_paths_per_unique_route = 100
    input_data.num_interactions = num_interactions
    input_data.num_diffractions = 0
    return input_data

if __name__ == "__main__":
    scene = nrt.Scene()
    scene.set_point_cloud("Data/ReconstructedCorridor.ply")
    scene.add_transmitter("tx0", [2.93, 5.79, 1.82])
    scene.add_receiver("rx0", [-1.15, 8.7, 0.95])
    scene.compute_paths(reconstructed_corridor_input_params(num_interactions=2))
    print(f"Found {len(scene.path_storage['tx0']['rx0'])} paths.")
