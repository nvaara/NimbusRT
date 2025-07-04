import nimbusrt as nrt
from sionna.rt import Transmitter, Receiver, PlanarArray

if __name__ == "__main__":
    scene = nrt.Scene()
    scene.set_point_cloud("corridor_point_cloud.ply", voxel_size=0.0625, point_radius=0.015, lambda_distance=100.0)
    
    params = nrt.RTParams(max_depth=5,
                          los=True,
                          reflection=True,
                          scattering=True,
                          diffraction=False,
                          ris=False,
                          refine_convergence_threshold=1e-4,
                          refine_angle_degrees_threshold=1.0,
                          refine_distance_threshold=0.01,
                          refine_max_iterations=50,
                          refine_max_correction_iterations=10
                          )

    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    scene.rx_array = scene.tx_array
    scene.synthetic_array = True
    scene.add(Transmitter(name="tx", position=[2.93, 5.79, 1.82]))
    scene.add(Receiver(name="rx", position=[-1.15, 8.7, 0.95]))

    for i in range(5):
        result_paths_tmp = scene.trace_paths(params)
        spec_paths, diff_paths, scat_paths, ris_paths, spec_paths_tmp, diff_paths_tmp, scat_paths_tmp, ris_paths_tmp = result_paths_tmp
        print(f"Paths: {scat_paths.objects.shape[3] + spec_paths.objects.shape[3]}")