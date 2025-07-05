import nimbusrt as nrt
from sionna.rt import Transmitter, Receiver, PlanarArray, LambertianPattern
from sionna.constants import PI, DIELECTRIC_PERMITTIVITY_VACUUM
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.channel import cir_to_ofdm_channel
from matplotlib.lines import Line2D
import matplotlib as mpl

class MaterialTrainer(tf.keras.layers.Layer):
    def __init__(self, scene, num_materials, train_relative_permittivity, train_conductivity, train_scattering_coefficient, train_xpd_coefficient):
        super(MaterialTrainer, self).__init__()

        self._scene = scene
        self._num_materials = num_materials
        self._relative_permittivity = tf.Variable(self.relative_permittivity_inverse_activation(2.0 * tf.ones(num_materials)), trainable=train_relative_permittivity, dtype=scene.dtype.real_dtype)
        self._conductivity = tf.Variable(self.conductivity_inverse_activation(0.1 * tf.ones(num_materials)), trainable=train_conductivity, dtype=scene.dtype.real_dtype)
        self._scattering_coefficient = tf.Variable(self.scattering_coefficient_inverse_activation(0.2 * tf.ones(num_materials)), trainable=train_scattering_coefficient, dtype=scene.dtype.real_dtype)
        self._xpd_coefficient = tf.Variable(self.xpd_coefficient_inverse_activation(0.0 * tf.ones(num_materials)), trainable=train_xpd_coefficient, dtype=scene.dtype.real_dtype)

    @property
    def num_materials(self):
        return self._num_materials

    @property
    def relative_permittivity(self):
        return 1.0 + self.relative_permittivity_activation(self._relative_permittivity)

    @property
    def conductivity(self):
        return self.conductivity_activation(self._conductivity)

    @property
    def scattering_coefficient(self):
        return self.scattering_coefficient_activation(self._scattering_coefficient)

    @property
    def xpd_coefficient(self):
        return self.xpd_coefficient_activation(self._xpd_coefficient)

    @relative_permittivity.setter
    def relative_permittivity(self, relative_permittivity):
        self._relative_permittivity = self.relative_permittivity_inverse_activation(relative_permittivity)

    @conductivity.setter
    def conductivity(self, conductivity):
        self._conductivity = self.conductivity_inverse_activation(conductivity)

    @scattering_coefficient.setter
    def scattering_coefficient(self, scattering_coefficient):
        self._scattering_coefficient = self.scattering_coefficient_inverse_activation(scattering_coefficient)

    @xpd_coefficient.setter
    def xpd_coefficient(self):
        self._xpd_coefficient = self.xpd_coefficient_inverse_activation(self._xpd_coefficient)

    @property
    def complex_relative_permittivity(self):
        rp = self.relative_permittivity
        c = self.conductivity
        omega = tf.cast(2. * PI * self._scene.frequency, rp.dtype.real_dtype)
        return tf.complex(rp, -tf.math.divide_no_nan(c, DIELECTRIC_PERMITTIVITY_VACUUM * omega))

    @staticmethod
    def relative_permittivity_activation(relative_permittivity):
        return 1.0 + tf.exp(relative_permittivity)

    def conductivity_activation(self, conductivity):
        return tf.exp(conductivity)

    def scattering_coefficient_activation(self, scattering_coefficient):
        return tf.sigmoid(scattering_coefficient)

    def xpd_coefficient_activation(self, xpd_coefficient):
        return tf.sigmoid(xpd_coefficient)

    def relative_permittivity_inverse_activation(self, relative_permittivity):
        return tf.math.log(relative_permittivity - 1)

    def conductivity_inverse_activation(self, conductivity):
        return tf.math.log(conductivity)

    def scattering_coefficient_inverse_activation(self, scattering_coefficient):
        return tf.math.log(scattering_coefficient/(1.0 - scattering_coefficient))

    def xpd_coefficient_inverse_activation(self, xpd_coefficient):
        return tf.math.log(xpd_coefficient/(1.0 - xpd_coefficient))

    def __call__(self, objects, vertices):
        valid_indices = tf.where(objects == -1, self.num_materials, objects)
        rp_tensor = tf.concat([self.complex_relative_permittivity, tf.constant([self.num_materials], dtype=self._scene.dtype)], axis=0)
        sc_tensor = tf.concat([self.scattering_coefficient, tf.constant([self.num_materials], dtype=self._scene.dtype.real_dtype)], axis=0)
        xpd_tensor = tf.concat([self.xpd_coefficient, tf.constant([self.num_materials], dtype=self._scene.dtype.real_dtype)], axis=0)
        rp = tf.gather(rp_tensor, valid_indices)
        sc = tf.gather(sc_tensor, valid_indices) 
        xpd = tf.gather(xpd_tensor, valid_indices)
        return rp, sc, xpd

def generate_gt_paths(scene, params, rx_pos):
    scene.get("rx").position = rx_pos
    return scene.trace_paths(params)

def freq_to_time(H_f, frequencies, tap_delays):
    exp_matrix = tf.exp(tf.complex(0.0, 2.0 * np.pi * tf.tensordot(tap_delays, frequencies, axes=0), H_f.dtype))
    return tf.linalg.matvec(H_f, exp_matrix) / tap_delays.shape[0]

def train(scene, max_iters, bandwidth=1500e6, num_samples=129):
    paths = []
    H_f_gts = []
    h_t_gts = []
    num_paths = []
    h_t_train = []
    tap_ind = tf.range(0, num_samples, dtype=tf.float32)
    tap_delays = tap_ind / bandwidth
    params = nrt.RTParams(max_depth=2,
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

    frequencies = scene.frequency + tf.range(-(num_samples - 1) / 2, (num_samples - 1) / 2 + 1, dtype=tf.float32) * bandwidth / (num_samples - 1)
    rx_positions =[
        np.array([0.0, 2.0, 0.3]),
        np.array([3.35, 2.5, 0.4]),
        np.array([-0.6, -0.1, 0.35]),
        np.array([1.65, 2.8, 0.25]),
        np.array([1.85, -0.7, 0.35]),
        np.array([4.85, -0.7, 0.35])
    ]
    for i in range(len(rx_positions)):
        paths.append(generate_gt_paths(scene, params, rx_positions[i]))
        fields = scene.compute_fields(paths[-1], check_scene=False, scat_random_phases=False)
        fields.normalize_delays = False
        gt_a, gt_tau = fields.cir()
        num_paths.append(tf.reshape(gt_a, [-1]).shape[0])
        print(f"Path coefficients: {gt_a.shape}")
        H_f_gts.append(cir_to_ofdm_channel(frequencies, gt_a, gt_tau))
        h_t_gts.append(freq_to_time(H_f_gts[-1], frequencies, tap_delays))
    h_t_train = h_t_gts.copy()
    print("Generated GT paths.")

    trainer = MaterialTrainer(scene=scene,
                              num_materials=scene.num_material_labels,
                              train_relative_permittivity=True,
                              train_conductivity=True,
                              train_scattering_coefficient=True,
                              train_xpd_coefficient=False)
    
    scene.radio_material_callable = trainer
    optimizer = tf.keras.optimizers.Adam(1e-2)

    relative_permittivity_iter = np.zeros((max_iters, scene.num_material_labels))
    conductivity_iter = np.zeros((max_iters, scene.num_material_labels))
    scattering_coefficient_iter = np.zeros((max_iters, scene.num_material_labels))

    for iter in range(max_iters):
        set_index = np.random.randint(0, len(paths))
        h_t_gt = h_t_gts[set_index]

        with tf.GradientTape() as tape:
            fields = scene.compute_fields(paths[set_index], check_scene=False, scat_random_phases=False)
            fields.normalize_delays = False
            a, tau = fields.cir()
            H_f = cir_to_ofdm_channel(frequencies, a, tau)
            h_t = freq_to_time(H_f, frequencies, tap_delays)
            h_t_train[set_index] = h_t
            loss = tf.reduce_mean(tf.abs(h_t-h_t_gt)**2) / tf.reduce_mean((tf.abs(h_t_gt)**2))
            
            if iter % 100 == 0:
                print(f"Iteration {iter} Loss: {loss}")
            
            grads = tape.gradient(loss, tape.watched_variables(), unconnected_gradients=tf.UnconnectedGradients.ZERO)
            optimizer.apply_gradients(zip(grads, tape.watched_variables()))
            relative_permittivity_iter[iter] = trainer.relative_permittivity.numpy()
            conductivity_iter[iter] = trainer.conductivity.numpy()
            scattering_coefficient_iter[iter] = trainer.scattering_coefficient.numpy()
    
    return relative_permittivity_iter.T, conductivity_iter.T, scattering_coefficient_iter.T, np.array(h_t_train), np.array(h_t_gts), tap_delays.numpy(), np.array(num_paths)


def generate_experiment_ply(scene, point_cloud, output_file_name="room0_point_cloud.ply"):
    ply_data = scene._get_ply_data(point_cloud)
    point_cloud, edges = scene._get_point_cloud(ply_data)
    start = 0
    step = 6
    for i in range(5):
        indices = np.nonzero((point_cloud["material"] >= start) & (point_cloud["material"] < start + step))
        point_cloud["material"][indices] = i
        start += 6
    
    from plyfile import PlyData, PlyElement
    print(point_cloud.__class__)
    el = PlyElement.describe(point_cloud, "vertex")
    PlyData([el]).write(output_file_name)


if __name__ == "__main__":
    scene = nrt.Scene()
    scene.set_point_cloud("room0_point_cloud.ply", voxel_size=0.0625, point_radius=0.015, lambda_distance=100.0)
    scene.frequency=8.0e9

    mat_list = [
        "itu_concrete",
        "itu_plasterboard",
        "itu_wood",
        "itu_brick",
        "itu_chipboard"
    ]
    max_iters = 5000
    gt_permittivity = np.zeros((len(mat_list), max_iters))
    gt_conductivity = np.zeros((len(mat_list), max_iters))
    gt_scattering = np.zeros((len(mat_list), max_iters))

    for mat_label in range(scene.num_material_labels):
        scene.set_itu_material_for_label(mat_label, mat_list[mat_label], scattering_coefficient = 0.05 + 0.05 * (mat_label), xpd_coefficient=0.0, scattering_pattern=LambertianPattern())
        mat = scene.get_label_material(mat_label)
        gt_permittivity[mat_label]= np.full(max_iters, mat.relative_permittivity)
        gt_conductivity[mat_label]= np.full(max_iters, mat.conductivity)
        gt_scattering[mat_label]= np.full(max_iters, mat.scattering_coefficient)

    for mat in scene.radio_materials.items():
        key, val = mat
        print(f"{val.relative_permittivity} {val.conductivity}")

    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    
    scene.rx_array = scene.tx_array
    scene.synthetic_array = True
    scene.add(Transmitter(name="tx", position=[6.5, 0.5, 0.0]))
    scene.add(Receiver(name="rx", position=[0.0, 2.0, 0.5]))

    #Compute GT coefficients
    rel, con, scat, h_t_train, h_t_gt, tap_delays, num_paths = train(scene, max_iters, bandwidth=1500e6, num_samples=129)
    np.savez('train_result.npz',
                gt_conductivity=gt_conductivity,
                gt_permittivity=gt_permittivity,
                gt_scattering=gt_scattering,
                train_permittivity=rel,
                train_conductivity=con,
                train_scattering=scat,
                h_t_train=h_t_train,
                h_t_gt=h_t_gt,
                tap_delays=tap_delays,
                num_paths=num_paths
                )

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 14
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    epochs = np.arange(max_iters)
    colors = plt.cm.tab10.colors

    for i in range(len(mat_list)):
        color = colors[i % len(colors)]
        ax[0].plot(epochs, gt_permittivity[i], color=color, label="True Relative Permittivity", linestyle="--")
        ax[0].plot(epochs, rel[i], color=color, label="Learned Relative Permittivity", linestyle="-")
        ax[0].set_ylabel("Relative Permittivity")
        ax[0].set_xlabel("Number of Iterations")
        ax[1].plot(epochs, gt_conductivity[i], color=color, label="True Conductivity", linestyle="--")
        ax[1].plot(epochs, con[i], color=color, label="Learned Conductivity", linestyle="-")
        ax[1].set_ylabel("Conductivity")
        ax[1].set_xlabel("Number of Iterations")
        ax[1].set_ylim(0, 0.5)
        ax[2].plot(epochs, gt_scattering[i], color=color, label="Ground Truth Scattering Coefficient", linestyle="--")
        ax[2].plot(epochs, scat[i], color=color, label="Learned Scattering Coefficient", linestyle="-")
        ax[2].set_ylabel("Scattering Coefficient")
        ax[2].set_xlabel("Number of Iterations")

    ax[3].plot(tap_delays * 1e9, 10 * np.log10(np.abs(h_t_train[4])**2).flatten(), linestyle="-")
    ax[3].plot(tap_delays * 1e9, 10 * np.log10(np.abs(h_t_gt[4])**2).flatten(), linestyle="--")
    ax[3].set_xlabel("Time Delay (ns)")
    ax[3].set_ylabel("$|h(t)|^2$ (dB)")

    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', label='Ground Truth'),
        Line2D([0], [0], color='black', linestyle='-', label='Learned')
    ]
    ax[1].legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()
