import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tensorflow as tf
from sionna.rt.utils import watt_to_dbm

class CoverageMap3D:
    def __init__(self, dimensions, scene, cm_voxel_size, path_gain, cell_centers):
        self._scene = scene
        self._dimensions = dimensions
        self._cm_voxel_size = cm_voxel_size
        self._path_gain = path_gain
        self._cell_centers = cell_centers
        self._scene_min = scene.center - scene.size * 0.5
        self._tx_name_2_ind = {}
        self._transmitters = []

        for tx_ind, (tx_name, tx) in enumerate(self._scene.transmitters.items()):
            self._tx_name_2_ind[tx_name] = tx_ind
            self._transmitters.append(tx)

    @property
    def path_gain(self):
        return self._path_gain

    @property
    def cell_centers(self):
        return self._cell_centers

    @property
    def rss(self):
        tx_powers = [tx.power for tx in self._transmitters]
        tx_powers = tf.convert_to_tensor(tx_powers)
        return tx_powers[:, tf.newaxis, tf.newaxis] * self.path_gain

    @property
    def transmitters(self):
        return self._transmitters

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def cm_voxel_size(self):
        return self._cm_voxel_size

    @property
    def sinr(self):
        """
        [num_tx, num_cells_y, num_cells_x], tf.float : SINR
        across the coverage map from all transmitters
        """
        # Total received power from all transmitters
        # [num_tx, num_cells_y, num_cells_x]
        total_pow = tf.reduce_sum(self.rss, axis=0)

        # Interference for each transmitter
        interference = total_pow[tf.newaxis] - self.rss

        # Thermal noise
        noise = self._scene.thermal_noise_power

        # SINR
        return self.rss / (interference + noise)

    def show(self,
             metric="path_gain",
             tx=None,
             vmin=None,
             vmax=None,
             show_tx=True,
             show_rx=False,
             show_ris=False):

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        if isinstance(tx, int):
            if tx >= self.num_tx:
                raise ValueError("Invalid transmitter index")
        elif isinstance(tx, str):
            if tx in self._tx_name_2_ind:
                tx = self._tx_name_2_ind[tx]
            else:
                raise ValueError(f"Unknown transmitter with name '{tx}'")
        elif tx is None:
            pass
        else:
            msg = "Invalid type for `tx`: Must be a string, int, or None"
            raise ValueError(msg)

        cm3d = getattr(self, metric)
        if tx is not None:
            cm3d = cm3d[tx]
        else:
            cm3d = tf.reduce_max(cm3d, axis=0)

        # Convert to dB-scale
        if metric in ["path_gain", "sinr"]:
            with warnings.catch_warnings(record=True) as _:
                # Convert the path gain to dB
                cm3d = 10.*np.log10(cm3d.numpy())
        else:
            with warnings.catch_warnings(record=True) as _:
                # Convert the signal strength to dBm
                cm3d = watt_to_dbm(cm3d).numpy()
        
        # Set label
        if metric == "path_gain":
            label = "Path gain [dB]"
            title = "Path gain"
        elif metric == "rss":
            label = "Received signal strength (RSS) [dBm]"
            title = 'RSS'
        else:
            label = "Signal-to-interference-plus-noise ratio (SINR) [dB]"
            title = 'SINR'
        if (tx is None) & (len(self._transmitters) > 1):
            title = 'Highest ' + title + ' across all TXs'
        elif tx is not None:
            title = title + f' for TX {tx}'

        fig_cm = plt.figure()
        ax = fig_cm.add_subplot(111, projection='3d')
        z, y, x = np.indices(cm3d.shape)
        sc = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=cm3d.flatten(), vmin=vmin, vmax=vmax, s=7)

        plt.colorbar(sc, ax=ax, label=label)
        ax.set_xlabel('Cell index (X-axis)')
        ax.set_ylabel('Cell index (Y-axis)')
        ax.set_zlabel('Cell index (Z-axis)')
        ax.set_title(title)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_box_aspect([
        (cm3d.shape[2]),
        (cm3d.shape[1]),
        (cm3d.shape[0]),
        ])
        
        if show_tx:
            if tx is not None:
                fig_cm.axes[0].scatter(*self._world_to_voxel(self._transmitters[tx].position), marker='P', c='r')
            else:
                for tx in self._transmitters:
                    fig_cm.axes[0].scatter(*self._world_to_voxel(tx.position), marker='P', c='r')

        if show_rx:
            for rx in self._scene.receivers.values():
                fig_cm.axes[0].scatter(*self._world_to_voxel(rx.position), marker='x', c='b')

        if show_ris:
            for ris in self._scene.ris.values():
                fig_cm.axes[0].scatter(*self._world_to_voxel(ris.position), marker='*', c='k')

        return fig_cm

    def _world_to_voxel(self, pos):
        return (pos - self._scene_min) / self._cm_voxel_size
