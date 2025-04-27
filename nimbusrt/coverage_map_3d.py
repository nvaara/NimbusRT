import warnings
import matplotlib.pyplot as plt
import numpy as np

class CoverageMap3D:
    def __init__(self, path_gain, cell_centers):
        self._path_gain = path_gain
        self._cell_centers = cell_centers

    @property
    def path_gain(self):
        return self._path_gain
    
    @property
    def cell_centers(self):
        return self._cell_centers


    def show(self,
             metric="path_gain",
             tx=None,
             vmin=None,
             vmax=None,
             show_tx=True,
             show_rx=False,
             show_ris=False):

        if metric is not "path_gain":
            raise ValueError("Currently only path gain is supported.")

        cm3d = self.path_gain[0] #Temporary
        title = "Path gain"
        label = "Path gain [dB]"       
        with warnings.catch_warnings(record=True) as _:
            cm3d = 10.*np.log10(cm3d.numpy())
        
        fig_cm = plt.figure()
        ax = fig_cm.add_subplot(111, projection='3d')
        z, y, x = np.indices(cm3d.shape)
        sc = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=cm3d.flatten(), cmap='viridis', s=5)

        plt.colorbar(sc, ax=ax, label=label)
        ax.set_xlabel('Cell index (X-axis)')
        ax.set_ylabel('Cell index (Y-axis)')
        ax.set_zlabel('Cell index (Z-axis)')
        ax.set_title('3D Scatter Plot of Volume Points')

        ax.set_box_aspect([
        (cm3d.shape[2]),
        (cm3d.shape[1]),
        (cm3d.shape[0]),
        ])

        return fig_cm
