import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .data import periodic_positions


class DensityField:
    def __init__(self, box_size=1000.0, grid_size=512, smoothing_radius=10.0):
        periodic_positions(np.empty((0, 3)), box_size)
        self.box_size = float(box_size)
        self.grid_size = grid_size
        self.smoothing_radius = float(smoothing_radius)
        self.field = None


    def fit(self, particle_positions):
        import MAS_library as mas
        positions = periodic_positions(particle_positions, self.box_size)

        density = np.zeros((self.grid_size,) * 3, dtype=np.float32)
        # El redondeo a float32 puede convertir L-epsilon en L.
        positions = np.mod(positions.astype(np.float32), self.box_size)
        mas.MA(positions, density, self.box_size, MAS='CIC', verbose=False)
        mean = float(density.mean())

        density /= mean
        density -= 1.0
        sigma = self.smoothing_radius * self.grid_size / self.box_size
        self.field = gaussian_filter(density, sigma=sigma, mode='wrap')
        return self


    def sample(self, positions):
        if self.field is None:
            raise RuntimeError('Run fit before sample')
        coordinates = periodic_positions(positions, self.box_size).T
        coordinates *= self.grid_size / self.box_size
        # grid-wrap interpola también entre la última celda y la primera.
        return map_coordinates(self.field, coordinates, order=1, mode='grid-wrap')


    def fit_sample(self, particle_positions, query_positions):
        return self.fit(particle_positions).sample(query_positions)