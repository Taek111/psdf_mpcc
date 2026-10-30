import datetime
import copy

import numpy as np

import matplotlib.patches as patches

from models.geometry_utils import *
from sim.simulation import *
from models.dd_dynamics import DifferentialDriveDynamics


class DifferentialDriveStates:
    def __init__(self, x, u=np.array([0.0, 0.0])):
        self._x = x
        self._u = u

    def translation(self):
        return np.array([[self._x[0]], [self._x[1]]])

    def rotation(self):
        return np.array(
            [
                [math.cos(self._x[2]), -math.sin(self._x[2])],
                [math.sin(self._x[2]), math.cos(self._x[2])],
            ]
        )


class LocalizationErrorModel:
    def __init__(
        self,
        position_bias=None,
        heading_bias=0.0,
        position_noise_std=None,
        heading_noise_std=0.0,
        position_noise_frame="world",
        seed=None,
    ):
        self._position_bias = np.asarray(position_bias if position_bias is not None else [0.0, 0.0], dtype=float)
        self._heading_bias = float(heading_bias)
        self._position_noise_std = np.asarray(
            position_noise_std if position_noise_std is not None else [0.0, 0.0],
            dtype=float,
        )
        self._heading_noise_std = float(heading_noise_std)
        self._position_noise_frame = str(position_noise_frame).lower()
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _rotation_matrix(theta):
        return np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=float,
        )

    def observe(self, true_state):
        estimated_state = np.asarray(true_state, dtype=float).copy()
        position_error = self._position_bias + self._rng.normal(loc=0.0, scale=self._position_noise_std, size=2)
        if self._position_noise_frame == "local":
            position_error = self._rotation_matrix(true_state[2]) @ position_error
        elif self._position_noise_frame != "world":
            raise ValueError(f"Unsupported localization position noise frame: {self._position_noise_frame}")
        heading_noise = float(self._rng.normal(loc=0.0, scale=self._heading_noise_std))
        estimated_state[:2] += position_error
        estimated_state[2] = self._wrap_angle(estimated_state[2] + self._heading_bias + heading_noise)
        return estimated_state
        
class DifferentialDriveRectangleGeometry:
    def __init__(self, length, width, rear_dist):
        self._length = length
        self._width = width
        self._rear_dist = rear_dist
        self._region = RectangleRegion((-length + rear_dist) / 2, (length + rear_dist) / 2, -width / 2, width / 2)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i=0, alpha=1.0):
        length, width, rear_dist = self._length, self._width, self._rear_dist
        x, y, theta = state[0], state[1], state[2]
        xc = x + (rear_dist / 2) * math.cos(theta)
        yc = y + (rear_dist / 2) * math.sin(theta)
        vertices = np.array(
            [
                [
                    xc + length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                    yc + length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                ],
                [
                    xc + length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                    yc + length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                ],
                [
                    xc - length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                    yc - length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                ],
                [
                    xc - length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                    yc - length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                ],
            ]
        )
        return patches.Polygon(vertices, alpha=alpha, closed=True, fc="None", ec="tab:brown", linewidth=0.5)

class DifferentialDrivePolygonGeometry:
    def __init__(self, vertices):
        self._vertices = np.array(vertices)
        self._region = PolytopeRegion.convex_hull(self._vertices)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i=0, alpha=1.0):
        x, y, theta = state[0], state[1], state[2]
        rotation_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )
        rotated_vertices = np.dot(self._vertices, rotation_matrix.T) + np.array([x, y])
        return patches.Polygon(rotated_vertices, alpha=alpha, closed=True, fc="None", ec="tab:blue", linewidth=0.5)


class DifferentialDriveMultipleGeometry:
    def __init__(self):
        self._num_geometry = 0
        self._geometries = []
        self._regions = []

    def equiv_rep(self):
        return self._regions

    def add_geometry(self, geometry):
        self._geometries.append(geometry)
        self._regions.append(geometry._region)
        self._num_geometry += 1

    def get_plot_patch(self, state, region_idx, alpha=0.5):
        return self._geometries[region_idx].get_plot_patch(state, alpha)

class DifferentialDriveCircleGeometry:
    def __init__(self, radius):
        self._radius = radius
        self._region = None

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i=0, alpha=1.0):
        x, y = state[0], state[1]
        return patches.Circle((x,y), radius=self._radius, fc="None", ec="black", linewidth=0.5, zorder=3)
    
class DifferentialDriveArrowGeometry:
    def __init__(self, radius, width):
        self._width = width
        self._radius = radius
        self._region = None

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i=0, alpha=0.5):
        x, y, theta = state[0], state[1], state[2]
        return patches.Arrow(x, y, self._radius * np.cos(theta),self._radius * np.sin(theta), fc="black", width=self._width, zorder=2)

class DifferentialDriveSystem(System):
    def __init__(self, state, geometry, dynamics, dt=0.1):
        super().__init__(state=state, geometry=geometry, dynamics=dynamics)
        self._dt = dt

    def get_state(self):
        return self._state._x

    def update(self, unew):
        xnew = self._dynamics.forward_dynamics(self.get_state(), unew, self._dt)
        self._state._x = xnew
        self._state._u = np.array(unew, dtype=float).copy()
        self._time += self._dt

    def logging(self, logger):
        logger._xs.append(self._state._x)
        logger._us.append(self._state._u)


class LocalizedDifferentialDriveSystem(DifferentialDriveSystem):
    def __init__(self, state, geometry, dynamics, localization_error_model, dt=0.1):
        estimated_state = copy.deepcopy(state)
        super().__init__(state=estimated_state, geometry=geometry, dynamics=dynamics, dt=dt)
        self._true_state = copy.deepcopy(state)
        self._localization_error_model = localization_error_model
        self._refresh_estimated_state()

    def _refresh_estimated_state(self):
        self._state._x = self._localization_error_model.observe(self._true_state._x)
        self._state._u = np.array(self._true_state._u, dtype=float).copy()

    def get_true_state(self):
        return self._true_state._x

    def update(self, unew):
        xnew = self._dynamics.forward_dynamics(self._true_state._x, unew, self._dt)
        self._true_state._x = xnew
        self._true_state._u = np.array(unew, dtype=float).copy()
        self._time += self._dt
        self._refresh_estimated_state()

    def logging(self, logger):
        logger._xs.append(np.array(self._true_state._x, dtype=float).copy())
        logger._us.append(np.array(self._true_state._u, dtype=float).copy())
