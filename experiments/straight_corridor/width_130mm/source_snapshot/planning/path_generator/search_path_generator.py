import math

import numpy as np
import matplotlib.pyplot as plt

from planning.path_generator.astar import *
from planning.path_generator.hybrid_astar import *


def plot_global_map(path, obstacles):
    fig, ax = plt.subplots()
    for o in obstacles:
        patch = o.get_plot_patch()
        ax.add_patch(patch)
    ax.plot(path[:, 0], path[:, 1])
    plt.xlim([-1 * 0.15, 11 * 0.15])
    plt.ylim([0 * 0.15, 8 * 0.15])
    plt.show()


class AstarPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=quad)
        self._margin = margin

    def generate_path(self, system, obstacles, goal_pos):
        print("generating global path with A*")
        # goal_pos may include orientation; A* operates on (x, y) only
        goal_xy = goal_pos[:2] if len(goal_pos) >= 2 else goal_pos
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.a_star(system.get_state()[:2], goal_xy)
        self._global_path = np.array([p.pos for p in path])
        print(self._global_path)
        if self._global_path == []:
            print("Global Path not found.")
            raise RuntimeError("Global Path not found.")
        if True:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)


class AstarLoSPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=quad)
        self._margin = margin

    def generate_path(self, system, obstacles, goal_pos):
        print("generating global path with A* + Line of Sight")
        # goal_pos may include orientation; A* operates on (x, y) only
        goal_xy = goal_pos[:2] if len(goal_pos) >= 2 else goal_pos
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.a_star(system.get_state()[:2], goal_xy)
        if len(path) == 0:
            print("Global Path not found.")
            raise RuntimeError("Global Path not found.")
        path = graph.reduce_path(path)
        if len(path) == 0:
            print("Global Path reduction failed.")
            raise RuntimeError("Global Path reduction failed.")
        self._global_path = np.array([p.pos for p in path])
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)


class ThetaStarPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=False)
        self._margin = margin

    def generate_path(self, system, obstacles, goal_pos):
        goal_xy = goal_pos[:2] if len(goal_pos) >= 2 else goal_pos
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.theta_star(system.get_state()[:2], goal_xy)
        self._global_path = np.array([p.pos for p in path])
        print(self._global_path)
        if self._global_path == []:
            print("Global Path not found.")
            raise RuntimeError("Global Path not found.")
        if True:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)


class HybridAStarPathGenerator:
    def __init__(self, grid, quad=None, margin=0.05, 
                 vehicle_length=0.3, vehicle_width=0.15,
                 xy_resolution=0.1, theta_resolution=0.2,
                 kinematics_type=KinematicsType.BICYCLE):
        """
        Hybrid A* Path Generator
        
        Args:
            grid: tuple of (bounds, cell_size) where bounds = ((x_min, y_min), (x_max, y_max))
            quad: not used in Hybrid A* (kept for interface compatibility)
            margin: safety margin for collision checking
            vehicle_length: vehicle length (wheelbase) in meters
            vehicle_width: vehicle width in meters  
            xy_resolution: spatial discretization resolution
            theta_resolution: angular discretization resolution
            kinematics_type: KinematicsType.BICYCLE or KinematicsType.DIFFERENTIAL_DRIVE
        """
        self._global_path = None
        self._global_poses = None  # Store poses (x, y, θ) 
        
        bounds, _ = grid  # Extract bounds, ignore cell_size for Hybrid A*
        self._hybrid_grid = HybridAStarGrid(bounds, xy_resolution, theta_resolution)
        self._margin = margin
        self._vehicle_length = vehicle_length
        self._vehicle_width = vehicle_width
        self._kinematics_type = kinematics_type

    def generate_path(self, sys, obstacles, goal_pos):
        """
        Generate path using Hybrid A* algorithm
        
        Args:
            sys: system with get_state() method returning [x, y, θ]
            obstacles: list of obstacle objects
            goal_pos: goal position [x, y] or [x, y, θ]
        
        Returns:
            np.ndarray: path as array of [x, y] positions
        """
        kinematics_name = "Bicycle" if self._kinematics_type == KinematicsType.BICYCLE else "Differential Drive"
        print(f"generating global path with Hybrid A* ({kinematics_name})")
        
        # Get start pose
        start_state = sys.get_state()
        if len(start_state) >= 3:
            start_pose = start_state[:3]  # [x, y, θ]
        else:
            start_pose = np.array([start_state[0], start_state[1], 0.0])  # assume θ=0
        
        # Prepare goal pose
        if len(goal_pos) >= 3:
            goal_pose = goal_pos[:3]
        else:
            # If only [x, y] provided, use current heading as goal heading
            goal_pose = np.array([goal_pos[0], goal_pos[1], start_pose[2]])
        
        # Create Hybrid A* search instance
        search = HybridAStarSearch(
            grid=self._hybrid_grid,
            obstacles=obstacles,
            vehicle_length=self._vehicle_length,
            vehicle_width=self._vehicle_width,
            kinematics_type=self._kinematics_type
        )
        
        # Run algorithm
        path_nodes = search.hybrid_a_star(start_pose, goal_pose)
        
        if len(path_nodes) == 0:
            print("Hybrid A* path not found.")
            import sys as system_module
            system_module.exit(1)
        
        # Extract positions and poses
        self._global_path = np.array([node.pos for node in path_nodes])
        self._global_poses = np.array([node.pose for node in path_nodes])
        
        print(f"Hybrid A* found path with {len(path_nodes)} waypoints")
        print(f"Start: {start_pose}")
        print(f"Goal: {goal_pose}")
        print(f"Path length: {len(self._global_path)}")
        
        return self._global_path

    def get_path_poses(self):
        """Get full poses (x, y, θ) along the path"""
        return self._global_poses

    def get_path_curvatures(self):
        """Get curvature information along the path"""
        if self._global_poses is None or len(self._global_poses) < 2:
            return []
        
        curvatures = []
        for i in range(1, len(self._global_poses)):
            # Simple curvature estimation from heading change
            theta_prev = self._global_poses[i-1, 2]
            theta_curr = self._global_poses[i, 2]
            
            # Normalize angle difference
            theta_diff = theta_curr - theta_prev
            while theta_diff > math.pi:
                theta_diff -= 2 * math.pi
            while theta_diff < -math.pi:
                theta_diff += 2 * math.pi
                
            # Distance between points
            pos_prev = self._global_poses[i-1, :2]
            pos_curr = self._global_poses[i, :2]
            distance = np.linalg.norm(pos_curr - pos_prev)
            
            # Curvature approximation
            if distance > 1e-6:
                curvature = abs(theta_diff) / distance
            else:
                curvature = 0.0
                
            curvatures.append(curvature)
            
        return np.array(curvatures)

    def logging(self, logger):
        logger._paths.append(self._global_path)
