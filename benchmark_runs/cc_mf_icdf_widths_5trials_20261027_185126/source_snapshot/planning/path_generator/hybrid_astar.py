import heapq as hq
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from enum import Enum

from models.geometry_utils import *


class KinematicsType(Enum):
    """Kinematics model types"""
    BICYCLE = "bicycle"
    DIFFERENTIAL_DRIVE = "differential_drive"


@dataclass
class MotionPrimitive:
    """Motion primitive for Hybrid A* algorithm with bicycle model"""
    steer_angle: float  # steering angle (rad)
    velocity: float     # velocity (m/s) - positive for forward, negative for reverse
    duration: float     # duration (s)
    kinematics_type: KinematicsType = KinematicsType.BICYCLE
    
    def __post_init__(self):
        self.is_reverse = self.velocity < 0


@dataclass
class MotionPrimitiveDD:
    """Motion primitive for Hybrid A* algorithm with differential drive"""
    v: float            # linear velocity (m/s)
    w: float            # angular velocity (rad/s) - positive for left turn
    duration: float     # duration (s)
    kinematics_type: KinematicsType = KinematicsType.DIFFERENTIAL_DRIVE
    
    def __post_init__(self):
        self.is_reverse = self.v < 0


# Type alias for motion primitives
MotionPrimitiveType = Union[MotionPrimitive, MotionPrimitiveDD]


class HybridNode:
    """Node for Hybrid A* algorithm with continuous pose (x, y, θ)"""
    
    def __init__(self, x: float, y: float, theta: float, parent=None, 
                 g_cost: float = math.inf, f_cost: float = math.inf,
                 motion_primitive: Optional[MotionPrimitiveType] = None):
        self.x = x
        self.y = y  
        self.theta = theta  # heading angle in radians
        self.parent = parent
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.motion_primitive = motion_primitive
        
        # Normalize theta to [-π, π]
        self.theta = self._normalize_angle(self.theta)
    
    @property
    def pos(self) -> np.ndarray:
        """Position as numpy array [x, y]"""
        return np.array([self.x, self.y])
    
    @property 
    def pose(self) -> np.ndarray:
        """Full pose as numpy array [x, y, θ]"""
        return np.array([self.x, self.y, self.theta])
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def distance_to(self, other: 'HybridNode') -> float:
        """Euclidean distance to another node"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_diff_to(self, other: 'HybridNode') -> float:
        """Angular difference to another node"""
        diff = abs(self.theta - other.theta)
        return min(diff, 2*math.pi - diff)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HybridNode):
            return False
        return (abs(self.x - other.x) < 1e-6 and 
                abs(self.y - other.y) < 1e-6 and
                abs(self.theta - other.theta) < 1e-6)
    
    def __lt__(self, other) -> bool:
        return self.f_cost < other.f_cost
    
    def __hash__(self) -> int:
        # For discretization - round to grid resolution
        return hash((round(self.x, 2), round(self.y, 2), round(self.theta, 1)))


class HybridAStarGrid:
    """Grid for discretization of continuous space"""
    
    def __init__(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                 xy_resolution: float = 0.1, theta_resolution: float = 0.1):
        self.bounds = bounds  # ((x_min, y_min), (x_max, y_max))
        self.xy_resolution = xy_resolution
        self.theta_resolution = theta_resolution
        
        # Grid dimensions
        self.x_min, self.y_min = bounds[0]
        self.x_max, self.y_max = bounds[1]
        self.theta_bins = int(2 * math.pi / theta_resolution)
        
    def get_grid_key(self, node: HybridNode) -> Tuple[int, int, int]:
        """Get discrete grid key for a continuous node"""
        x_idx = int((node.x - self.x_min) / self.xy_resolution)
        y_idx = int((node.y - self.y_min) / self.xy_resolution)
        theta_idx = int((node.theta + math.pi) / self.theta_resolution) % self.theta_bins
        return (x_idx, y_idx, theta_idx)
    
    def is_valid_position(self, x: float, y: float) -> bool:
        """Check if position is within bounds"""
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max)


class HybridAStarSearch:
    """Hybrid A* pathfinding algorithm supporting multiple kinematics models"""
    
    def __init__(self, grid: HybridAStarGrid, obstacles: List, 
                 vehicle_length: float = 0.3, vehicle_width: float = 0.15,
                 kinematics_type: KinematicsType = KinematicsType.BICYCLE):
        self.grid = grid
        self.obstacles = obstacles
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.kinematics_type = kinematics_type
        
        # Algorithm parameters automatically adjusted based on grid resolution
        self.goal_tolerance_xy = self.grid.xy_resolution * 1.0
        self.goal_tolerance_theta = self.grid.theta_resolution * 1.0
        
        # Cost weights - tuned to penalize inefficient moves more
        self.w_distance = 1.0
        self.w_curvature = 0.2         # Increased slightly
        self.w_reverse = 3.0           # Increased significantly to avoid reversing
        self.w_steering_change = 0.7   # Increased to promote smoother paths
        
        # Heuristic weights
        self.alpha_distance = 1.0
        self.beta_orientation = 0.3
        
        # Collision checking parameters
        self.collision_check_step = 5  # Check every N integration steps
        
        # Generate motion primitives based on kinematics type
        if self.kinematics_type == KinematicsType.BICYCLE:
            self.motion_primitives = self._generate_motion_primitives_bicycle()
        else:  # DIFFERENTIAL_DRIVE
            self.motion_primitives = self._generate_motion_primitives_dd()
    
    def _generate_motion_primitives_bicycle(self) -> List[MotionPrimitive]:
        """
        Generate motion primitives for bicycle model. Duration is calculated to ensure the vehicle
        moves to a new grid cell to avoid getting stuck.
        """
        primitives = []
        
        # Define base velocities and steering angles
        velocities = [0.15, -0.08]  # Forward and reverse
        steer_angles = np.linspace(-0.5, 0.5, 5)  # ~28.6 degrees, 5 steps

        for v in velocities:
            # Duration should be long enough to cross a grid cell
            duration = 1.5 * self.grid.xy_resolution / abs(v)
            for steer in steer_angles:
                # Make straight primitives slightly longer to encourage straight movement
                effective_duration = duration * 1.2 if abs(steer) < 1e-3 else duration
                primitives.append(MotionPrimitive(steer, v, effective_duration))
        
        return primitives
    
    def _generate_motion_primitives_dd(self) -> List[MotionPrimitiveDD]:
        """
        Generate motion primitives for differential drive model.
        """
        primitives = []
        
        # Define base velocities and angular velocities
        v_set = [0.25, -0.15]  # Forward and reverse velocities
        w_set = [-1.5, -0.75, 0, 0.75, 1.5]  # Angular velocities (rad/s)
        base_duration = 0.4  # Base duration in seconds
        
        for v in v_set:
            for w in w_set:
                # Filter out combinations that would create too tight turns
                if abs(w) > 1e-3:  # If turning
                    turn_radius = abs(v / w)
                    min_turn_radius = self.vehicle_length * 0.5  # Minimum feasible turn radius
                    if turn_radius < min_turn_radius:
                        continue
                
                # Adjust duration to ensure movement to new grid cell
                if abs(v) > 1e-3:
                    duration = max(base_duration, 1.2 * self.grid.xy_resolution / abs(v))
                else:
                    duration = base_duration
                
                # Make straight primitives slightly longer to encourage straight movement
                if abs(w) < 1e-3:
                    duration *= 1.2
                
                primitives.append(MotionPrimitiveDD(v, w, duration))
        
        return primitives
    
    def apply_motion_primitive(self, node: HybridNode, primitive: MotionPrimitiveType) -> Tuple[Optional[HybridNode], bool]:
        """
        Apply motion primitive to get next node using appropriate kinematics model
        Returns: (next_node, is_collision_free)
        """
        if isinstance(primitive, MotionPrimitive):
            return self._apply_bicycle_primitive(node, primitive)
        elif isinstance(primitive, MotionPrimitiveDD):
            return self._apply_dd_primitive(node, primitive)
        else:
            raise ValueError(f"Unknown motion primitive type: {type(primitive)}")
    
    def _apply_bicycle_primitive(self, node: HybridNode, primitive: MotionPrimitive) -> Tuple[Optional[HybridNode], bool]:
        """Apply bicycle model motion primitive"""
        L = self.vehicle_length  # wheelbase
        dt = 0.05  # Integration step
        steps = int(primitive.duration / dt)
        
        x, y, theta = node.x, node.y, node.theta
        
        # Integrate bicycle model
        for step in range(steps):
            x += primitive.velocity * dt * math.cos(theta)
            y += primitive.velocity * dt * math.sin(theta)
            theta += (primitive.velocity / L) * math.tan(primitive.steer_angle) * dt
            
            # Check collision at regular intervals during integration
            if step % self.collision_check_step == 0:
                if not self.grid.is_valid_position(x, y):
                    return None, False
                
                temp_node = HybridNode(x, y, theta)
                if self.check_collision(temp_node):
                    return None, False
        
        # Final collision check
        final_node = HybridNode(x, y, theta, parent=node, motion_primitive=primitive)
        if not self.grid.is_valid_position(x, y) or self.check_collision(final_node):
            return None, False
        
        return final_node, True
    
    def _apply_dd_primitive(self, node: HybridNode, primitive: MotionPrimitiveDD) -> Tuple[Optional[HybridNode], bool]:
        """Apply differential drive motion primitive"""
        dt = 0.05  # Integration step
        steps = int(primitive.duration / dt)
        
        x, y, theta = node.x, node.y, node.theta
        
        # Integrate differential drive model
        for step in range(steps):
            x += primitive.v * dt * math.cos(theta)
            y += primitive.v * dt * math.sin(theta)
            theta += primitive.w * dt
            
            # Check collision at regular intervals during integration
            if step % self.collision_check_step == 0:
                if not self.grid.is_valid_position(x, y):
                    return None, False
                
                temp_node = HybridNode(x, y, theta)
                if self.check_collision(temp_node):
                    return None, False
        
        # Final collision check
        final_node = HybridNode(x, y, theta, parent=node, motion_primitive=primitive)
        if not self.grid.is_valid_position(x, y) or self.check_collision(final_node):
            return None, False
        
        return final_node, True
    
    def calculate_cost(self, current: HybridNode, next_node: HybridNode) -> float:
        """Calculate transition cost with improved steering change handling for both models"""
        distance = current.distance_to(next_node)
        
        # Curvature cost - depends on kinematics model
        curvature = 0
        if next_node.motion_primitive:
            if isinstance(next_node.motion_primitive, MotionPrimitive):
                # Bicycle model curvature
                curvature = abs(math.tan(next_node.motion_primitive.steer_angle) / self.vehicle_length)
            elif isinstance(next_node.motion_primitive, MotionPrimitiveDD):
                # Differential drive curvature
                if abs(next_node.motion_primitive.v) > 1e-6:
                    curvature = abs(next_node.motion_primitive.w / next_node.motion_primitive.v)
                else:
                    curvature = abs(next_node.motion_primitive.w)  # Pure rotation
        
        # Reverse penalty
        reverse_penalty = 0
        if next_node.motion_primitive and next_node.motion_primitive.is_reverse:
            reverse_penalty = self.w_reverse
        
        # Steering/angular velocity change penalty
        control_change_penalty = 0
        if (current.motion_primitive is not None and 
            next_node.motion_primitive is not None and
            type(current.motion_primitive) == type(next_node.motion_primitive)):
            
            if isinstance(next_node.motion_primitive, MotionPrimitive) and isinstance(current.motion_primitive, MotionPrimitive):
                # Bicycle model - steering angle change
                control_diff = abs(current.motion_primitive.steer_angle - 
                              next_node.motion_primitive.steer_angle)
                control_change_penalty = self.w_steering_change * control_diff
            elif isinstance(next_node.motion_primitive, MotionPrimitiveDD) and isinstance(current.motion_primitive, MotionPrimitiveDD):
                # Differential drive - angular velocity change
                control_diff = abs(current.motion_primitive.w - 
                                 next_node.motion_primitive.w)
                control_change_penalty = self.w_steering_change * control_diff
        
        total_cost = (self.w_distance * distance + 
                     self.w_curvature * curvature * distance +
                     reverse_penalty +
                     control_change_penalty)
        
        return total_cost
    
    def heuristic(self, node: HybridNode, goal: HybridNode) -> float:
        """
        Improved heuristic function with model-specific Dubins-like estimation
        """
        euclidean_dist = node.distance_to(goal)
        angle_diff = node.angle_diff_to(goal)
        
        # Model-specific minimum turn radius estimation
        if self.kinematics_type == KinematicsType.BICYCLE:
            # Bicycle model with maximum steering angle
            max_steer_angle = 0.6  # rad
            min_turn_radius = self.vehicle_length / math.tan(max_steer_angle)
        else:  # DIFFERENTIAL_DRIVE
            # Differential drive with maximum angular velocity
            max_angular_velocity = 1.5  # rad/s
            max_linear_velocity = 0.25  # m/s
            min_turn_radius = max_linear_velocity / max_angular_velocity
        
        # Estimate if direct path is possible or if turning is needed
        direct_heading = math.atan2(goal.y - node.y, goal.x - node.x)
        heading_to_goal_diff = abs(node.theta - direct_heading)
        heading_to_goal_diff = min(heading_to_goal_diff, 2*math.pi - heading_to_goal_diff)
        
        # If significant turning is needed, add turning cost
        turn_cost = 0
        if heading_to_goal_diff > 0.3:  # ~17 degrees
            estimated_turn_length = heading_to_goal_diff * min_turn_radius
            turn_cost = 0.5 * estimated_turn_length
        
        return (self.alpha_distance * euclidean_dist + 
                self.beta_orientation * angle_diff +
                turn_cost)
    
    def check_collision(self, node: HybridNode) -> bool:
        """Check collision for vehicle at given pose using improved method"""
        # Get vehicle corners
        corners = self._get_vehicle_corners(node)
        
        for i, obstacle in enumerate(self.obstacles):
            if self._vehicle_obstacle_collision_improved(corners, obstacle):
                # Debug information for collision
                # print(f"DEBUG: Collision detected at pose ({node.x:.3f}, {node.y:.3f}, {node.theta:.3f}) with obstacle {i}")
                # print(f"DEBUG: Vehicle corners: {corners}")
                return True
        return False
    
    def _get_vehicle_corners(self, node: HybridNode) -> np.ndarray:
        """Get vehicle corner positions"""
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        
        # Vehicle corners in local frame
        local_corners = np.array([
            [half_length, half_width],
            [half_length, -half_width], 
            [-half_length, -half_width],
            [-half_length, half_width]
        ])
        
        # Transform to global frame
        cos_theta = math.cos(node.theta)
        sin_theta = math.sin(node.theta)
        
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])
        
        global_corners = local_corners @ R.T + node.pos
        return global_corners
    
    def _vehicle_obstacle_collision_improved(self, vehicle_corners: np.ndarray, obstacle) -> bool:
        """
        Improved collision detection using SAT-like approach
        Check both: vehicle corners inside obstacle AND obstacle vertices inside vehicle
        """
        A, b = obstacle.get_convex_rep()
        b = b.reshape((len(b),))
        
        # Method 1: Check if any vehicle corner is inside obstacle
        for corner in vehicle_corners:
            if all(A @ corner - b <= 1e-6):  # Small tolerance for numerical stability
                return True
        
        # Method 2: Check if any obstacle vertex is inside vehicle rectangle
        # Get obstacle vertices (simplified - assumes rectangular obstacles)
        try:
            obstacle_vertices = self._get_obstacle_vertices(obstacle)
            if obstacle_vertices is not None:
                for vertex in obstacle_vertices:
                    if self._point_in_oriented_rectangle(vertex, vehicle_corners):
                        return True
        except:
            # Fallback to original method if obstacle vertex extraction fails
            pass
        
        return False
    
    def _get_obstacle_vertices(self, obstacle):
        """Extract obstacle vertices (simplified for rectangular obstacles)"""
        try:
            A, b = obstacle.get_convex_rep()
            b = b.reshape((len(b),))
            
            # For rectangular obstacles, solve the constraint intersection
            # This is a simplified approach - for complex polygons, more sophisticated methods needed
            if len(A) == 4:  # Assuming rectangular obstacle
                # Find intersection points of constraint lines
                vertices = []
                for i in range(len(A)):
                    for j in range(i+1, len(A)):
                        # Solve A[i] @ x = b[i] and A[j] @ x = b[j]
                        try:
                            A_2x2 = np.array([A[i], A[j]])
                            b_2x1 = np.array([b[i], b[j]])
                            vertex = np.linalg.solve(A_2x2, b_2x1)
                            
                            # Check if this vertex satisfies all constraints
                            if all(A @ vertex - b <= 1e-6):
                                vertices.append(vertex)
                        except np.linalg.LinAlgError:
                            continue
                
                return np.array(vertices) if len(vertices) > 0 else None
        except:
            pass
        return None
    
    def _point_in_oriented_rectangle(self, point: np.ndarray, rect_corners: np.ndarray) -> bool:
        """Check if point is inside oriented rectangle defined by corners"""
        # Use cross product method to check if point is inside the quadrilateral
        def cross_product_2d(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]
        
        # Check if point is on the same side of all edges
        for i in range(len(rect_corners)):
            j = (i + 1) % len(rect_corners)
            edge = rect_corners[j] - rect_corners[i]
            to_point = point - rect_corners[i]
            
            cross = cross_product_2d(edge, to_point)
            if cross < -1e-6:  # Point is on the wrong side
                return False
        
        return True
    
    def _vehicle_obstacle_collision(self, corners: np.ndarray, obstacle) -> bool:
        """Original collision detection method (kept as fallback)"""
        A, b = obstacle.get_convex_rep()
        b = b.reshape((len(b),))
        
        # Check if any corner is inside obstacle
        for corner in corners:
            if all(A @ corner - b <= 0):
                return True
        return False
    
    def is_goal_reached(self, node: HybridNode, goal: HybridNode) -> bool:
        """Check if goal is reached within tolerance"""
        xy_dist = node.distance_to(goal)
        angle_diff = node.angle_diff_to(goal)
        
        return (xy_dist <= self.goal_tolerance_xy and 
                angle_diff <= self.goal_tolerance_theta)
    
    def hybrid_a_star(self, start_pose: np.ndarray, goal_pose: np.ndarray) -> List[HybridNode]:
        """Main Hybrid A* algorithm with improved OPEN set management"""
        start_node = HybridNode(start_pose[0], start_pose[1], start_pose[2], g_cost=0)
        goal_node = HybridNode(goal_pose[0], goal_pose[1], goal_pose[2])
        
        # Check if start and goal positions are collision-free
        if self.check_collision(start_node):
            print("ERROR: Start position is in collision!")
            return []
        
        if self.check_collision(goal_node):
            print("ERROR: Goal position is in collision!")
            return []
        
        start_node.f_cost = self.heuristic(start_node, goal_node)
        
        open_set = []
        hq.heappush(open_set, start_node)
        
        # Improved OPEN set management to avoid duplicates
        closed_set = set()
        open_dict = {self.grid.get_grid_key(start_node): start_node}
        
        iteration_count = 0
        max_iterations = 6000  # Adjusted for a balance of speed and completeness
        
        while open_set and iteration_count < max_iterations:
            iteration_count += 1

            if iteration_count % 1000 == 0:
                print(f"  [Hybrid A* Progress] Iteration: {iteration_count}/{max_iterations}, Open Set Size: {len(open_set)}")
            
            current = hq.heappop(open_set)
            current_key = self.grid.get_grid_key(current)
            
            # Skip if already processed or if this is not the best node for this grid cell
            if current_key in closed_set:
                continue
            if current_key in open_dict and open_dict[current_key].g_cost < current.g_cost:
                continue
                
            closed_set.add(current_key)
            if current_key in open_dict:
                del open_dict[current_key]
            
            # Check if goal reached
            if self.is_goal_reached(current, goal_node):
                print(f"Hybrid A* converged after {iteration_count} iterations")
                return self._reconstruct_path(current)
            
            # Expand neighbors using motion primitives
            for primitive in self.motion_primitives:
                next_node, is_valid = self.apply_motion_primitive(current, primitive)
                
                if not is_valid or next_node is None:
                    continue
                
                next_key = self.grid.get_grid_key(next_node)
                if next_key in closed_set:
                    continue
                
                # Calculate costs
                tentative_g = current.g_cost + self.calculate_cost(current, next_node)
                
                # Check if this path is better
                if next_key not in open_dict or tentative_g < open_dict[next_key].g_cost:
                    next_node.g_cost = tentative_g
                    next_node.f_cost = tentative_g + self.heuristic(next_node, goal_node)
                    next_node.parent = current
                    
                    # Update open_dict and push to heap
                    open_dict[next_key] = next_node
                    hq.heappush(open_set, next_node)
        
        if iteration_count >= max_iterations:
            print(f"Hybrid A* reached maximum iterations ({max_iterations})")
        else:
            print("Hybrid A* failed to find path")
        
        return []  # No path found
    
    def _reconstruct_path(self, node: HybridNode) -> List[HybridNode]:
        """Reconstruct path from goal to start"""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path)) 