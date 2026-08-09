import math
import sys, os 
sys.path.append(os.path.dirname((os.path.abspath(os.path.dirname(__file__)))))
import casadi as ca
import numpy as np

from models.geometry_utils import RectangleRegion
from sim.logger import (
    ControllerLogger,
    GlobalPlannerLogger,
    LocalPlannerLogger,
    SystemLogger,
)


class System:
    def __init__(self, time=0.0, state=None, geometry=None, dynamics=None):
        self._time = time
        self._state = state
        self._geometry = geometry
        self._dynamics = dynamics


class Robot:
    def __init__(self, system):
        self._system = system
        self._system_logger = SystemLogger()

    def _get_navigation_state(self):
        if hasattr(self._system, "get_true_state"):
            return self._system.get_true_state()
        return self._system.get_state()

    def set_global_planner(self, global_planner):
        self._global_planner = global_planner
        self._global_planner_logger = GlobalPlannerLogger()

    def set_local_planner(self, local_planner):
        self._local_planner = local_planner
        self._local_planner_logger = LocalPlannerLogger()

    def set_controller(self, controller):
        self._controller = controller
        self._controller_logger = ControllerLogger()

    def run_global_planner(self, sys, obstacles, goal_pos):
        # TODO: global path shall be generated with `system` and `obstacles`.
        # print(f'Generating global path to {goal_pos} with obstacles {obstacles}')
        self._global_path = self._global_planner.generate_path(sys, obstacles, goal_pos)
        self._global_planner.logging(self._global_planner_logger)

    def run_local_planner(self):
        # TODO: local path shall be generated with `obstacles`.
        self._local_trajectory = self._local_planner.generate_trajectory(self._system, self._global_path)
        self._local_planner.logging(self._local_planner_logger)

    def run_controller(self, obstacles):
        self._control_action = self._controller.generate_control_input(
            self._system, self._global_path, self._local_trajectory, obstacles
        )
        self._controller.logging(self._controller_logger)

    def run_system(self):
        self._system.update(self._control_action)
        self._system.logging(self._system_logger)

    def is_goal_reached(self, position_tolerance=0.01, angle_tolerance=0.01):
        """
        목표 경로의 마지막 지점에 도달했는지 확인합니다.
        
        Args:
            position_tolerance: 위치 도착 판정 거리 임계값 (미터)
            angle_tolerance: 각도 도착 판정 임계값 (라디안)
            
        Returns:
            bool: 목적지 도착 여부
        """
        if self._global_path is None or len(self._global_path) == 0:
            return False

        # 기본적으로 global_path의 마지막 점을 목표로 설정
        goal_position = self._global_path[-1]

        # global_planner가 전체 pose(자세) 정보를 제공하는 경우, 해당 정보를 사용
        if hasattr(self._global_planner, 'get_path_poses'):
            poses = self._global_planner.get_path_poses()
            if poses is not None and len(poses) > 0:
                goal_position = poses[-1]

        current_pos = self._get_navigation_state()  # [x, y, theta]
        goal_pos = np.array(goal_position)
        
        # 위치만 비교 (x, y 좌표)
        current_xy = current_pos[:2]
        goal_xy = goal_pos[:2]

        # 유클리디안 거리 계산
        distance = np.linalg.norm(current_xy - goal_xy)

        # 각도 오차(θ)가 제공된 경우에만 계산
        if len(goal_pos) >= 3 and len(current_pos) >= 3:
            # 회전 각도 오차를 -pi ~ pi 범위로 정규화
            angle_err = abs(((current_pos[2] - goal_pos[2] + math.pi) % (2 * math.pi)) - math.pi)
            return (distance <= position_tolerance) and (angle_err <= angle_tolerance)
        else:
            # goal θ가 정의되지 않은 경우 위치만으로 도착 판정
            return distance <= position_tolerance


class SingleAgentSimulation:
    def __init__(
        self,
        robot,
        obstacles,
        goal_position,
        goal_position_tolerance=0.01,
        goal_angle_tolerance=0.01,
        failure_checker=None,
    ):
        self._robot = robot
        self._obstacles = obstacles
        self._goal_position = goal_position
        self._goal_position_tolerance = float(goal_position_tolerance)
        self._goal_angle_tolerance = float(goal_angle_tolerance)
        self._failure_checker = failure_checker

    def _distance_to_goal(self):
        current_pos = self._robot._get_navigation_state()[:2]
        goal_xy = self._robot._global_path[-1][:2]
        return float(np.linalg.norm(current_pos - goal_xy))

    def run_navigation(self, navigation_time):
        self._robot.run_global_planner(self._robot._system, self._obstacles, self._goal_position)
        
        print(f"시작 위치: {self._robot._get_navigation_state()[:2]}")
        print(f"목적지: {self._goal_position[:2]}")
        print(
            "도착 판정 기준: "
            f"position <= {self._goal_position_tolerance}m, "
            f"angle <= {self._goal_angle_tolerance}rad"
        )
        
        while self._robot._system._time < navigation_time:
            self._robot.run_local_planner()
            self._robot.run_controller(self._obstacles)
            self._robot.run_system()
            
            # 목적지 도착 여부 확인
            if self._robot.is_goal_reached(self._goal_position_tolerance, self._goal_angle_tolerance):
                current_pos = self._robot._get_navigation_state()[:2]
                print(f"목적지에 도착했습니다! 현재 위치: {current_pos}, 시간: {self._robot._system._time:.2f}초")
                return {
                    "status": "success",
                    "failure_reason": None,
                    "goal_reached": True,
                    "final_time": float(self._robot._system._time),
                    "distance_to_goal": self._distance_to_goal(),
                }

            if self._failure_checker is not None:
                failure_info = self._failure_checker(self)
                if failure_info is not None:
                    reason = failure_info.get("reason", "failure")
                    print(f"실패 조건 충족으로 시뮬레이션 종료: {reason}")
                    return {
                        "status": "failure",
                        "failure_reason": reason,
                        "goal_reached": False,
                        "final_time": float(self._robot._system._time),
                        "distance_to_goal": self._distance_to_goal(),
                    }
        else:
            # while loop이 시간 초과로 종료된 경우
            current_pos = self._robot._get_navigation_state()[:2]
            goal_xy = self._robot._global_path[-1][:2]
            distance_to_goal = np.linalg.norm(current_pos - goal_xy)
            print(f"시간 초과로 시뮬레이션 종료. 현재 위치: {current_pos}, 목적지까지 거리: {distance_to_goal:.3f}m")
            return {
                "status": "timeout",
                "failure_reason": "timeout",
                "goal_reached": False,
                "final_time": float(self._robot._system._time),
                "distance_to_goal": float(distance_to_goal),
            }
