import os.path as osp
import sys

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from pb_ompl.pb_ompl import PbOMPL, PbOMPLRobot, add_sphere, add_box

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))


class FrankaDemo():
    def __init__(self):
        self.obstacles = []

        self.pybullet_client = bullet_client.BulletClient(p.GUI, options='')
        # pybullet_client.setGravity(0, 0, -9.8)
        self.pybullet_client.setGravity(0, 0, 0)
        self.pybullet_client.setTimeStep(1./240.)

        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pybullet_client.loadURDF("plane.urdf")

        # load robot
        urdf_path = osp.join(osp.dirname(osp.abspath(__file__)),
                             "../pb_ompl/models/franka_description/robots/panda_arm_hand.urdf")
        robot_id = p.loadURDF(urdf_path,(0, 0, 0), useFixedBase=1)
        robot = PbOMPLRobot(self.pybullet_client, robot_id, urdf_path=urdf_path, link_name_ee="panda_hand")
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = PbOMPL(self.pybullet_client, self.robot, self.obstacles, min_distance_robot_env=0.01)
        # self.pb_ompl_interface.set_planner("BITstar")
        self.pb_ompl_interface.set_planner("PRM")
        # self.pb_ompl_interface.set_planner("PRMstar")
        # self.pb_ompl_interface.set_planner("ABITstar")
        # self.pb_ompl_interface.set_planner("AITstar")
        # self.pb_ompl_interface.set_planner("RRTConnect")
        # self.pb_ompl_interface.set_planner("RRTstar")

        # add obstacles
        self.add_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_obstacles(self):
        # add boxes
        self.obstacles.append(add_box(self.pybullet_client, [1, 0, 0.7], [0.5, 0.5, 0.05]))
        self.obstacles.append(add_box(self.pybullet_client, [1, 0, 0.1], [0.5, 0.5, 0.05]))
        self.obstacles.append(add_box(self.pybullet_client, [-1, 0, 0.7], [0.5, 0.5, 0.05]))
        self.obstacles.append(add_box(self.pybullet_client, [-1, 0, 0.1], [0.5, 0.5, 0.05]))

        # add spheres
        self.obstacles.append(add_sphere(self.pybullet_client, [-1, 0, 0.1], 0.5))
        self.obstacles.append(add_sphere(self.pybullet_client, [1, 0, 1], 0.2))

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def demo(self, duration=5.0, ee_pose_target=None):
        start = self.pb_ompl_interface.get_state_not_in_collision()
        goal = self.pb_ompl_interface.get_state_not_in_collision(ee_pose_target=ee_pose_target)
        print(f'start: {start}')
        print(f'goal: {goal}')

        self.robot.set_state(start)
        res, path, bspline_params = self.pb_ompl_interface.plan(
            goal,
            allowed_time=3.0,
            interpolate_num=250,
            smooth_with_bspline=True, smooth_bspline_max_tries=10000, smooth_bspline_min_change=0.05,
            create_bspline=True, bspline_num_knots=20, bspline_degree=5,
            debug=True,
        )

        if res:
            self.pb_ompl_interface.execute(path, sleep_time=duration/len(path))

        return res, path, bspline_params

    def terminate(self):
        self.pybullet_client.disconnect()


if __name__ == '__main__':
    env = FrankaDemo()
    W_H_EE = np.eye(4)
    W_H_EE[:3, 3] = [0.6, 0.5, 0.4]
    for _ in range(2):
        env.demo(ee_pose_target=W_H_EE)
    env.terminate()
