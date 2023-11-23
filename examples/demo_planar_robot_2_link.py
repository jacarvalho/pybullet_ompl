import os.path as osp
import sys

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from pb_ompl.pb_ompl import PbOMPL, PbOMPLRobot, add_sphere, add_box

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))


class PlanarRobot2Link():
    def __init__(self):
        self.obstacles = []

        self.pybullet_client = bullet_client.BulletClient(p.GUI, options='')
        # self.pybullet_client.setGravity(0, 0, -9.8)
        self.pybullet_client.setGravity(0, 0, 0)
        self.pybullet_client.setTimeStep(1./240.)

        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pybullet_client.loadURDF("plane.urdf")

        # load robot
        robot_id = p.loadURDF(
            osp.join(osp.dirname(osp.abspath(__file__)), "../pb_ompl/models/planar_robot_2_link.urdf"),
            (0,0,0), useFixedBase = 1)
        robot = PbOMPLRobot(self.pybullet_client, robot_id, link_name_ee='link_ee')
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = PbOMPL(self.pybullet_client, self.robot, self.obstacles, min_distance_robot_env=0.02)
        # self.pb_ompl_interface.set_planner("BITstar")
        # self.pb_ompl_interface.set_planner("PRM")
        self.pb_ompl_interface.set_planner("PRMstar")
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
        # add spheres
        self.obstacles.append(add_sphere(self.pybullet_client, [0.3, 0.3, 0.], 0.1))
        self.obstacles.append(add_sphere(self.pybullet_client, [0.3, -0.3, 0.], 0.1))
        self.obstacles.append(add_sphere(self.pybullet_client, [-0.3, 0.25, 0.], 0.1))
        self.obstacles.append(add_sphere(self.pybullet_client, [-0.3, -0.25, 0.], 0.1))

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def demo(self, duration=5.0):
        start = self.pb_ompl_interface.get_state_not_in_collision()
        goal = self.pb_ompl_interface.get_state_not_in_collision()
        print(f'start: {start}')
        print(f'goal: {goal}')

        self.robot.set_state(start)
        res, path, bspline_params = self.pb_ompl_interface.plan(
            goal,
            allowed_time=10.0,
            interpolate_num=250,
            smooth_with_bspline=True, smooth_bspline_max_tries=10000, smooth_bspline_min_change=0.05,
            create_bspline=True, bspline_num_knots=20, bspline_degree=5,
            debug=True,
        )

        if res:
            # path = [[0., 0.] * len(path)]
            self.pb_ompl_interface.execute(
                path,
                sleep_time=duration/len(path),
                # sleep_time=100000
            )

        return res, path, bspline_params

    def terminate(self):
        self.pybullet_client.disconnect()


if __name__ == '__main__':
    env = PlanarRobot2Link()
    env.demo()
    env.terminate()
