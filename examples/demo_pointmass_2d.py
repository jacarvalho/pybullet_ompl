import os.path as osp
import sys

import pybullet as p

from pb_ompl import PbOMPLRobot, add_sphere, PbOMPL, add_box

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))


class PointMass2DDemo():
    def __init__(self):
        self.obstacles = []

        p.connect(p.GUI)
        # p.setGravity(0, 0, -9.8)
        p.setGravity(0, 0, 0.)
        p.setTimeStep(1./240.)

        # load robot
        robot_id = p.loadURDF("../models/point_mass_robot_2d.urdf", (0,0,0))
        robot = PbOMPLRobot(robot_id)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = PbOMPL(self.robot, self.obstacles, min_distance_robot_env=0.02)
        # self.pb_ompl_interface.set_planner("BITstar")
        # self.pb_ompl_interface.set_planner("ABITstar")
        # self.pb_ompl_interface.set_planner("AITstar")
        # self.pb_ompl_interface.set_planner("PRMstar")
        # self.pb_ompl_interface.set_planner("RRTConnect")
        self.pb_ompl_interface.set_planner("RRTstar")

        # add obstacles
        self.add_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    # def add_obstacles(self):
    #     # add spheres
    #     self.obstacles.append(pb_ompl.add_sphere([-0.43378472328186035, 0.3334643840789795, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([0.3313474655151367, 0.6288051009178162, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([-0.5656964778900146, -0.484994500875473, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([0.42124247550964355, -0.6656165719032288, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([0.05636655166745186, -0.5149664282798767, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([-0.36961784958839417, -0.12315540760755539, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([-0.8740217089653015, -0.4034936726093292, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([-0.6359214186668396, 0.6683124899864197, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([0.808782160282135, 0.5287870168685913, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([-0.023786112666130066, 0.4590069353580475, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([0.1455741971731186, 0.16420497000217438, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([0.628413736820221, -0.43461447954177856, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([0.17965620756149292, -0.8926276564598083, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([0.6775968670845032, 0.8817358016967773, 0], 0.125))
    #     self.obstacles.append(pb_ompl.add_sphere([-0.3608766794204712, 0.8313458561897278, 0], 0.125))
    #
    #     # store obstacles
    #     self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_obstacles(self):
        # add spheres
        self.obstacles.append(add_sphere([0.3313474655151367, 0.6288051009178162, 0], 0.125))
        self.obstacles.append(add_sphere([-0.36961784958839417, -0.12315540760755539, 0], 0.125))
        self.obstacles.append(add_sphere([-0.8740217089653015, -0.4034936726093292, 0], 0.125))
        self.obstacles.append(add_sphere([0.808782160282135, 0.5287870168685913, 0], 0.125))
        self.obstacles.append(add_sphere([0.6775968670845032, 0.8817358016967773, 0], 0.125))
        self.obstacles.append(add_sphere([-0.3608766794204712, 0.8313458561897278, 0], 0.125))
        self.obstacles.append(add_sphere([0.7156378626823425, -0.6923345923423767, 0], 0.125))
        self.obstacles.append(add_sphere([0.35, 0, 0], 0.125))

        # add boxes
        self.obstacles.append(add_box([0.607781708240509, 0.19512386620044708, 0], [0.20000000298023224/2, 0.20000000298023224/2, 0.1/2]))
        self.obstacles.append(add_box([-0.3352295458316803, -0.6887519359588623, 0], [0.20000000298023224/2, 0.20000000298023224/2, 0.1/2]))
        self.obstacles.append(add_box([-0.6572632193565369, 0.41827881932258606, 0], [0.20000000298023224/2, 0.20000000298023224/2, 0.1/2]))
        self.obstacles.append(add_box([-0.664594292640686, 0.016457155346870422, 0], [0.20000000298023224/2, 0.20000000298023224/2, 0.1/2]))
        self.obstacles.append(add_box([0.8165988922119141, -0.19856023788452148, 0], [0.20000000298023224/2, 0.20000000298023224/2, 0.1/2]))
        self.obstacles.append(add_box([-0.8222246170043945, -0.6448580026626587, 0], [0.20000000298023224/2, 0.20000000298023224/2, 0.1/2]))
        self.obstacles.append(add_box([-0.8946458101272583, 0.8962447643280029, 0], [0.20000000298023224/2, 0.20000000298023224/2, 0.1/2]))
        self.obstacles.append(add_box([-0.23994405567646027, 0.6021060943603516, 0], [0.20000000298023224/2, 0.20000000298023224/2, 0.1/2]))
        self.obstacles.append(add_box([0.305103600025177, -0.3661990463733673, 0], [0.20000000298023224/2, 0.20000000298023224/2, 0.1/2]))
        self.obstacles.append(add_box([0., 0.5+0.05, 0], [0.2/2, (1.0-0.1/2)/2, 0.1/2]))
        self.obstacles.append(add_box([0., -0.5-0.05, 0], [0.2/2, (1.0-0.1/2)/2, 0.1/2]))

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
            allowed_time=3.0,
            interpolate_num=250,
            smooth_with_bspline=True, smooth_bspline_max_tries=10000, smooth_bspline_min_change=0.05,
            create_bspline=True, bspline_num_knots=20, bspline_degree=5,
            debug=True,
        )

        if res:
            self.pb_ompl_interface.execute(path, sleep_time=duration/len(path))

        return res, path, bspline_params


if __name__ == '__main__':
    maze = PointMass2DDemo()
    maze.demo()
