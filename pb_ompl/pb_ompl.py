from functools import partial

import numpy as np
from scipy import interpolate

# if the ompl module is not in the PYTHONPATH assume it is installed in a
# subdirectory of the parent directory called "py-bindings."
from os.path import abspath, dirname, join
import sys

from pb_ompl import utils

sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
# print(*sys.path, sep='\n')

from ompl import base as ob
from ompl import geometric as og
import pybullet as p
import time
from itertools import product
import copy

INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 5.0



class PbOMPLRobot:
    '''
    To use with Pb_OMPL. You need to construct a instance of this class and pass to PbOMPL.

    Note:
    This parent class by default assumes that all joints are acutated and should be planned. If this is not your desired
    behaviour, please write your own inheritated class that overrides respective functionalities.
    '''
    def __init__(self, id) -> None:
        # Public attributes
        self.id = id

        # prune fixed joints
        all_joint_num = p.getNumJoints(id)
        all_joint_idx = list(range(all_joint_num))
        joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx
        print(self.joint_idx)
        self.joint_bounds = []

        self.reset()

    def _is_not_fixed(self, joint_idx):
        joint_info = p.getJointInfo(self.id, joint_idx)
        return joint_info[2] != p.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = p.getJointInfo(self.id, joint_id)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds
            if low < high:
                self.joint_bounds.append([low, high])
        print("Joint bounds: {}".format(self.joint_bounds))
        return self.joint_bounds

    def get_cur_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset(self):
        '''
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        '''
        state = [0] * self.num_dim
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0)

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler


class PbOMPL():
    def __init__(self, robot, obstacles = [],
                 min_distance_robot_env=0.0,
                 ) -> None:
        '''
        Args
            robot: A PbOMPLRobot instance.
            obstacles: list of obstacle ids. Optional.
        '''
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        print(self.obstacles)

        self.space = PbStateSpace(robot.num_dim)

        bounds = ob.RealVectorBounds(robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)

        # for sampling inside joint bounds
        self.joint_bounds_np = np.array(joint_bounds)
        self.joint_bounds_low = self.joint_bounds_np[:, 0]
        self.joint_bounds_high = self.joint_bounds_np[:, 1]

        self.ss = og.SimpleSetup(self.space)
        self.min_distance_robot_env = min_distance_robot_env
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(obstacles)
        self.set_planner("RRT") # RRT by default

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state, max_distance=None, check_bounds=False):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set
        # Check for joint bounds due to the bspline interpolation
        if check_bounds:
            if np.any(np.logical_or(state < self.joint_bounds_np[:, 0], state > self.joint_bounds_np[:, 1])):
                return False

        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2, max_distance=0.):  # max_distance=0: don't admit any self-collision
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        if max_distance is None:
            max_distance = self.min_distance_robot_env
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(
                    body1, body2,
                    max_distance=max_distance):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions=True, allow_collision_links=[]):
        self.check_link_pairs = utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []
        moving_links = frozenset(
            [item for item in utils.get_moving_links(robot.id, robot.joint_idx) if not item in allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "PRMstar":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif planner_name == "ABITstar":
            self.planner = og.ABITstar(self.ss.getSpaceInformation())
        elif planner_name == "AITstar":
            self.planner = og.AITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            raise NotImplementedError

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time=DEFAULT_PLANNING_TIME,
                        smooth_with_bspline=False, smooth_bspline_max_tries=10000, smooth_bspline_min_change=0.01,
                        interpolate_num=INTERPOLATE_NUM,
                        create_bspline=False,
                        bspline_num_control_points=32,
                        bspline_degree=3,
                        debug=False,
                        **kwargs):
        '''
        plan a path to gaol from the given robot start state
        '''
        # clear the planning data
        self.ss.clear()

        print("\n#################################################################################")
        print("START planning")
        print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        planner_status = self.ss.solve(allowed_time)
        solved = self.ss.haveExactSolutionPath()

        res = False
        sol_path_list = []
        bspline_params = None
        if solved:
            print("\nFound solution: (smoothing and) interpolating into {} segments".format(interpolate_num))
            sol_path_geometric = self.ss.getSolutionPath()

            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            print(f'len before bspline: {len(sol_path_list)}')

            if smooth_with_bspline:
                ps = og.PathSimplifier(self.si)
                print(f"shortcut path: {ps.shortcutPath(sol_path_geometric)}")
                print(f"simplifymax path: {ps.simplifyMax(sol_path_geometric)}")
                ps.smoothBSpline(sol_path_geometric, smooth_bspline_max_tries, smooth_bspline_min_change)

            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            print(f'len after bspline: {len(sol_path_list)}')

            sol_path_geometric.interpolate(interpolate_num)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            print(f'len after interpolation: {len(sol_path_list)}')

            ############################################################
            # create bspline and check if all states are valid
            try:
                if create_bspline:
                    path_np = np.array(sol_path_list)

                    # https://arxiv.org/pdf/2301.04330.pdf
                    # these knots ensure the first and last control points are the start and goal states
                    d = bspline_degree
                    c = bspline_num_control_points
                    knots = np.zeros(d+1)
                    knots = np.append(knots, np.linspace(1/(c-d), (c-d-1)/(c-d), c-d-1))
                    knots = np.append(knots, np.ones(d+1))

                    tck, u = interpolate.splprep(path_np.T, k=bspline_degree, t=knots, task=-1)
                    tt, cc, k = tck
                    cc = np.array(cc)

                    bspline_params = tck

                    print(f'u shape: {u.shape}')
                    print(f'knots shape: {tt.shape}')
                    print(f'coefficients shape: {cc.shape}')

                    bspl = interpolate.BSpline(tt, cc.T, k)  # note the transpose
                    u_interpolation = np.linspace(0, 1, interpolate_num)
                    bspline_path_interpolated = bspl(u_interpolation)
                    sol_path_list = bspline_path_interpolated.tolist()

                    if debug:
                        import matplotlib.pyplot as plt
                        plt.figure()
                        for i, (joint_spline, joint) in enumerate(zip(bspline_path_interpolated.T, path_np.T)):
                            plt.plot(u_interpolation, joint, linestyle='dashed')
                            plt.plot(u_interpolation, joint_spline, lw=3, alpha=0.7, label=f'BSpline-{i}', zorder=10)
                        plt.legend(loc='best')
                        plt.show()

                        if cc.shape[0] == 2:
                            plt.figure()
                            plt.plot(cc.T[:, 0], cc.T[:, 1], marker='o', label='Control Points')
                            plt.plot(bspline_path_interpolated[:, 0], bspline_path_interpolated[:, 1], c='b', lw=3, alpha=0.7)
                            plt.show()

            except Exception as e:
                print(f'Exception: {e}')
                sol_path_list = []
                bspline_params = None

            ############################################################
            print(f'Checking if all states are valid...')
            all_states_valid = True
            if len(sol_path_list) == 0:
                all_states_valid = False
            else:
                for sol_path in sol_path_list:
                    # set the collision margin of interpolated points to 0
                    if not self.is_state_valid(sol_path, max_distance=0., check_bounds=True):
                        all_states_valid = False
                        break

            if all_states_valid:
                print(f'...all states are valid\n')
                res = True
            else:
                print(f'...NOT all states are valid\n')
                res = False
                sol_path_list = []
                bspline_params = None
        else:
            print("No EXACT solution found\n")

        if create_bspline:
            return res, sol_path_list, bspline_params
        else:
            return res, sol_path_list

    def plan(self, goal, allowed_time=DEFAULT_PLANNING_TIME, **kwargs):
        '''
        plan a path to gaol from current robot state
        '''
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time=allowed_time, **kwargs)

    def execute(self, path, dynamics=False, sleep_time=0.05):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        orig_robot_state = path[0]
        self.robot.set_state(orig_robot_state)
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    p.setJointMotorControl2(self.robot.id, i, p.POSITION_CONTROL, q[i], force=5 * 240.)
            else:
                self.robot.set_state(q)
            p.stepSimulation()
            time.sleep(sleep_time)

    def get_state_not_in_collision(self, max_tries=5000):
        """
        Get a state not in collision
        """
        for _ in range(max_tries):
            rv = np.random.random(self.robot.num_dim)
            state = self.joint_bounds_low + rv * (self.joint_bounds_high - self.joint_bounds_low)
            if self.is_state_valid(state):
                return state
        raise RuntimeError("Failed to find a state not in collision")

    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]


###############################################################################################################
# HELPER FUNCTIONS
def add_box(box_pos, half_box_size, orientation=(0, 0, 0, 1)):  # orientation quaternion xyzw
    colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
    box_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos, baseOrientation=orientation
    )
    return box_id


def add_sphere(sphere_pos, sphere_radius, orientation=(0, 0, 0, 1)):  # orientation quaternion xyzw
    colBoxId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
    sphere_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=sphere_pos, baseOrientation=orientation
    )
    return sphere_id
