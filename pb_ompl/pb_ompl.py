import math
from functools import partial

import numpy as np
from matplotlib import cm
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=3, suppress=True)
import pinocchio
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
    def __init__(self, pybullet_client, id, urdf_path=None, link_name_ee=None) -> None:
        # Public attributes
        self.pybullet_client = pybullet_client
        self.id = id

        assert link_name_ee is not None, "Please specify end-effector link name"
        self.link_name_ee = link_name_ee

        # prune fixed joints
        all_joint_num = self.pybullet_client.getNumJoints(id)
        all_joint_idx = list(range(all_joint_num))
        joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx
        print(self.joint_idx)
        self.joint_bounds = []

        # for sampling inside joint bounds
        self.joint_bounds_np = np.array(self.get_joint_bounds())
        self.joint_bounds_low_np = self.joint_bounds_np[:, 0]
        self.joint_bounds_low_l = self.joint_bounds_low_np.tolist()
        self.joint_bounds_high_np = self.joint_bounds_np[:, 1]
        self.joint_bounds_high_l = self.joint_bounds_high_np.tolist()
        self.joint_ranges_np = self.joint_bounds_high_np - self.joint_bounds_low_np
        self.joint_ranges_l = self.joint_ranges_np.tolist()

        # get link name to index
        self._link_name_to_index = {self.pybullet_client.getBodyInfo(id)[0].decode('UTF-8'): -1, }
        for _id in range(self.pybullet_client.getNumJoints(id)):
            _name = self.pybullet_client.getJointInfo(id, _id)[12].decode('UTF-8')
            self._link_name_to_index[_name] = _id

        self.link_ee_idx = self._link_name_to_index[self.link_name_ee]

        # pinocchio
        self.pinocchio_robot_model = None
        if urdf_path is not None:
            self.pinocchio_robot_model = pinocchio.buildModelFromUrdf(urdf_path)
            self.pinocchio_robot_model_data = self.pinocchio_robot_model.createData()
            self.pinocchio_ee_frameid = self.pinocchio_robot_model.getFrameId(self.link_name_ee)
            self.pinocchio_ee_parent_joint_id = self.pinocchio_robot_model.frames[self.pinocchio_ee_frameid].parent

        ############################################################
        self.reset()

    def _is_not_fixed(self, joint_idx):
        joint_info = self.pybullet_client.getJointInfo(self.id, joint_idx)
        return joint_info[2] != self.pybullet_client.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        joint_bounds = []
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = self.pybullet_client.getJointInfo(self.id, joint_id)
            low = joint_info[8]  # lower bounds
            high = joint_info[9]  # higher bounds
            if low < high:
                joint_bounds.append([low, high])
        print("Joint bounds: {}".format(joint_bounds))
        self.joint_bounds = joint_bounds
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
            self.pybullet_client.resetJointState(self.id, joint, value, targetVelocity=0)

    def get_random_joint_position(self):
        rv = np.random.random(self.num_dim)
        state = self.joint_bounds_low_np + rv * (self.joint_bounds_high_np - self.joint_bounds_low_np)
        return state

    def get_ee_pose(self, state, quat_wxyz=False, return_transformation=False, **kwargs):
        """
        Get end-effector pose
        """
        self.set_state(self.state_to_list(state))
        position, orientation = self.pybullet_client.getLinkState(
            self.id, self.link_ee_idx,
            # computeLinkVelocity=0, computeForwardKinematics=1
        )[:2]
        if quat_wxyz:  # orientation in pybullet is xyzw
            orientation = [orientation[3], orientation[0], orientation[1], orientation[2]]
        if return_transformation:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(orientation).as_matrix()
            pose[:3, 3] = np.array(position)
            return pose
        else:
            return [np.array(position), np.array(orientation)]

    def state_to_list(self, state):
        return [state[i] for i in range(self.num_dim)]

    def run_ik(self, ee_pose_target_in_world, ik_initial_pose=None, debug=False, **kwargs):
        # Gradient-based IK solver using pinochio
        assert self.pinocchio_robot_model is not None, "Please specify urdf_path when constructing PbOMPLRobot"

        if debug:
            print(f'Running Inverse Kinematics for {self.link_name_ee}...')

        ee_pose_target_in_world = pinocchio.SE3(ee_pose_target_in_world)

        if ik_initial_pose is None:
            q = self.get_random_joint_position()
        else:
            q = ik_initial_pose + np.random.normal(0, 0.5, size=ik_initial_pose.shape)
            q = np.clip(q, self.joint_bounds_low_np, self.joint_bounds_high_np)

        eps_position = 1e-2
        eps_orientation = np.deg2rad(1)
        IT_MAX = 1000
        DT = 1e-2
        damp = 1e-12

        np_eye_6 = np.eye(6)

        i = 0
        while True:
            pinocchio.forwardKinematics(self.pinocchio_robot_model, self.pinocchio_robot_model_data, q)
            pinocchio.updateFramePlacements(self.pinocchio_robot_model, self.pinocchio_robot_model_data)
            pinocchio.framesForwardKinematics(self.pinocchio_robot_model, self.pinocchio_robot_model_data, q)

            # Transform the EE target pose from the world frame to the (local/parent) joint frame
            joint_pose_in_world = self.pinocchio_robot_model_data.oMi[self.pinocchio_ee_parent_joint_id]
            ee_pose_in_world = self.pinocchio_robot_model_data.oMf[self.pinocchio_ee_frameid]

            ee_pose_target_in_joint = ee_pose_target_in_world.act(ee_pose_in_world.actInv(joint_pose_in_world))

            dMi = ee_pose_target_in_joint.actInv(joint_pose_in_world)

            err = pinocchio.log(dMi).vector
            err_position = err[:3]
            err_orientation = err[3:]
            # print(np.linalg.norm(err_position), np.linalg.norm(err_orientation))
            if np.all(err_position < eps_position) and np.linalg.norm(err_orientation) < eps_orientation:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break
            pinocchio.computeJointJacobians(self.pinocchio_robot_model, self.pinocchio_robot_model_data, q)
            J = pinocchio.computeJointJacobian(
                self.pinocchio_robot_model, self.pinocchio_robot_model_data, q,
                self.pinocchio_ee_parent_joint_id
            )
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np_eye_6, err))
            q = pinocchio.integrate(self.pinocchio_robot_model, q, v * DT)
            if not i % 10 and debug:
                print('%d: error = %s' % (i, err.T))
            i += 1

        if debug:
            print('------------------')
            print('IK RESULTS\n')
            if success:
                print("IK convergence achieved!")
            else:
                print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

            print(f'\nIK joint position: {q.flatten()}')
            print(f'\nIK error: {err.T}')
            print(f'...Done Inverse Kinematics for {self.link_name_ee}\n')

        if success:
            return q.flatten().tolist()
        else:
            return None

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
    def __init__(self, pybullet_client, robot, obstacles = [],
                 min_distance_robot_env=0.0,
                 min_distance_robot_env_waypoint_checking=0.01,
                 ) -> None:
        '''
        Args
            robot: A PbOMPLRobot instance.
            obstacles: list of obstacle ids. Optional.
        '''
        self.pybullet_client = pybullet_client
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

        self.ss = og.SimpleSetup(self.space)
        self.min_distance_robot_env = min_distance_robot_env
        self.min_distance_robot_env_waypoint_checking = min(min_distance_robot_env, min_distance_robot_env_waypoint_checking)
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
        # Satisfy bounds
        # Should be unecessary if joint bounds is properly set
        # Check for joint bounds due to the bspline interpolation
        if check_bounds:
            if np.any(np.logical_or(state < self.robot.joint_bounds_np[:, 0], state > self.robot.joint_bounds_np[:, 1])):
                return False

        # set the robot internal state
        self.robot.set_state(self.state_to_list(state))

        # check self-collision
        for link1, link2 in self.check_link_pairs:
            # max_distance=0: don't admit any self-collision
            if utils.pairwise_link_collision(self.pybullet_client, self.robot_id, link1, self.robot_id, link2, max_distance=0.):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        if max_distance is None:
            max_distance = self.min_distance_robot_env
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(
                    self.pybullet_client,
                    body1, body2,
                    max_distance=max_distance):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions=True, allow_collision_links=[]):
        self.check_link_pairs = utils.get_self_link_pairs(self.pybullet_client, robot.id, robot.joint_idx) if self_collisions else []
        moving_links = frozenset(
            [item for item in utils.get_moving_links(self.pybullet_client, robot.id, robot.joint_idx) if not item in allow_collision_links])
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
                        simplify_path=True,
                        smooth_with_bspline=False, smooth_bspline_max_tries=10000, smooth_bspline_min_change=0.01,
                        interpolate_num=INTERPOLATE_NUM,
                        create_bspline=False,
                        bspline_num_control_points=32,
                        bspline_degree=3,
                        bspline_zero_vel_at_start_and_end=True,
                        bspline_zero_acc_at_start_and_end=True,
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
                if simplify_path:
                    # https://ompl.kavrakilab.org/classompl_1_1geometric_1_1PathSimplifier.html
                    print(f"shortcut path return: {ps.shortcutPath(sol_path_geometric, maxSteps=1000)}")
                    # print(f"simplify path return: {ps.simplify(sol_path_geometric, maxTime=1.0)}")
                    # print(f"simplifymax path return: {ps.simplifyMax(sol_path_geometric)}")
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

                    # Fit a b-spline to the path
                    tck, u = interpolate.splprep(path_np.T, k=bspline_degree, t=knots, task=-1)
                    tt, cc, k = tck
                    cc = np.array(cc)

                    if bspline_zero_vel_at_start_and_end:
                        # The initial and final velocity should be zero
                        # Set the second and second-to-last control points to be the same
                        # as the first and last control points.
                        cc[:, 1] = cc[:, 0].copy()
                        cc[:, -2] = cc[:, -1].copy()
                    if bspline_zero_acc_at_start_and_end:
                        # The initial and final acceleration should be zero
                        # Set the third and third-to-last control points to be the same
                        # as the first and last control points.
                        cc[:, 2] = cc[:, 0].copy()
                        cc[:, -3] = cc[:, -1].copy()

                    # Update the bspline parameters
                    tck[1] = cc.copy()

                    bspline_params = tck

                    print(f'u shape: {u.shape}')
                    print(f'knots shape: {tt.shape}')
                    print(f'coefficients shape: {cc.shape}')

                    bspl = interpolate.BSpline(tt, cc.T, k)  # note the transpose
                    u_interpolation = np.linspace(0, 1, interpolate_num)
                    bspline_path_interpolated = bspl(u_interpolation)
                    sol_path_list = bspline_path_interpolated.tolist()
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
                    # Check if all states in the path are not in collision
                    # They can be in collision due to the bspline interpolation
                    if not self.is_state_valid(
                            sol_path,
                            max_distance=self.min_distance_robot_env_waypoint_checking,
                            check_bounds=True):
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

        # Plot the bspline for debugging
        if debug and bspline_params is not None:
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(1, 2, squeeze=False)
            # for i, (joint_spline, joint) in enumerate(zip(bspline_path_interpolated.T, path_np.T)):
            #     axs[0, 0].plot(u_interpolation, joint, linestyle='dashed')
            #     axs[0, 0].plot(u_interpolation, joint_spline, lw=3, alpha=0.7, label=f'BSpline-{i}', zorder=10)
            # plt.ylim(np.min(self.robot.joint_bounds_low_np), np.max(self.robot.joint_bounds_high_np))
            # plt.legend(loc='best')
            # plt.show()

            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(16, 6), squeeze=False)
            # Get the trajectory velocity and acceleration from the b-spline
            bspline_path_interpolated_vel = bspl(u_interpolation, nu=1)
            bspline_path_interpolated_acc = bspl(u_interpolation, nu=2)
            colors = cm.rainbow(np.linspace(0, 1, path_np.shape[1]))
            for i, (joint_spline, joint_spline_vel, joint_spline_acc, joint_path) in enumerate(
                    zip(bspline_path_interpolated.T, bspline_path_interpolated_vel.T, bspline_path_interpolated_acc.T,
                        path_np.T)):
                axs[0, 0].plot(u_interpolation, joint_path, linestyle='dashed', color=colors[i])
                axs[0, 0].plot(u_interpolation, joint_spline, lw=3, alpha=0.7, color=colors[i], label=f'BSpline-{i}', zorder=10)
                axs[0, 1].plot(u_interpolation, joint_spline_vel, lw=3, alpha=0.7, color=colors[i], label=f'BSpline-{i}-vel', zorder=10)
                axs[0, 2].plot(u_interpolation, joint_spline_acc, lw=3, alpha=0.7, color=colors[i], label=f'BSpline-{i}-acc', zorder=10)

            axs[0, 0].set_ylim(np.min(self.robot.joint_bounds_low_np), np.max(self.robot.joint_bounds_high_np))
            axs[0, 0].legend(loc='best')
            axs[0, 0].set_title('Position')
            axs[0, 1].legend(loc='best')
            axs[0, 1].set_title('Velocity')
            axs[0, 2].legend(loc='best')
            axs[0, 2].set_title('Acceleration')
            plt.show()

            if cc.shape[0] == 2:
                plt.figure()
                plt.plot(cc.T[:, 0], cc.T[:, 1], marker='o', label='Control Points')
                plt.plot(bspline_path_interpolated[:, 0], bspline_path_interpolated[:, 1], c='b', lw=3, alpha=0.7)
                plt.show()

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

    def execute(self, path, dynamics=False, sleep_time=0.05, substeps=10):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
            sleep_time: float, sleep time between each step
            substeps: int, number of physics simulations per step
        '''
        orig_robot_state = path[0]
        self.robot.set_state(orig_robot_state)
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    self.pybullet_client.setJointMotorControl2(self.robot.id, i, self.pybullet_client.POSITION_CONTROL, q[i], force=5 * 240.)
                for _ in range(substeps):
                    self.pybullet_client.stepSimulation()
            else:
                self.robot.set_state(q)

            time.sleep(sleep_time)

    def get_state_not_in_collision(self, ee_pose_target=None, max_tries=500, raise_error=True, **kwargs):
        """
        Get a state not in collision, with IK if ee_pose_target is not None
        """
        print(f'\n---> Getting state not in collision...')
        state_valid = None
        state = None
        for j in range(max_tries):
            if ee_pose_target is not None:
                state = self.robot.run_ik(ee_pose_target, **kwargs)
                if state is None:
                    continue
            else:
                state = self.robot.get_random_joint_position()

            if self.is_state_valid(state, check_bounds=True):
                state_valid = state
                break

        if state_valid is not None:
            print(f'...Found a valid state after {j}/{max_tries} tries')
            return state_valid

        if not raise_error:
            return None

        if ee_pose_target is not None and state is None:
            raise RuntimeError(f"Failed to solve IK for:\n{ee_pose_target}")

        raise RuntimeError("Failed to find a state not in collision")

    def get_ee_pose(self, state, **kwargs):
        """
        Get end-effector pose
        """
        return self.robot.get_ee_pose(state, **kwargs)

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
def add_box(pybullet_client, box_pos, half_box_size, orientation=(0, 0, 0, 1)):  # orientation quaternion xyzw
    colBoxId = pybullet_client.createCollisionShape(pybullet_client.GEOM_BOX, halfExtents=half_box_size)
    box_id = pybullet_client.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos, baseOrientation=orientation
    )
    return box_id


def add_sphere(pybullet_client, sphere_pos, sphere_radius, orientation=(0, 0, 0, 1)):  # orientation quaternion xyzw
    colBoxId = pybullet_client.createCollisionShape(pybullet_client.GEOM_SPHERE, radius=sphere_radius)
    sphere_id = pybullet_client.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=sphere_pos, baseOrientation=orientation
    )
    return sphere_id


def finite_difference_vector(x, dt=1., method='central'):
    # finite differences with zero padding at the borders
    diff_vector = np.zeros_like(x)
    if method == 'forward':
        diff_vector[..., :-1, :] = np.diff(x, axis=-2) / dt
    elif method == 'backward':
        diff_vector[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]) / dt
    elif method == 'central':
        diff_vector[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2*dt)
    else:
        raise NotImplementedError
    return diff_vector
