"""
This script is used to generate a dataset for the planner in the real world for training INN.
It will generate a series of random waypoints [x, y, z, yaw]
and using min snap trajectory optimization to generate the trajectory.

@NOTE: when doing real-world exp, better to first take off, then start plan the trajectory.
       and the lower bound of z in the world.json is better set to be above 0
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import pkg_resources

from rotorpy.world import World
from rotorpy.utils.occupancy_map import OccupancyMap
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.environments import Environment
from rotorpy.wind.default_winds import NoWind
from rotorpy.trajectories.minsnap import MinSnap

class RandomMinSnapGen:
    def __init__(self):
        ### world and map settings
        self.res = 0.2 # resolution of the map
        self.inflate = 0.25 # inflate the map
        package = pkg_resources.get_distribution("rotorpy").location
        self.world = World.from_file(package + '/rotorpy/worlds/empty.json') # world boundary
        self.occupancy_map = OccupancyMap(world=self.world,
                                          resolution=[self.res, self.res, self.res],
                                          margin=self.inflate)

        ### min snap trajectory settings
        self.v_avg = 1.0
        self.v_start = [0, 0, 0]
        self.v_end = [0, 0, 0]
        self.last_pos = np.array([0.5, 1, 0.7])
        self.last_quat = [0 ,0, 0, 1]
        self.v_max = 2.1 # max vel for each dimension
        self.a_max = 2.1 # max acc for each dimension

        # range of random waypoints
        self.x_range = (0.01, 5)
        self.y_range = (0.01, 5)
        self.z_range = (0.2, 2)
        self.yaw_range = (-1.5, 1.5)  # may cause problem when planning with yaw(KKT singular error) ?
        self.n_waypoints = 4 # number of waypoints in each trajectory(not include start point)

        #controller and simulation settings
        self.sim_instance = Environment(vehicle=Multirotor(quad_params),           # vehicle object, must be specified.
                                   controller=SE3Control(quad_params),        # controller object, must be specified.
                                   trajectory=None,         # trajectory object, must be specified.
                                   wind_profile=NoWind(),               # OPTIONAL: wind profile object, if none is supplied it will choose no wind.
                                   sim_rate     = 100,                        # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
                                   imu          = None,                       # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
                                   mocap        = None,                       # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.
                                   estimator    = None,                       # OPTIONAL: estimator object
                                   world        = self.world,                 # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
                                   safety_margin= 0.25                        # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
                                   )


    def generate_waypoints(self):
        x_values = np.random.uniform(self.x_range[0], self.x_range[1], (self.n_waypoints, 1))
        y_values = np.random.uniform(self.y_range[0], self.y_range[1], (self.n_waypoints, 1))
        z_values = np.random.uniform(self.z_range[0], self.z_range[1], (self.n_waypoints, 1))
        waypoints = np.hstack((x_values, y_values, z_values))
        yaw_angles = np.random.uniform(self.yaw_range[0], self.yaw_range[1], self.n_waypoints)

        #append the current position to the waypoints
        waypoints = np.vstack((self.last_pos, waypoints))
        yaw_angles = np.append(self.quat_to_yaw(self.last_quat), yaw_angles)
        print("waypoints: \n")
        print(waypoints)
        print("yaw angles: \n")
        print(yaw_angles)
        return waypoints, yaw_angles

    def is_safe_traj(self, traj, ts = 0.1):
        """
        Check if the trajectory is safe.
        """
        # Evaluate polynomial.
        t_total = traj.t_keyframes[-1]
        t_list = np.arange(0, t_total + ts, ts)
        for t in t_list:
            cur_x = traj.update(t)['x']
            cur_yaw = traj.update(t)['yaw']
            cur_v = traj.update(t)['x_dot']
            cur_a = traj.update(t)['x_ddot']
            # position check
            if not self.occupancy_map.is_valid_metric(cur_x) \
               or not (self.yaw_range[0] <= cur_yaw <= self.yaw_range[1]):
                return False
            if not np.all(np.abs(cur_v) <= self.v_max) or not np.all(np.abs(cur_a) <= self.a_max):
                return False
        return True

    def get_initial_state(self, last_pos, last_quat):
        """
        note, in real world exp, last_pos and quat could be from vicon
        """
        return {'x': last_pos,
                'v': [0, 0, 0],
                'q': last_quat,
                'w': np.zeros(3,),
                'wind': np.array([0,0,0]),
                'rotor_speeds': np.array([1788.53] * 4)}

    def return_traj(self):
        """
        Function to return a trajectory and it's initial state.
        """
        waypoints, yaw_angles = self.generate_waypoints()
        x0 = self.get_initial_state(self.last_pos, self.last_quat)
        traj = MinSnap(waypoints, yaw_angles, v_max=5, v_avg=self.v_avg,
                       v_start=self.v_start, v_end=self.v_end, verbose=True)
        return traj, x0

    def quat_to_yaw(self, quat):
        """
        quat: [i, j, k, w ]
        """
        r = R.from_quat(quat)
        return r.as_euler('zyx')[0]

    def run_controller(self, traj, x0):
        """
        Run the simulation and controller,
        if reach goal, return true, current pos and quat
        else return false, None, None
        @note:
        when doing real-world exp, current pos and quat could be from vicon,
        here in simulation, we retrieve them from vehicle states
        """

        self.sim_instance.trajectory = traj
        # Setting an initial state. This is optional, and the state representation depends on the vehicle used.
        # Generally, vehicle objects should have an "initial_state" attribute.
        self.sim_instance.vehicle.initial_state = x0
        # Executing the simulator as specified above is easy using the "run" method:
        # All the arguments are listed below with their descriptions.
        # You can save the animation (if animating) using the fname argument. Default is None which won't save it.
        results = self.sim_instance.run(t_final      = traj.t_keyframes[-1],       # The maximum duration of the environment in seconds
                                   use_mocap    = False,       # Boolean: determines if the controller should use the motion capture estimates.
                                   terminate    = None,       # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                                   plot            = True,     # Boolean: plots the vehicle states and commands
                                   plot_mocap      = False,     # Boolean: plots the motion capture pose and twist measurements
                                   plot_estimator  = False,     # Boolean: plots the estimator filter states and covariance diagonal elements
                                   plot_imu        = False,     # Boolean: plots the IMU measurements
                                   animate_bool    = True,     # Boolean: determines if the animation of vehicle state will play.
                                   animate_wind    = False,    # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                                   verbose         = True,     # Boolean: will print statistics regarding the simulation.
                                   fname   = None # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/.
                                   )

        if results['exit'].name == 'COMPLETE'\
            or results['exit'].name == 'TIMEOUT':
            return (True, results['state']['x'][-1], results['state']['q'][-1])

        return (False, None, None)


    def run_planner(self, max_run_time = 1):
        """
        Function to keep the simulation running for max_run_time.
        """
        # @note: update the current pos and quat, could be from vicon
        # here in simulation, we manually set the initial values in __init__()
        #self.last_pos =
        #self.last_quat =
        iter_num = 0
        while iter_num < max_run_time:

            traj, x0 = self.return_traj()
            if traj is None or not traj.if_success: # planner failed
                print("Trajectory is None, optimizing fail, regenerating...")
                continue

            print("[planner]: generated a trajectory ")

            if not self.is_safe_traj(traj):
                print("traj not safe or not feasible, regenerating...")
                continue

            iter_num += 1

            # run the simulation and controller
            if_success, cur_x, cur_quat = self.run_controller(traj, x0)
            # see if it reaches the goal
            # update the last_pos and last_yaw
            if if_success:
                print("[controller]: goal reached!")
                self.last_pos = cur_x
                self.last_quat = cur_quat
                # if reaches the goal, begin with next traj
                print("begin the next round!")

            else:
                print("[controller]: goal not reached!")

                # some errors happened, may let it hover or take off

                break
        print("finish")

if __name__ == "__main__":
    print("start to get dataset of trajectories")
    planner = RandomMinSnapGen()
    planner.run_planner(3)

