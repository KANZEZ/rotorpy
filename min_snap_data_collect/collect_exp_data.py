"""
This script is used to generate a dataset for the planner in the real world for training INN.
It will generate a series of random waypoints [x, y, z, yaw]
and using min snap trajectory optimization to generate the trajectory.

@NOTE: when doing real-world exp, better to first take off, then start plan the trajectory.
       and the lower bound of z in the world.json is better set to be above 0
"""
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import pkg_resources

from rotorpy.world import World
from rotorpy.utils.occupancy_map import OccupancyMap
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.environments import Environment
from rotorpy.wind.default_winds import NoWind, SinusoidWind
from rotorpy.trajectories.minsnap import MinSnap

res = 0.2 # resolution of the map
inflate = 0.25 # inflate the map
package = pkg_resources.get_distribution("rotorpy").location
world = World.from_file(package + '/rotorpy/worlds/empty.json') # world boundary
occupancy_map = OccupancyMap(world=world, resolution=[res, res, res], margin=inflate)

# range of random waypoints
x_range = (0.01, 5)
y_range = (0.01, 5)
z_range = (0.2, 2)
yaw_range = (-2.0, 2.0)  # may cause problem when planning with yaw(KKT singular error)
n_waypoints = 4 # number of waypoints in each trajectory(not include start point)

# MinSnap trajectory settings
v_avg = 0.5                                                     # Average velocity, used for time allocation
v_start = v_end = [0, 0, 0]                                             # Start (x,y,z) velocity
last_pos = np.array([0, 1, 0.7])      # current position, could be from vicon
last_yaw = 0.0   # current yaw angle, could be from vicon

def generate_waypoints(x_range, y_range, z_range, n_points=4):
    x_values = np.random.uniform(x_range[0], x_range[1], (n_points, 1))
    y_values = np.random.uniform(y_range[0], y_range[1], (n_points, 1))
    z_values = np.random.uniform(z_range[0], z_range[1], (n_points, 1))
    waypoints = np.hstack((x_values, y_values, z_values))
    yaw_angles = np.random.uniform(yaw_range[0], yaw_range[1], n_points)
    return waypoints, yaw_angles

def get_initial_state(last_pos, yaw_angle):
    """
    could be from vicon
    """
    euler_angles = [yaw_angle, 0.0, 0.0]
    r = R.from_euler('zyx', euler_angles)
    init_q = r.as_quat()
    return {'x': last_pos,
            'v': [0, 0, 0],
            'q': init_q,
            'w': np.zeros(3,),
            'wind': np.array([0,0,0]),
            'rotor_speeds': np.array([1788.53] * 4)}

def is_safe_traj(traj, ts = 0.1):
    """
    Check if the trajectory is safe or out of map.
    """
    # Evaluate polynomial.
    t_total = traj.t_keyframes[-1]
    t_list = np.arange(0, t_total + ts, ts)
    for t in t_list:
        cur_x = traj.update(t)['x']
        cur_yaw = traj.update(t)['yaw']
        if not occupancy_map.is_valid_metric(cur_x) \
            or not (yaw_range[0] <= cur_yaw <= yaw_range[1]):
            return False
    return True

def keep_running(max_run_time = 3):
    """
    Function to keep it running for max_run_time.
    """
    global v_avg, v_start, v_end, last_pos, last_yaw

    waypoints, yaw_angles = generate_waypoints(x_range, y_range, z_range, n_points=n_waypoints)
    waypoints = np.vstack((last_pos, waypoints))
    yaw_angles = np.hstack((last_yaw, yaw_angles))
    x0 = get_initial_state(waypoints[0], yaw_angles[0])

    with tqdm(total=max_run_time) as pbar:
        cur_run_time = 0
        while cur_run_time < max_run_time:
            # generate min snap trajectory
            traj = MinSnap(waypoints, yaw_angles, v_max=5, v_avg=v_avg, v_start=v_start, v_end=v_end, verbose=True)
            # check if the trajectory is valid
            # if not, regenerate the trajectory
            if not traj.if_success: # optimization fail
                print("Trajectory is None, optimizing fail, regenerating...")
            elif not is_safe_traj(traj): # out of map
                print("Trajectory is not safe, regenerating...")
            else:
                run(traj, x0)
                cur_run_time += 1
                pbar.update(1)
                last_pos = traj.update(traj.t_keyframes[-1])['x']
                last_yaw = traj.update(traj.t_keyframes[-1])['yaw']

            waypoints, yaw_angles = generate_waypoints(x_range, y_range, z_range, n_points=n_waypoints)
            waypoints = np.vstack((last_pos, waypoints))
            yaw_angles = np.hstack((last_yaw, yaw_angles))
            x0 = get_initial_state(last_pos, last_yaw)

def run(traj, x0):
    sim_instance = Environment(vehicle=Multirotor(quad_params),           # vehicle object, must be specified.
                               controller=SE3Control(quad_params),        # controller object, must be specified.
                               trajectory=traj,         # trajectory object, must be specified.
                               wind_profile=NoWind(),               # OPTIONAL: wind profile object, if none is supplied it will choose no wind.
                               sim_rate     = 100,                        # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
                               imu          = None,                       # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
                               mocap        = None,                       # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.
                               estimator    = None,                       # OPTIONAL: estimator object
                               world        = world,                      # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
                               safety_margin= 0.25                        # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
                               )

    # Setting an initial state. This is optional, and the state representation depends on the vehicle used.
    # Generally, vehicle objects should have an "initial_state" attribute.
    sim_instance.vehicle.initial_state = x0

    # Executing the simulator as specified above is easy using the "run" method:
    # All the arguments are listed below with their descriptions.
    # You can save the animation (if animating) using the fname argument. Default is None which won't save it.

    results = sim_instance.run(t_final      = traj.t_keyframes[-1],       # The maximum duration of the environment in seconds
                               use_mocap    = False,       # Boolean: determines if the controller should use the motion capture estimates.
                               terminate    = None,       # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                               plot            = True,     # Boolean: plots the vehicle states and commands
                               plot_mocap      = False,     # Boolean: plots the motion capture pose and twist measurements
                               plot_estimator  = False,     # Boolean: plots the estimator filter states and covariance diagonal elements
                               plot_imu        = False,     # Boolean: plots the IMU measurements
                               animate_bool    = False,     # Boolean: determines if the animation of vehicle state will play.
                               animate_wind    = False,    # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                               verbose         = False,     # Boolean: will print statistics regarding the simulation.
                               fname   = None # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/.
                               )
    #sim_instance.save_to_csv("sim_data.csv")

if __name__=="__main__":
    print("start to generate trajectories")
    keep_running(1) # run for 1 time
