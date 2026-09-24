#!/usr/bin/env python3
import argparse
import time
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG, TD3
from python_scripts.Logger import Logger
from environments.utils.enums import ObservationType, ActionType
from helpers.simulation import sync
from helpers.cast import str2bool
from environments import environment_map
from python_scripts.simulation_helpers import get_policy, in_degrees
import pandas as pd
import os


def extract_initial_conditions(base_dir):
    """
    Extract initial position and orientation from 100 flight tests.

    Args:
        base_dir: Path to the directory containing save-flight-Test-XX folders

    Returns:
        position: Tuple of 100 (x, y, z) tuples
        orientation: Tuple of 100 (roll, pitch, yaw) tuples
    """
    positions = []
    orientations = []

    for i in range(100):
        test_dir = os.path.join(base_dir, f"save-flight-Test-{i:02d}-*")
        # Find the actual directory (since it has timestamp suffix)
        import glob
        matching_dirs = glob.glob(test_dir)
        if not matching_dirs:
            continue
        test_dir = matching_dirs[0]

        # Read first value from each CSV
        x = pd.read_csv(os.path.join(test_dir, "x0.csv"), header=None).iloc[0, 1]
        y = pd.read_csv(os.path.join(test_dir, "y0.csv"), header=None).iloc[0, 1]
        z = pd.read_csv(os.path.join(test_dir, "z0.csv"), header=None).iloc[0, 1]

        roll = pd.read_csv(os.path.join(test_dir, "rollRad0.csv"), header=None).iloc[0, 1]
        pitch = pd.read_csv(os.path.join(test_dir, "pitchRad0.csv"), header=None).iloc[0, 1]
        yaw = pd.read_csv(os.path.join(test_dir, "yawRad0.csv"), header=None).iloc[0, 1]

        positions.append((x, y, z))
        orientations.append((roll, pitch, yaw))

    return tuple(positions), tuple(orientations)


def quat_xyzw_conjugate(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    return  np.array([-x, -y, -z, w], dtype=np.float64)


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    u = q[0:3]
    s = q[3]

    uv = np.cross(u, v)
    uuv = np.cross(u, uv)

    return v + 2.0 * (s * uv + uuv)


def run_simulation(
        test_env,
        policy_path,
        algorithm='ppo',
        model='best_model.zip',
        gui=True,
        record_video=False,
        simulation_length=20,
        reset=False,
        save=False,
        plot=False,
        debug=False,
        comment="",
        test_number=None,
        test_base_dir="results/MED26/reported/baseline/save-MED-67ddd62-02.21.2026_09.49.30/"
):
    """
    Runs a simulation using the provided environment, policy, and specified parameters.

    The function initializes the test environment with specific configurations, prepares the policy using
    the selected reinforcement learning algorithm and model, and iteratively steps through the simulation while
    logging data, rendering the GUI, and handling state updates. It supports recording videos of the simulation,
    debugging outputs, and optional resetting when the simulation terminates. Additionally, it provides options
    to plot and save logged data after the simulation ends.

    Args:
        test_env: The environment class to be used for simulation.
        policy_path: Path to the directory containing the policy model.
        algorithm: The RL algorithm to use for simulation (default: 'ppo').
        model: The specific model file to load within the policy directory (default: 'best_model.zip').
        gui: Whether to enable GUI rendering during simulation (default: True).
        record_video: Whether to record the video of the simulation (default: False).
        simulation_length: Length of the simulation in seconds (default: 20).
        reset: Whether to reset the environment upon termination (default: False).
        save: Whether to save logged data as a CSV (default: False).
        plot: Whether to plot logged data graphs after simulation (default: False).
        debug: Whether to enable detailed debug outputs during simulation (default: False).
        comment: Additional comments to include in the saved logs (default: "").
        obs_delay_s: Seconds to delay the observation vector by (default: 0.0).
        wind: Dictionary containing wind parameters for simulation (default: None).

    Raises:
        KeyError: If the specified algorithm is not available in the model_map dictionary.

    Returns:
        None

    How To:
        If you want the change the initial position of the camera, define it before the simulation loop:
        p.resetDebugVisualizerCamera(1, 125, -10, [1, 1, 1])
    """

    model_map = {
        'ppo': PPO,
        'sac': SAC,
        'ddpg': DDPG,
        'td3': TD3
    }

    policy = get_policy(model_map[algorithm], policy_path, model)

    if test_number is not None:
        positions, orientations = extract_initial_conditions(test_base_dir)
        INIT_XYZS = np.array([list(positions[test_number])])
        INIT_RPYS = np.array([list(orientations[test_number])])
    else:
        INIT_XYZS = np.array([[0.0, 0.0, 0.1]])
        INIT_RPYS = np.array([[np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]])

    test_env = test_env(
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        gui=gui,
        observation_space=ObservationType('kin'),
        action_space=ActionType('rpm'),
        record=record_video)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=policy_path,
        colab=False
    )

    obs, info = test_env.reset()

    simulation_seconds = simulation_length * test_env.CTRL_FREQ

    start = time.time()

    for i in range(simulation_seconds):
        clipped_actions, _states = policy.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = test_env.step(clipped_actions)

        state = test_env._getDroneStateVector(0)
        position = state[0:3]
        quaternion = state[3:7]
        rpy = state[7:10]
        linear_velocity = state[10:13]
        angular_velocity = rotate_vector_by_quaternion(state[13:16], quat_xyzw_conjugate(quaternion))
        clipped_rpm = state[16:20].squeeze()

        if debug:
            print(f"""
            #################################################################
            Observations:
            Position: {obs[0][0:3]}
            Orientation: {in_degrees(obs[0][3:6])}
            Linear Velocity: {obs[0][6:9]}
            Angular Velocity: {obs[0][9:12]}
            -----------------------------------------------------------------
            Raw Actions Clipped: type {type(clipped_actions)} value {clipped_actions}
            Raw RPM Clipped: {clipped_rpm}
            Terminated: {terminated}
            Truncated: {truncated}
            -----------------------------------------------------------------
            Policy Architecture: {policy.policy}
            #################################################################
            """)

        logger.log(
            drone=0,
            timestamp=i / test_env.CTRL_FREQ,
            state=np.hstack([position,
                             quaternion,
                             rpy,
                             linear_velocity,
                             angular_velocity,
                             clipped_rpm
                             ]),
            reward=reward,
            control=np.zeros(12)
        )

        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if reset and terminated:
            obs, info = test_env.reset(seed=42, options={})

    test_env.close()

    if plot:
        logger.plot_position_and_orientation()
        logger.plot_pwms()

    if save:
        logger.save_as_csv(comment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a simulation given a trained policy")
    parser.add_argument('--policy_path', help='The path to a zip file containing the trained policy')
    parser.add_argument('--model', help='The zip file containing the trained policy')
    parser.add_argument('--algorithm', default='ppo', help='The algorithm used for training')
    parser.add_argument('--test_env', default='CLStage1Sim2Real', type=str,help='The name of the environment to learn, registered with gym_pybullet_drones')
    parser.add_argument('--simulation-length', default=20, type=int, help='The length of the simulation in seconds')
    parser.add_argument('--reset', default=False, type=str2bool, help="If you want to reset the environment, every time that the drone achieve the target position")
    parser.add_argument('--save', default=False, type=str2bool, help='Allow to save the trained data using csv and npy files')
    parser.add_argument('--comment', default="", type=str, help="A comment to describe de simulation saved data")
    parser.add_argument('--plot', default=False, type=str2bool, help="If are shown demo plots")
    parser.add_argument('--debug', default=False, type=str2bool, help="Prints debug information")
    parser.add_argument('--record-video', default=False, type=str2bool, help="Record simulation video")
    parser.add_argument(
        '--test_number',
        default=None,
        type=int,
        help='Test number (0-99) to load initial conditions from (default: None, uses random)',
        metavar=''
    )
    parser.add_argument(
        '--test_base_dir',
        default='results/MED26/reported/baseline/save-MED-67ddd62-02.21.2026_09.49.30/',
        type=str,
        help='Base directory containing the test results (default: results/MED26/reported/acrobatic/save-MED-67ddd62-02.21.2026_09.49.30)',
        metavar=''
    )

    args = parser.parse_args()

    environment_class = environment_map.get(args.test_env)
    if environment_class is None:
        raise ValueError(f"Unknown environment: {args.test_env}")

    run_simulation(
        test_env=environment_class,
        policy_path=args.policy_path,
        algorithm=args.algorithm,
        model=args.model,
        gui=True,
        record_video=args.record_video,
        simulation_length=args.simulation_length,
        reset=args.reset,
        save=args.save,
        comment=args.comment,
        plot=args.plot,
        debug=args.debug,
        test_number=args.test_number,
        test_base_dir=args.test_base_dir
    )
