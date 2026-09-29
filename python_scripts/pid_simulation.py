#!/usr/bin/env python3

import argparse
import time
from math import radians
from pathlib import Path

import numpy as np

from environments.pid_simulation_env import (
    PIDSimulationEnv,
    create_pid_simulation_config,
    load_pid_gain_overrides,
)
from environments.utils.enums import ActionType
from helpers.cast import str2bool
from python_scripts.Logger import Logger


DEFAULT_OUTPUT_FOLDER = "results"


def create_simulation_action(
    environment: PIDSimulationEnv,
    args: argparse.Namespace,
) -> np.ndarray:
    """Create the normalized command for the selected simulation mode."""
    if environment.ACT_TYPE == ActionType.POSITION:
        return environment.create_position_action(
            target_xyz=np.asarray(args.target_position, dtype=np.float64),
            target_yaw_radians=radians(args.target_yaw_deg),
        )

    if environment.ACT_TYPE == ActionType.VELOCITY:
        return environment.create_velocity_action(
            target_velocity=np.asarray(args.target_velocity, dtype=np.float64),
            target_yaw_rate_radians_per_second=radians(
                args.target_yaw_rate_deg_per_second
            ),
        )

    raise ValueError("PID simulation supports only position or velocity mode.")


def create_logger_control_target(
    action_type: ActionType,
    args: argparse.Namespace,
) -> np.ndarray:
    """Create the 12-element control target expected by QRLab's Logger."""
    if action_type == ActionType.POSITION:
        return np.array(
            [
                *args.target_position,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                radians(args.target_yaw_deg),
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    if action_type == ActionType.VELOCITY:
        return np.array(
            [
                0.0,
                0.0,
                0.0,
                *args.target_velocity,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                radians(args.target_yaw_rate_deg_per_second),
            ],
            dtype=np.float64,
        )

    raise ValueError("PID simulation supports only position or velocity mode.")


def synchronize_simulation(
    start_time: float,
    completed_steps: int,
    timestep_seconds: float,
) -> None:
    """Synchronize GUI simulation approximately to wall-clock time."""
    target_elapsed = completed_steps * timestep_seconds
    remaining = target_elapsed - (time.perf_counter() - start_time)

    if remaining > 0.0:
        time.sleep(remaining)


def run_pid_simulation(args: argparse.Namespace) -> None:
    """Run one deterministic firmware-like PID simulation."""
    action_type = ActionType(args.mode)
    simulation_config = create_pid_simulation_config(
        battery_voltage=args.battery_voltage,
        body_frame_velocity=args.body_frame_velocity,
    )
    gain_overrides = load_pid_gain_overrides(args.pid_gains)

    environment = PIDSimulationEnv(
        action_type=action_type,
        initial_xyz=np.asarray(args.initial_position, dtype=np.float64),
        initial_rpy=np.radians(
            np.asarray(args.initial_rpy_deg, dtype=np.float64)
        ),
        config=simulation_config,
        gain_overrides=gain_overrides,
        gui=args.gui,
        record=args.record_video,
    )
    normalized_action = create_simulation_action(environment=environment, args=args)
    control_target = create_logger_control_target(
        action_type=action_type,
        args=args,
    )
    logger = (
        Logger(
            logging_freq_hz=int(environment.CTRL_FREQ),
            output_folder=args.output_folder,
            num_drones=1,
            colab=False,
        )
        if args.save
        else None
    )

    try:
        environment.reset()
        total_steps = int(round(args.duration_seconds * environment.CTRL_FREQ))
        start_time = time.perf_counter()

        for step_index in range(total_steps):
            _, reward, _, _, _ = environment.step(normalized_action)

            if logger is not None:
                logger.log(
                    drone=0,
                    timestamp=(step_index + 1) / environment.CTRL_FREQ,
                    state=environment._getDroneStateVector(0),
                    reward=reward,
                    control=control_target,
                )

            if args.gui and args.real_time:
                synchronize_simulation(
                    start_time=start_time,
                    completed_steps=step_index + 1,
                    timestep_seconds=environment.CTRL_TIMESTEP,
                )

        final_state = environment._getDroneStateVector(0)
        print(
            "Final position [m]: "
            f"{np.array2string(final_state[0:3], precision=4)}"
        )
        print(
            "Final RPY [deg]: "
            f"{np.array2string(np.degrees(final_state[7:10]), precision=3)}"
        )
        print(
            "Final linear velocity [m/s]: "
            f"{np.array2string(final_state[10:13], precision=4)}"
        )
    finally:
        environment.close()

    if logger is not None:
        logger.save_as_csv(args.comment)


def parse_arguments() -> argparse.Namespace:
    """Parse standalone PID simulation arguments."""
    parser = argparse.ArgumentParser(
        description="Run the QRLab firmware-like Crazyflie PID controller."
    )

    parser.add_argument(
        "--mode",
        choices=["position", "velocity"],
        default="position",
    )
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument(
        "--initial-position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.1),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--initial-rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument("--battery-voltage", type=float, default=3.7)
    parser.add_argument(
        "--pid-gains",
        type=Path,
        default=None,
        help="Sparse YAML file containing simulation-only PID gain overrides.",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--save",
        default=False,
        type=str2bool,
        help="Save simulation results using QRLab's Logger.",
    )
    parser.add_argument(
        "--output-folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help="Directory used by QRLab's Logger.",
    )
    parser.add_argument(
        "--comment",
        default="pid",
        type=str,
        help="Comment included in the Logger output directory name.",
    )

    position_group = parser.add_argument_group("position command")
    position_group.add_argument(
        "--target-position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    position_group.add_argument("--target-yaw-deg", type=float, default=0.0)

    velocity_group = parser.add_argument_group("velocity command")
    velocity_group.add_argument(
        "--target-velocity",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "VZ"),
    )
    velocity_group.add_argument(
        "--target-yaw-rate-deg-per-second",
        type=float,
        default=0.0,
    )
    velocity_group.add_argument("--body-frame-velocity", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run_pid_simulation(parse_arguments())
