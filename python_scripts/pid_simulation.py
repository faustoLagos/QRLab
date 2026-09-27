#!/usr/bin/env python3

import argparse
import csv
import time
from dataclasses import dataclass
from math import radians
from pathlib import Path

import numpy as np

from environments.pid_simulation_env import (
    PIDSimulationEnv,
    create_pid_simulation_config,
    load_pid_gain_overrides,
)
from environments.utils.enums import ActionType


@dataclass(frozen=True)
class SimulationSample:
    time_seconds: float
    position: np.ndarray
    rpy_radians: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity_world: np.ndarray
    motor_rpm: np.ndarray
    desired_roll_degrees: float
    desired_pitch_degrees: float
    desired_velocity_body: np.ndarray
    desired_body_rates_degrees_per_second: np.ndarray
    legacy_control: np.ndarray
    filtered_battery_voltage: float


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


def read_simulation_sample(
    environment: PIDSimulationEnv,
    time_seconds: float,
) -> SimulationSample:
    """Read vehicle state and held controller outputs."""
    state = environment._getDroneStateVector(0)
    runtime_state = environment.CRAZYFLIE_RUNTIME_STATES[0]
    outer_output = runtime_state.held_outer_output
    legacy_control = runtime_state.held_legacy_control

    return SimulationSample(
        time_seconds=time_seconds,
        position=state[0:3].copy(),
        rpy_radians=state[7:10].copy(),
        linear_velocity=state[10:13].copy(),
        angular_velocity_world=state[13:16].copy(),
        motor_rpm=state[16:20].copy(),
        desired_roll_degrees=float(outer_output.roll_degrees),
        desired_pitch_degrees=float(outer_output.pitch_degrees),
        desired_velocity_body=np.array(
            [
                outer_output.velocity_setpoint_body_x,
                outer_output.velocity_setpoint_body_y,
                outer_output.velocity_setpoint_z,
            ],
            dtype=np.float64,
        ),
        desired_body_rates_degrees_per_second=np.array(
            [
                legacy_control.desired_body_rates.roll_degrees_per_second,
                legacy_control.desired_body_rates.pitch_degrees_per_second,
                legacy_control.desired_body_rates.yaw_degrees_per_second,
            ],
            dtype=np.float64,
        ),
        legacy_control=np.array(
            [
                legacy_control.thrust,
                legacy_control.roll,
                legacy_control.pitch,
                legacy_control.yaw,
            ],
            dtype=np.float64,
        ),
        filtered_battery_voltage=float(runtime_state.battery.supply_voltage),
    )


def write_samples_to_csv(
    samples: list[SimulationSample],
    output_path: Path,
) -> None:
    """Write PID simulation signals to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "vx_m_s",
        "vy_m_s",
        "vz_m_s",
        "wx_world_rad_s",
        "wy_world_rad_s",
        "wz_world_rad_s",
        "rpm_1",
        "rpm_2",
        "rpm_3",
        "rpm_4",
        "desired_roll_deg",
        "desired_pitch_deg",
        "desired_vx_body_m_s",
        "desired_vy_body_m_s",
        "desired_vz_m_s",
        "desired_roll_rate_deg_s",
        "desired_pitch_rate_deg_s",
        "desired_yaw_rate_deg_s",
        "legacy_thrust",
        "legacy_roll",
        "legacy_pitch",
        "legacy_yaw",
        "filtered_battery_voltage_v",
    ]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for sample in samples:
            writer.writerow(
                [
                    sample.time_seconds,
                    *sample.position,
                    *sample.rpy_radians,
                    *sample.linear_velocity,
                    *sample.angular_velocity_world,
                    *sample.motor_rpm,
                    sample.desired_roll_degrees,
                    sample.desired_pitch_degrees,
                    *sample.desired_velocity_body,
                    *sample.desired_body_rates_degrees_per_second,
                    *sample.legacy_control,
                    sample.filtered_battery_voltage,
                ]
            )


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


def run_pid_simulation(
    args: argparse.Namespace,
) -> list[SimulationSample]:
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
    normalized_action = create_simulation_action(
        environment=environment,
        args=args,
    )
    samples: list[SimulationSample] = []

    try:
        environment.reset()

        total_steps = int(round(args.duration_seconds * environment.CTRL_FREQ))
        start_time = time.perf_counter()

        for step_index in range(total_steps):
            environment.step(normalized_action)

            sample_time = (step_index + 1) / environment.CTRL_FREQ
            samples.append(
                read_simulation_sample(
                    environment=environment,
                    time_seconds=sample_time,
                )
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

    if args.csv is not None:
        write_samples_to_csv(
            samples=samples,
            output_path=args.csv,
        )

    return samples


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
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=10.0,
    )
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
    parser.add_argument(
        "--battery-voltage",
        type=float,
        default=3.7,
    )
    parser.add_argument(
        "--pid-gains",
        type=Path,
        default=None,
        help="Sparse YAML file containing simulation-only PID gain overrides.",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--csv", type=Path)

    position_group = parser.add_argument_group("position command")
    position_group.add_argument(
        "--target-position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    position_group.add_argument(
        "--target-yaw-deg",
        type=float,
        default=0.0,
    )

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
    velocity_group.add_argument(
        "--body-frame-velocity",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_pid_simulation(parse_arguments())
