from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from math import radians
from pathlib import Path
from typing import Any, Callable

import numpy as np

from environments.controller_monte_carlo_env import PIDMonteCarloEnv
from environments.pid_simulation_env import (
    PIDSimulationGainOverrides,
    create_pid_simulation_config,
    load_pid_gain_overrides,
)
from environments.utils.enums import ActionType, Physics
from python_scripts.ICRA_27.controller_monte_carlo_study import (
    STUDY_MODES,
    execute_controller_analysis_pipeline,
    parse_study_selection,
)
from python_scripts.ICRA_27.monte_carlo_core import RecoveryCriteria


def create_pid_environment_factory(
    target_position: np.ndarray,
    target_yaw_radians: float,
    episode_length_seconds: float,
    battery_voltage: float,
    gain_overrides: PIDSimulationGainOverrides,
    hard_max_position_error: float | None,
    terminate_on_ground_contact: bool,
) -> Callable[[str], PIDMonteCarloEnv]:
    simulation_config = replace(
        create_pid_simulation_config(
            battery_voltage=battery_voltage,
            body_frame_velocity=False,
        ),
        physics=Physics.PYB_GND,
    )

    def create_environment(study_type: str) -> PIDMonteCarloEnv:
        mode = STUDY_MODES[study_type]
        return PIDMonteCarloEnv(
            action_type=ActionType.POSITION,
            initial_xyz=target_position,
            initial_rpy=np.array(
                [0.0, 0.0, target_yaw_radians],
                dtype=np.float64,
            ),
            config=simulation_config,
            gain_overrides=gain_overrides,
            gui=False,
            record=False,
            target_xyzs=target_position,
            target_yaw_radians=target_yaw_radians,
            episode_length_seconds=episode_length_seconds,
            randomize_initial_conditions=mode["randomize_initial_conditions"],
            dr_on_reset=mode["dr_on_reset"],
            hard_max_position_error=hard_max_position_error,
            terminate_on_ground_contact=terminate_on_ground_contact,
        )

    return create_environment


def create_pid_action_factory(
    target_position: np.ndarray,
    target_yaw_radians: float,
) -> Callable[[PIDMonteCarloEnv], np.ndarray]:
    return lambda environment: environment.create_position_action(
        target_xyz=target_position,
        target_yaw_radians=target_yaw_radians,
    )


def build_recovery_criteria(args: argparse.Namespace) -> RecoveryCriteria:
    return RecoveryCriteria(
        position_tolerance_m=args.position_tol,
        speed_tolerance_m_s=args.speed_tol,
        attitude_tolerance_rad=np.deg2rad(args.attitude_tol_deg),
        angular_speed_tolerance_rad_s=args.angular_speed_tol,
        dwell_time_s=args.dwell_seconds,
        terminal_window_s=args.terminal_window_seconds,
    )


def run_pid_monte_carlo_study(args: argparse.Namespace) -> None:
    target_position = np.asarray(args.target_position, dtype=np.float64)
    target_yaw_radians = radians(args.target_yaw_deg)
    gain_overrides = load_pid_gain_overrides(args.pid_gains)
    environment_factory = create_pid_environment_factory(
        target_position=target_position,
        target_yaw_radians=target_yaw_radians,
        episode_length_seconds=args.episode_seconds,
        battery_voltage=args.battery_voltage,
        gain_overrides=gain_overrides,
        hard_max_position_error=args.hard_max_position_error,
        terminate_on_ground_contact=not args.ignore_ground_contact,
    )
    controller_configuration: dict[str, Any] = {
        "mode": "position",
        "battery_voltage": float(args.battery_voltage),
        "pid_gain_file": args.pid_gains,
        "pid_gain_overrides": asdict(gain_overrides),
        "physics": Physics.PYB_GND.value,
        "plant_parameters_randomized_only": True,
    }

    execute_controller_analysis_pipeline(
        study_types=parse_study_selection(args.study),
        total_trials=args.trials,
        environment_factory=environment_factory,
        action_factory=create_pid_action_factory(
            target_position=target_position,
            target_yaw_radians=target_yaw_radians,
        ),
        controller_name="pid",
        controller_configuration=controller_configuration,
        output_directory=args.output_dir,
        base_seed=args.base_seed,
        confidence_level=args.confidence_level,
        episode_length_seconds=args.episode_seconds,
        recovery_criteria=build_recovery_criteria(args),
        maximum_steps=args.max_steps,
        show_plots=args.show_plots,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo recovery/stabilization study for the firmware-like "
            "Crazyflie PID baseline."
        )
    )
    parser.add_argument(
        "--study",
        choices=["nominal", "ic", "dr", "combined", "all"],
        default="all",
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        default="monte_carlo_results/pid",
    )
    parser.add_argument("--show-plots", action="store_true")

    parser.add_argument(
        "--target-position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)
    parser.add_argument("--battery-voltage", type=float, default=3.7)
    parser.add_argument(
        "--pid-gains",
        type=Path,
        default=Path("helpers/pid_gains.yaml"),
    )

    parser.add_argument("--position-tol", type=float, default=0.10)
    parser.add_argument("--speed-tol", type=float, default=0.20)
    parser.add_argument("--attitude-tol-deg", type=float, default=5.0)
    parser.add_argument("--angular-speed-tol", type=float, default=0.30)
    parser.add_argument("--dwell-seconds", type=float, default=1.0)
    parser.add_argument("--terminal-window-seconds", type=float, default=1.0)
    parser.add_argument("--hard-max-position-error", type=float, default=20.0)
    parser.add_argument("--ignore-ground-contact", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_pid_monte_carlo_study(parse_arguments())
