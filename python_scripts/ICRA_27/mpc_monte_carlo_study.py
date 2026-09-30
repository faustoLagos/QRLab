from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from math import radians
from pathlib import Path
from typing import Any, Callable

import numpy as np

from controllers.mpc import (
    MPCWeights,
    apply_mpc_weight_overrides,
    create_default_mpc_weights,
    load_mpc_weight_overrides,
)
from environments.controller_monte_carlo_env import MPCMonteCarloEnv
from environments.mpc_simulation_env import create_mpc_simulation_config
from environments.utils.enums import Physics
from python_scripts.ICRA_27.controller_monte_carlo_study import (
    STUDY_MODES,
    execute_controller_analysis_pipeline,
    parse_study_selection,
)
from python_scripts.ICRA_27.monte_carlo_core import RecoveryCriteria


def create_mpc_environment_factory(
    target_position: np.ndarray,
    target_yaw_radians: float,
    episode_length_seconds: float,
    horizon_steps: int,
    effective_weights: MPCWeights,
    hard_max_position_error: float | None,
    terminate_on_ground_contact: bool,
) -> Callable[[str], MPCMonteCarloEnv]:
    simulation_config = replace(
        create_mpc_simulation_config(
            horizon_steps=horizon_steps,
            weights=effective_weights,
        ),
        physics=Physics.PYB_GND,
    )

    def create_environment(study_type: str) -> MPCMonteCarloEnv:
        mode = STUDY_MODES[study_type]
        return MPCMonteCarloEnv(
            initial_xyz=target_position,
            initial_rpy=np.array(
                [0.0, 0.0, target_yaw_radians],
                dtype=np.float64,
            ),
            config=simulation_config,
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


def create_mpc_action_factory(
    target_position: np.ndarray,
    target_yaw_radians: float,
) -> Callable[[MPCMonteCarloEnv], np.ndarray]:
    return lambda environment: environment.create_position_command(
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


def run_mpc_monte_carlo_study(args: argparse.Namespace) -> None:
    target_position = np.asarray(args.target_position, dtype=np.float64)
    target_yaw_radians = radians(args.target_yaw_deg)
    weight_overrides = load_mpc_weight_overrides(args.mpc_weights)
    effective_weights = apply_mpc_weight_overrides(
        default_weights=create_default_mpc_weights(),
        overrides=weight_overrides,
    )
    environment_factory = create_mpc_environment_factory(
        target_position=target_position,
        target_yaw_radians=target_yaw_radians,
        episode_length_seconds=args.episode_seconds,
        horizon_steps=args.horizon_steps,
        effective_weights=effective_weights,
        hard_max_position_error=args.hard_max_position_error,
        terminate_on_ground_contact=not args.ignore_ground_contact,
    )
    controller_configuration: dict[str, Any] = {
        "horizon_steps": int(args.horizon_steps),
        "mpc_weight_file": args.mpc_weights,
        "effective_weights": asdict(effective_weights),
        "physics": Physics.PYB_GND.value,
        "plant_parameters_randomized_only": True,
        "internal_model_randomized": False,
        "control_allocation_randomized": False,
    }

    execute_controller_analysis_pipeline(
        study_types=parse_study_selection(args.study),
        total_trials=args.trials,
        environment_factory=environment_factory,
        action_factory=create_mpc_action_factory(
            target_position=target_position,
            target_yaw_radians=target_yaw_radians,
        ),
        controller_name="mpc",
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
            "Monte Carlo recovery/stabilization study for the constrained "
            "linear MPC Crazyflie baseline."
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
        default="monte_carlo_results/mpc",
    )
    parser.add_argument("--show-plots", action="store_true")

    parser.add_argument("--horizon-steps", type=int, default=20)
    parser.add_argument(
        "--mpc-weights",
        type=Path,
        default=Path("helpers/mpc_weights_example.yaml"),
    )
    parser.add_argument(
        "--target-position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)

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
    run_mpc_monte_carlo_study(parse_arguments())
