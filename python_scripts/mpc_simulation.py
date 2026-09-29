#!/usr/bin/env python3

import argparse
import time
from math import radians
from pathlib import Path

import numpy as np

from controllers.mpc import (
    apply_mpc_weight_overrides,
    create_default_mpc_weights,
    create_input_weight_vector,
    load_mpc_weight_overrides,
)
from environments.mpc_simulation_env import (
    MPCSimulationEnv,
    create_mpc_simulation_config,
)
from helpers.cast import str2bool
from python_scripts.Logger import Logger


DEFAULT_OUTPUT_FOLDER = "results"


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


def create_simulation_command(
    environment: MPCSimulationEnv,
    args: argparse.Namespace,
) -> np.ndarray:
    """Create the physical position-and-yaw command used by the MPC baseline."""
    return environment.create_position_command(
        target_xyz=np.asarray(args.target_position, dtype=np.float64),
        target_yaw_radians=radians(args.target_yaw_deg),
    )


def create_logger_control_target(args: argparse.Namespace) -> np.ndarray:
    """Create the 12-element position/yaw target expected by QRLab's Logger."""
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


def print_effective_mpc_weights(environment: MPCSimulationEnv) -> None:
    """Print the complete effective Q/R configuration used by the solver."""
    workspace = environment.MPC_WORKSPACE
    if workspace is None:
        raise RuntimeError("MPC workspace is not initialized.")

    print(
        "Effective Q diagonal: "
        f"{np.array2string(np.diag(workspace.state_weight_matrix), precision=6)}"
    )
    print(
        "Effective normalized-input R weights: "
        f"{np.array2string(create_input_weight_vector(workspace.config.weights.input), precision=6)}"
    )
    print(
        "Effective physical-input R diagonal: "
        f"{np.array2string(np.diag(workspace.input_weight_matrix), precision=6)}"
    )


def run_mpc_simulation(args: argparse.Namespace) -> None:
    """Run one deterministic constrained linear-MPC simulation."""
    weight_overrides = load_mpc_weight_overrides(args.mpc_weights)
    effective_weights = apply_mpc_weight_overrides(
        default_weights=create_default_mpc_weights(),
        overrides=weight_overrides,
    )
    simulation_config = create_mpc_simulation_config(
        horizon_steps=args.horizon_steps,
        weights=effective_weights,
    )
    environment = MPCSimulationEnv(
        initial_xyz=np.asarray(args.initial_position, dtype=np.float64),
        initial_rpy=np.radians(
            np.asarray(args.initial_rpy_deg, dtype=np.float64)
        ),
        config=simulation_config,
        gui=args.gui,
        record=args.record_video,
    )
    command = create_simulation_command(environment=environment, args=args)
    control_target = create_logger_control_target(args=args)
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
    solve_times: list[float] = []
    solver_iterations: list[int] = []
    objective_values: list[float] = []

    try:
        environment.reset()
        print_effective_mpc_weights(environment)
        total_steps = int(round(args.duration_seconds * environment.CTRL_FREQ))
        start_time = time.perf_counter()

        for step_index in range(total_steps):
            _, reward, _, _, _ = environment.step(command)
            solution = environment.LAST_MPC_SOLUTION
            if solution is None:
                raise RuntimeError(
                    "No MPC solution is available after the simulation step."
                )

            solve_times.append(solution.solve_time_seconds)
            solver_iterations.append(solution.iterations)
            objective_values.append(solution.objective_value)

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
        solve_time_array = np.asarray(
            solve_times,
            dtype=np.float64,
        )
        iteration_array = np.asarray(
            solver_iterations,
            dtype=np.int64,
        )
        objective_array = np.asarray(
            objective_values,
            dtype=np.float64,
        )

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

        if solve_time_array.size > 0:
            print("\nMPC solver statistics")
            print(
                "Solve time [ms] mean/max: "
                f"{1e3 * solve_time_array.mean():.3f} / "
                f"{1e3 * solve_time_array.max():.3f}"
            )
            print(
                "Iterations mean/max: "
                f"{iteration_array.mean():.2f} / "
                f"{iteration_array.max()}"
            )
            print(
                "Objective mean/final: "
                f"{objective_array.mean():.6f} / "
                f"{objective_array[-1]:.6f}"
            )
    finally:
        environment.close()

    if logger is not None:
        logger.save_as_csv(args.comment)


def parse_arguments() -> argparse.Namespace:
    """Parse standalone MPC simulation arguments."""
    parser = argparse.ArgumentParser(
        description="Run the QRLab constrained linear MPC baseline on the CF2X plant."
    )
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--horizon-steps", type=int, default=20)
    parser.add_argument(
        "--mpc-weights",
        type=Path,
        default=None,
        help="Sparse YAML file containing simulation-only MPC Q/R weight overrides.",
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
        "--target-position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)
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
        default="mpc",
        type=str,
        help="Comment included in the Logger output directory name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_mpc_simulation(parse_arguments())
