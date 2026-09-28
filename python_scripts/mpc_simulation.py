#!/usr/bin/env python3

import argparse
import csv
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class MPCSimulationSample:
    time_seconds: float
    position: np.ndarray
    rpy_radians: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity_world: np.ndarray
    motor_rpm: np.ndarray
    commanded_wrench: np.ndarray
    delta_wrench: np.ndarray
    solve_time_seconds: float
    solver_iterations: int
    objective_value: float


def read_simulation_sample(
    environment: MPCSimulationEnv,
    time_seconds: float,
) -> MPCSimulationSample:
    """Read vehicle state and the latest MPC solution."""
    state = environment._getDroneStateVector(0)
    solution = environment.LAST_MPC_SOLUTION
    if solution is None:
        raise RuntimeError("No MPC solution is available after the simulation step.")

    return MPCSimulationSample(
        time_seconds=time_seconds,
        position=state[0:3].copy(),
        rpy_radians=state[7:10].copy(),
        linear_velocity=state[10:13].copy(),
        angular_velocity_world=state[13:16].copy(),
        motor_rpm=state[16:20].copy(),
        commanded_wrench=environment.LAST_COMMANDED_WRENCH.copy(),
        delta_wrench=solution.delta_wrench.copy(),
        solve_time_seconds=solution.solve_time_seconds,
        solver_iterations=solution.iterations,
        objective_value=solution.objective_value,
    )


def simulation_sample_to_csv_row(sample: MPCSimulationSample) -> list[float | int]:
    """Convert one simulation sample to the shared MPC CSV schema."""
    return [
        sample.time_seconds,
        *sample.position,
        *sample.rpy_radians,
        *sample.linear_velocity,
        *sample.angular_velocity_world,
        *sample.motor_rpm,
        *sample.commanded_wrench,
        *sample.delta_wrench,
        sample.solve_time_seconds,
        sample.solver_iterations,
        sample.objective_value,
    ]


def write_samples_to_csv(
    samples: list[MPCSimulationSample],
    output_path: Path,
) -> None:
    """Write MPC simulation signals and solver diagnostics to CSV."""
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
        "commanded_thrust_n",
        "commanded_tau_x_nm",
        "commanded_tau_y_nm",
        "commanded_tau_z_nm",
        "delta_thrust_n",
        "delta_tau_x_nm",
        "delta_tau_y_nm",
        "delta_tau_z_nm",
        "mpc_solve_time_s",
        "mpc_iterations",
        "mpc_objective",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(map(simulation_sample_to_csv_row, samples))


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


def run_mpc_simulation(
    args: argparse.Namespace,
) -> list[MPCSimulationSample]:
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
        initial_rpy=np.radians(np.asarray(args.initial_rpy_deg, dtype=np.float64)),
        config=simulation_config,
        gui=args.gui,
        record=args.record_video,
    )
    command = create_simulation_command(environment=environment, args=args)
    samples: list[MPCSimulationSample] = []

    try:
        environment.reset()
        print_effective_mpc_weights(environment)
        total_steps = int(round(args.duration_seconds * environment.CTRL_FREQ))
        start_time = time.perf_counter()

        for step_index in range(total_steps):
            environment.step(command)
            samples.append(
                read_simulation_sample(
                    environment=environment,
                    time_seconds=(step_index + 1) / environment.CTRL_FREQ,
                )
            )

            if args.gui and args.real_time:
                synchronize_simulation(
                    start_time=start_time,
                    completed_steps=step_index + 1,
                    timestep_seconds=environment.CTRL_TIMESTEP,
                )

        final_state = environment._getDroneStateVector(0)
        solve_times = np.asarray(
            [sample.solve_time_seconds for sample in samples],
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
        if solve_times.size > 0:
            print(
                "MPC solve time [ms] mean/max: "
                f"{1e3 * solve_times.mean():.3f} / {1e3 * solve_times.max():.3f}"
            )
    finally:
        environment.close()

    if args.csv is not None:
        write_samples_to_csv(samples=samples, output_path=args.csv)

    return samples


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
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run_mpc_simulation(parse_arguments())
