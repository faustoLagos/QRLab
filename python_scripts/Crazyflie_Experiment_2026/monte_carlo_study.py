"""Driver for Crazyflie Monte Carlo recovery/stabilization studies.

Supported factor-class studies:

- nominal:  fixed nominal initial condition, domain randomization off
- ic:       randomized initial conditions, domain randomization off
- dr:       nominal hover initial condition, domain randomization on
- combined: randomized initial conditions and domain randomization on

Trial i uses the same base seed across studies. MonteCarloEnv splits that seed
into independent IC, DR, and observation-noise streams, enabling paired
comparisons between the factor classes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from environments.monte_carlo_env import MonteCarloEnv
from python_scripts.Crazyflie_Experiment_2026.monte_carlo_core import (
    MonteCarloSummary,
    RecoveryCriteria,
    RolloutResult,
    convert_results_to_dataframe,
    evaluate_all_trials,
    export_dataframe_to_csv,
    summarize_monte_carlo_results,
    summary_to_dictionary,
)
from python_scripts.Crazyflie_Experiment_2026.monte_carlo_plots import (
    generate_binned_probability_plot,
    generate_outcome_fraction_plot,
    generate_recovered_metric_ecdf_plot,
    has_parameter_variation,
    render_projected_outcome_map,
)


STUDY_MODES: Dict[str, Dict[str, bool]] = {
    "nominal": {
        "randomize_initial_conditions": False,
        "dr_on_reset": False,
    },
    "ic": {
        "randomize_initial_conditions": True,
        "dr_on_reset": False,
    },
    "dr": {
        "randomize_initial_conditions": False,
        "dr_on_reset": True,
    },
    "combined": {
        "randomize_initial_conditions": True,
        "dr_on_reset": True,
    },
}

DEFAULT_BATCH_STUDIES = ("ic", "dr", "combined")


def _criteria_dictionary(criteria: RecoveryCriteria) -> dict:
    return {
        "position_tolerance_m": float(criteria.position_tolerance_m),
        "speed_tolerance_m_s": float(criteria.speed_tolerance_m_s),
        "attitude_tolerance_rad": float(criteria.attitude_tolerance_rad),
        "attitude_tolerance_deg": float(np.rad2deg(criteria.attitude_tolerance_rad)),
        "angular_speed_tolerance_rad_s": float(criteria.angular_speed_tolerance_rad_s),
        "dwell_time_s": float(criteria.dwell_time_s),
        "terminal_window_s": float(criteria.terminal_window_s),
    }


def _validate_study_type(study_type: str) -> str:
    study_type = study_type.lower().strip()
    if study_type not in STUDY_MODES:
        raise ValueError(
            f"Unsupported study type {study_type!r}; choose from {sorted(STUDY_MODES)}"
        )
    return study_type


def initialize_evaluation_environment(
    study_type: str,
    episode_length_seconds: float,
    hard_max_position_error: float | None = 20.0,
    terminate_on_ground_contact: bool = True,
) -> MonteCarloEnv:
    study_type = _validate_study_type(study_type)
    mode = STUDY_MODES[study_type]

    return MonteCarloEnv(
        gui=False,
        record=False,
        dr_on_reset=mode["dr_on_reset"],
        randomize_initial_conditions=mode["randomize_initial_conditions"],
        episode_length_seconds=episode_length_seconds,
        hard_max_position_error=hard_max_position_error,
        terminate_on_ground_contact=terminate_on_ground_contact,
    )


def load_trained_policy(model_path: str) -> PPO:
    return PPO.load(model_path)


def _default_max_steps(environment: Any, episode_length_seconds: float) -> int:
    """Guard slightly beyond the environment's own time truncation."""
    return int(ceil(float(episode_length_seconds) * float(environment.CTRL_FREQ))) + 5


def execute_evaluation_trials(
    total_trials: int,
    environment: Any,
    policy: Any,
    maximum_steps: int,
    base_seed: int,
    recovery_criteria: RecoveryCriteria,
) -> List[RolloutResult]:
    return evaluate_all_trials(
        trial_indices=list(range(total_trials)),
        environment=environment,
        policy=policy,
        max_steps=maximum_steps,
        base_seed=base_seed,
        recovery_criteria=recovery_criteria,
    )


def display_statistical_summary(
    study_type: str,
    summary: MonteCarloSummary,
    confidence_level: float,
) -> None:
    print(f"\nMonte Carlo recovery summary — {study_type}")
    print("-" * 52)
    print(f"Trials:                                  {summary.total_trials}")
    print(f"Recovered:                               {summary.recovered_trials}")
    print(f"  Recovered cleanly:                     {summary.recovered_cleanly_trials}")
    print(
        "  Recovered + training-envelope violation: "
        f"{summary.recovered_with_envelope_violation_trials}"
    )
    print(f"Not recovered by horizon:                {summary.not_recovered_by_horizon_trials}")
    print(f"Hard failures:                           {summary.hard_failure_trials}")
    print(f"Incomplete/evaluation guard:             {summary.incomplete_trials}")
    print(f"Recovery rate:                           {summary.recovery_rate:.4f}")
    print(
        f"{100 * confidence_level:.0f}% Wilson CI:                           "
        f"[{summary.recovery_ci_lower:.4f}, {summary.recovery_ci_upper:.4f}]"
    )
    print(
        "Training-envelope violation rate:        "
        f"{summary.training_envelope_violation_rate:.4f}"
    )
    print(
        "Recovered median settling time:          "
        f"{summary.recovered_median_settling_time_s:.4f} s"
    )
    print(
        "Recovered 95% settling time:             "
        f"{summary.recovered_p95_settling_time_s:.4f} s"
    )
    print(
        "Recovered mean terminal RMSE:            "
        f"{summary.recovered_mean_terminal_rmse:.4f} m"
    )
    print(
        "Recovered median terminal RMSE:          "
        f"{summary.recovered_median_terminal_rmse:.4f} m"
    )
    print(
        "Recovered 95% terminal RMSE:             "
        f"{summary.recovered_p95_terminal_rmse:.4f} m"
    )
    print(
        "Recovered mean whole-episode RMSE:       "
        f"{summary.recovered_mean_whole_episode_rmse:.4f} m"
    )
    print(
        "Recovered 95% maximum position error:    "
        f"{summary.recovered_p95_maximum_position_error:.4f} m"
    )


def _save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _study_metadata(
    study_type: str,
    environment: MonteCarloEnv,
    policy_path: str,
    total_trials: int,
    maximum_steps: int,
    base_seed: int,
    confidence_level: float,
    recovery_criteria: RecoveryCriteria,
) -> dict:
    mode = STUDY_MODES[study_type]
    criteria = _criteria_dictionary(recovery_criteria)

    return {
        "study_type": study_type,
        "policy_path": policy_path,
        "total_trials": int(total_trials),
        "maximum_steps_guard": int(maximum_steps),
        "base_seed": int(base_seed),
        "confidence_level": float(confidence_level),
        "episode_length_seconds": float(environment.EPISODE_LENGTH_SECONDS),
        "control_frequency_hz": float(environment.CTRL_FREQ),
        "physics_frequency_hz": float(environment.PYB_FREQ),
        "policy_deterministic": True,
        "randomize_initial_conditions": mode["randomize_initial_conditions"],
        "domain_randomization_on_reset": mode["dr_on_reset"],
        "recovery_criteria": criteria,
        "training_envelope": {
            "logged_only": True,
            "min_altitude_m": float(environment.TRAINING_MIN_ALTITUDE),
            "max_position_error_m": float(environment.TRAINING_MAX_POSITION_ERROR),
            "max_roll_pitch_deg": float(
                np.rad2deg(environment.TRAINING_MAX_ROLL_PITCH_RAD)
            ),
            "max_speed_m_s": float(environment.TRAINING_MAX_SPEED),
            "max_angular_speed_rad_s": float(environment.TRAINING_MAX_ANGULAR_SPEED),
        },
        "hard_failure_criteria": {
            "terminate_on_ground_contact": bool(environment.TERMINATE_ON_GROUND_CONTACT),
            "hard_max_position_error_m": environment.HARD_MAX_POSITION_ERROR,
            "non_finite_state_terminates": True,
        },
        "rng_design": {
            "trial_seed": "base_seed + trial_index",
            "separate_ic_dr_noise_streams": True,
            "paired_across_factor_class_studies": True,
        },
    }


def _save_probability_plot_and_table(
    dataframe: pd.DataFrame,
    parameter_column: str,
    event_column: str,
    event_label: str,
    stem: str,
    output_directory: Path,
    show: bool,
    bins: int = 10,
) -> None:
    if not has_parameter_variation(dataframe, parameter_column):
        return

    grouped = generate_binned_probability_plot(
        dataframe=dataframe,
        parameter_column=parameter_column,
        event_column=event_column,
        event_label=event_label,
        bins=bins,
        output_path=str(output_directory / f"{stem}.png"),
        show=show,
    )
    grouped.to_csv(output_directory / f"{stem}.csv", index=False)


def generate_evaluation_visualizations(
    dataframe: pd.DataFrame,
    output_directory: Path,
    show: bool,
) -> None:
    generate_outcome_fraction_plot(
        dataframe,
        output_path=str(output_directory / "outcome_fractions.png"),
        show=show,
    ).to_csv(output_directory / "outcome_fractions.csv", index=False)

    generate_recovered_metric_ecdf_plot(
        dataframe=dataframe,
        metric_column="terminal_rmse_position",
        metric_label="Terminal position RMSE (m)",
        output_path=str(output_directory / "recovered_terminal_rmse_ecdf.png"),
        show=show,
    )
    generate_recovered_metric_ecdf_plot(
        dataframe=dataframe,
        metric_column="settling_time_s",
        metric_label="Settling time (s)",
        output_path=str(output_directory / "recovered_settling_time_ecdf.png"),
        show=show,
    )
    generate_recovered_metric_ecdf_plot(
        dataframe=dataframe,
        metric_column="whole_episode_rmse_position",
        metric_label="Whole-episode position RMSE (m)",
        output_path=str(output_directory / "recovered_whole_episode_rmse_ecdf.png"),
        show=show,
    )

    if (
        has_parameter_variation(dataframe, "dr_mass")
        and has_parameter_variation(dataframe, "dr_kf")
    ):
        render_projected_outcome_map(
            dataframe=dataframe,
            x_axis_column="dr_mass",
            y_axis_column="dr_kf",
            color_metric_column="terminal_rmse_position",
            color_metric_label="Terminal position RMSE (m)",
            output_path=str(output_directory / "dr_mass_vs_kf_recovery_outcomes.png"),
            show=show,
        )

    if (
        has_parameter_variation(dataframe, "ic_x0")
        and has_parameter_variation(dataframe, "ic_y0")
    ):
        render_projected_outcome_map(
            dataframe=dataframe,
            x_axis_column="ic_x0",
            y_axis_column="ic_y0",
            color_metric_column="terminal_rmse_position",
            color_metric_label="Terminal position RMSE (m)",
            output_path=str(output_directory / "initial_xy_recovery_outcomes.png"),
            show=show,
        )

    for parameter in ("dr_mass", "dr_kf", "dr_km", "dr_hover_rpm"):
        _save_probability_plot_and_table(
            dataframe=dataframe,
            parameter_column=parameter,
            event_column="recovered",
            event_label="Recovery probability",
            stem=f"recovery_probability_vs_{parameter}",
            output_directory=output_directory,
            show=show,
        )
        _save_probability_plot_and_table(
            dataframe=dataframe,
            parameter_column=parameter,
            event_column="training_envelope_violated",
            event_label="Training-envelope violation probability",
            stem=f"training_envelope_violation_probability_vs_{parameter}",
            output_directory=output_directory,
            show=show,
        )


def execute_single_study(
    study_type: str,
    total_trials: int,
    policy: Any,
    policy_path: str,
    output_directory: Path,
    base_seed: int,
    confidence_level: float,
    episode_length_seconds: float,
    recovery_criteria: RecoveryCriteria,
    maximum_steps: int | None = None,
    show_plots: bool = False,
    hard_max_position_error: float | None = 20.0,
    terminate_on_ground_contact: bool = True,
) -> MonteCarloSummary:
    study_type = _validate_study_type(study_type)
    study_output = output_directory / study_type
    study_output.mkdir(parents=True, exist_ok=True)

    environment = initialize_evaluation_environment(
        study_type=study_type,
        episode_length_seconds=episode_length_seconds,
        hard_max_position_error=hard_max_position_error,
        terminate_on_ground_contact=terminate_on_ground_contact,
    )

    try:
        effective_max_steps = (
            _default_max_steps(environment, episode_length_seconds)
            if maximum_steps is None
            else int(maximum_steps)
        )

        results = execute_evaluation_trials(
            total_trials=total_trials,
            environment=environment,
            policy=policy,
            maximum_steps=effective_max_steps,
            base_seed=base_seed,
            recovery_criteria=recovery_criteria,
        )
        summary = summarize_monte_carlo_results(
            results, confidence_level=confidence_level
        )
        display_statistical_summary(study_type, summary, confidence_level)

        dataframe = convert_results_to_dataframe(results)
        export_dataframe_to_csv(
            dataframe, str(study_output / "monte_carlo_trials.csv")
        )

        metadata = _study_metadata(
            study_type=study_type,
            environment=environment,
            policy_path=policy_path,
            total_trials=total_trials,
            maximum_steps=effective_max_steps,
            base_seed=base_seed,
            confidence_level=confidence_level,
            recovery_criteria=recovery_criteria,
        )
        _save_json(
            {
                "summary": summary_to_dictionary(summary),
                "study_configuration": metadata,
            },
            study_output / "monte_carlo_summary.json",
        )

        generate_evaluation_visualizations(
            dataframe=dataframe,
            output_directory=study_output,
            show=show_plots,
        )
        return summary
    finally:
        environment.close()


def _comparison_row(study_type: str, summary: MonteCarloSummary) -> dict:
    row = {"study_type": study_type}
    row.update(summary_to_dictionary(summary))
    return row


def execute_analysis_pipeline(
    study_types: Sequence[str],
    total_trials: int,
    policy_path: str,
    output_directory: str,
    base_seed: int = 12345,
    confidence_level: float = 0.95,
    episode_length_seconds: float = 10.0,
    recovery_criteria: RecoveryCriteria | None = None,
    maximum_steps: int | None = None,
    show_plots: bool = False,
    hard_max_position_error: float | None = 20.0,
    terminate_on_ground_contact: bool = True,
) -> pd.DataFrame:
    criteria = recovery_criteria or RecoveryCriteria()
    criteria.validate()

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = load_trained_policy(policy_path)
    summaries = []

    for study_type in study_types:
        normalized = _validate_study_type(study_type)
        summary = execute_single_study(
            study_type=normalized,
            total_trials=total_trials,
            policy=policy,
            policy_path=policy_path,
            output_directory=output_dir,
            base_seed=base_seed,
            confidence_level=confidence_level,
            episode_length_seconds=episode_length_seconds,
            recovery_criteria=criteria,
            maximum_steps=maximum_steps,
            show_plots=show_plots,
            hard_max_position_error=hard_max_position_error,
            terminate_on_ground_contact=terminate_on_ground_contact,
        )
        summaries.append(_comparison_row(normalized, summary))

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(output_dir / "study_comparison_summary.csv", index=False)
    _save_json(
        {
            "studies": summaries,
            "shared_configuration": {
                "policy_path": policy_path,
                "total_trials_per_study": int(total_trials),
                "base_seed": int(base_seed),
                "confidence_level": float(confidence_level),
                "episode_length_seconds": float(episode_length_seconds),
                "recovery_criteria": _criteria_dictionary(criteria),
            },
        },
        output_dir / "study_comparison_summary.json",
    )
    return comparison


def _parse_study_selection(value: str) -> Sequence[str]:
    value = value.lower().strip()
    if value == "all":
        return DEFAULT_BATCH_STUDIES
    return (_validate_study_type(value),)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo recovery/stabilization study for a Crazyflie PPO controller"
    )
    parser.add_argument("--model", required=True, help="Path to trained Stable-Baselines3 PPO model")
    parser.add_argument(
        "--study",
        choices=["nominal", "ic", "dr", "combined", "all"],
        default="all",
        help="'all' runs IC-only, DR-only, and combined with paired trial seeds",
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Emergency guard only; default is just beyond the environment time horizon",
    )
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--output-dir", default="monte_carlo_results")
    parser.add_argument("--show-plots", action="store_true")

    parser.add_argument("--position-tol", type=float, default=0.10)
    parser.add_argument("--speed-tol", type=float, default=0.20)
    parser.add_argument("--attitude-tol-deg", type=float, default=5.0)
    parser.add_argument("--angular-speed-tol", type=float, default=0.30)
    parser.add_argument("--dwell-seconds", type=float, default=1.0)
    parser.add_argument("--terminal-window-seconds", type=float, default=1.0)

    parser.add_argument(
        "--hard-max-position-error",
        type=float,
        default=20.0,
        help="Catastrophic-divergence guard in metres; not a performance threshold",
    )
    parser.add_argument(
        "--ignore-ground-contact",
        action="store_true",
        help="Do not hard-terminate on ground contact (normally not recommended)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    criteria = RecoveryCriteria(
        position_tolerance_m=args.position_tol,
        speed_tolerance_m_s=args.speed_tol,
        attitude_tolerance_rad=np.deg2rad(args.attitude_tol_deg),
        angular_speed_tolerance_rad_s=args.angular_speed_tol,
        dwell_time_s=args.dwell_seconds,
        terminal_window_s=args.terminal_window_seconds,
    )

    execute_analysis_pipeline(
        study_types=_parse_study_selection(args.study),
        total_trials=args.trials,
        policy_path=args.model,
        output_directory=args.output_dir,
        base_seed=args.base_seed,
        confidence_level=args.confidence_level,
        episode_length_seconds=args.episode_seconds,
        recovery_criteria=criteria,
        maximum_steps=args.max_steps,
        show_plots=args.show_plots,
        hard_max_position_error=args.hard_max_position_error,
        terminate_on_ground_contact=not args.ignore_ground_contact,
    )
