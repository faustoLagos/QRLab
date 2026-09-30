from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from python_scripts.ICRA_27.monte_carlo_core import (
    MonteCarloSummary,
    RecoveryCriteria,
    convert_results_to_dataframe,
    evaluate_all_trials,
    export_dataframe_to_csv,
    summarize_monte_carlo_results,
    summary_to_dictionary,
)
from python_scripts.ICRA_27.monte_carlo_plots import (
    generate_binned_probability_plot,
    generate_outcome_fraction_plot,
    generate_recovered_metric_ecdf_plot,
    has_parameter_variation,
    render_projected_outcome_map,
)


STUDY_MODES: dict[str, dict[str, bool]] = {
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


@dataclass(frozen=True)
class ConstantCommandController:
    action: np.ndarray

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        del observation, deterministic
        return np.asarray(self.action, dtype=np.float64).copy(), None


def validate_study_type(study_type: str) -> str:
    normalized = study_type.lower().strip()
    if normalized not in STUDY_MODES:
        raise ValueError(
            f"Unsupported study type {normalized!r}; choose from {sorted(STUDY_MODES)}"
        )
    return normalized


def parse_study_selection(value: str) -> Sequence[str]:
    normalized = value.lower().strip()
    return (
        DEFAULT_BATCH_STUDIES
        if normalized == "all"
        else (validate_study_type(normalized),)
    )


def criteria_dictionary(criteria: RecoveryCriteria) -> dict[str, float]:
    return {
        "position_tolerance_m": float(criteria.position_tolerance_m),
        "speed_tolerance_m_s": float(criteria.speed_tolerance_m_s),
        "attitude_tolerance_rad": float(criteria.attitude_tolerance_rad),
        "attitude_tolerance_deg": float(
            np.rad2deg(criteria.attitude_tolerance_rad)
        ),
        "angular_speed_tolerance_rad_s": float(
            criteria.angular_speed_tolerance_rad_s
        ),
        "dwell_time_s": float(criteria.dwell_time_s),
        "terminal_window_s": float(criteria.terminal_window_s),
    }


def default_max_steps(environment: Any, episode_length_seconds: float) -> int:
    return int(
        ceil(float(episode_length_seconds) * float(environment.CTRL_FREQ))
    ) + 5


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def save_json(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_value(payload), indent=2),
        encoding="utf-8",
    )


def display_statistical_summary(
    controller_name: str,
    study_type: str,
    summary: MonteCarloSummary,
    confidence_level: float,
) -> None:
    print(f"\n{controller_name} Monte Carlo recovery summary — {study_type}")
    print("-" * 60)
    print(f"Trials:                                  {summary.total_trials}")
    print(f"Recovered:                               {summary.recovered_trials}")
    print(f"  Recovered cleanly:                     {summary.recovered_cleanly_trials}")
    print(
        "  Recovered + training-envelope violation: "
        f"{summary.recovered_with_envelope_violation_trials}"
    )
    print(
        "Not recovered by horizon:                "
        f"{summary.not_recovered_by_horizon_trials}"
    )
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
        "Recovered 95% terminal RMSE:             "
        f"{summary.recovered_p95_terminal_rmse:.4f} m"
    )


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

    metric_plots = (
        (
            "terminal_rmse_position",
            "Terminal position RMSE (m)",
            "recovered_terminal_rmse_ecdf.png",
        ),
        (
            "settling_time_s",
            "Settling time (s)",
            "recovered_settling_time_ecdf.png",
        ),
        (
            "whole_episode_rmse_position",
            "Whole-episode position RMSE (m)",
            "recovered_whole_episode_rmse_ecdf.png",
        ),
    )
    tuple(
        map(
            lambda plot: generate_recovered_metric_ecdf_plot(
                dataframe=dataframe,
                metric_column=plot[0],
                metric_label=plot[1],
                output_path=str(output_directory / plot[2]),
                show=show,
            ),
            metric_plots,
        )
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
            output_path=str(
                output_directory / "dr_mass_vs_kf_recovery_outcomes.png"
            ),
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
            output_path=str(
                output_directory / "initial_xy_recovery_outcomes.png"
            ),
            show=show,
        )

    parameter_names = ("dr_mass", "dr_kf", "dr_km", "dr_hover_rpm")
    tuple(
        map(
            lambda parameter: (
                _save_probability_plot_and_table(
                    dataframe=dataframe,
                    parameter_column=parameter,
                    event_column="recovered",
                    event_label="Recovery probability",
                    stem=f"recovery_probability_vs_{parameter}",
                    output_directory=output_directory,
                    show=show,
                ),
                _save_probability_plot_and_table(
                    dataframe=dataframe,
                    parameter_column=parameter,
                    event_column="training_envelope_violated",
                    event_label="Training-envelope violation probability",
                    stem=(
                        "training_envelope_violation_probability_vs_"
                        f"{parameter}"
                    ),
                    output_directory=output_directory,
                    show=show,
                ),
            ),
            parameter_names,
        )
    )


def study_metadata(
    controller_name: str,
    controller_configuration: Mapping[str, Any],
    study_type: str,
    environment: Any,
    total_trials: int,
    maximum_steps: int,
    base_seed: int,
    confidence_level: float,
    recovery_criteria: RecoveryCriteria,
) -> dict[str, Any]:
    mode = STUDY_MODES[study_type]
    return {
        "controller": controller_name,
        "controller_configuration": dict(controller_configuration),
        "study_type": study_type,
        "total_trials": int(total_trials),
        "maximum_steps_guard": int(maximum_steps),
        "base_seed": int(base_seed),
        "confidence_level": float(confidence_level),
        "episode_length_seconds": float(environment.EPISODE_LENGTH_SECONDS),
        "control_frequency_hz": float(environment.CTRL_FREQ),
        "physics_frequency_hz": float(environment.PYB_FREQ),
        "target_position_m": np.asarray(environment.TARGET_POS, dtype=float),
        "target_quaternion_xyzw": np.asarray(
            environment.TARGET_QUATERNION,
            dtype=float,
        ),
        "randomize_initial_conditions": mode["randomize_initial_conditions"],
        "domain_randomization_on_reset": mode["dr_on_reset"],
        "recovery_criteria": criteria_dictionary(recovery_criteria),
        "training_envelope": {
            "logged_only": True,
            "min_altitude_m": float(environment.TRAINING_MIN_ALTITUDE),
            "max_position_error_m": float(
                environment.TRAINING_MAX_POSITION_ERROR
            ),
            "max_roll_pitch_deg": float(
                np.rad2deg(environment.TRAINING_MAX_ROLL_PITCH_RAD)
            ),
            "max_speed_m_s": float(environment.TRAINING_MAX_SPEED),
            "max_angular_speed_rad_s": float(
                environment.TRAINING_MAX_ANGULAR_SPEED
            ),
        },
        "hard_failure_criteria": {
            "terminate_on_ground_contact": bool(
                environment.TERMINATE_ON_GROUND_CONTACT
            ),
            "hard_max_position_error_m": environment.HARD_MAX_POSITION_ERROR,
            "non_finite_state_terminates": True,
            **dict(getattr(environment, "CONTROLLER_FAILURE_CRITERIA", {})),
        },
        "rng_design": {
            "trial_seed": "base_seed + trial_index",
            "ic_stream_matches_trained_policy_study": True,
            "dr_sampler_matches_domain_randomization_mixin": True,
            "paired_across_factor_class_studies": True,
            "paired_across_controllers_when_base_seed_matches": True,
        },
    }


def execute_single_controller_study(
    study_type: str,
    total_trials: int,
    environment_factory: Callable[[str], Any],
    action_factory: Callable[[Any], np.ndarray],
    controller_name: str,
    controller_configuration: Mapping[str, Any],
    output_directory: Path,
    base_seed: int,
    confidence_level: float,
    episode_length_seconds: float,
    recovery_criteria: RecoveryCriteria,
    maximum_steps: int | None = None,
    show_plots: bool = False,
) -> MonteCarloSummary:
    normalized_study_type = validate_study_type(study_type)
    study_output = output_directory / normalized_study_type
    study_output.mkdir(parents=True, exist_ok=True)
    environment = environment_factory(normalized_study_type)

    try:
        effective_max_steps = (
            default_max_steps(environment, episode_length_seconds)
            if maximum_steps is None
            else int(maximum_steps)
        )
        controller = ConstantCommandController(action=action_factory(environment))
        results = evaluate_all_trials(
            trial_indices=list(range(total_trials)),
            environment=environment,
            policy=controller,
            max_steps=effective_max_steps,
            base_seed=base_seed,
            recovery_criteria=recovery_criteria,
        )
        summary = summarize_monte_carlo_results(
            results,
            confidence_level=confidence_level,
        )
        display_statistical_summary(
            controller_name=controller_name,
            study_type=normalized_study_type,
            summary=summary,
            confidence_level=confidence_level,
        )

        dataframe = convert_results_to_dataframe(results)
        export_dataframe_to_csv(
            dataframe,
            str(study_output / "monte_carlo_trials.csv"),
        )
        save_json(
            {
                "summary": summary_to_dictionary(summary),
                "study_configuration": study_metadata(
                    controller_name=controller_name,
                    controller_configuration=controller_configuration,
                    study_type=normalized_study_type,
                    environment=environment,
                    total_trials=total_trials,
                    maximum_steps=effective_max_steps,
                    base_seed=base_seed,
                    confidence_level=confidence_level,
                    recovery_criteria=recovery_criteria,
                ),
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


def _comparison_row(
    study_type: str,
    summary: MonteCarloSummary,
) -> dict[str, Any]:
    return {
        "study_type": study_type,
        **summary_to_dictionary(summary),
    }


def execute_controller_analysis_pipeline(
    study_types: Sequence[str],
    total_trials: int,
    environment_factory: Callable[[str], Any],
    action_factory: Callable[[Any], np.ndarray],
    controller_name: str,
    controller_configuration: Mapping[str, Any],
    output_directory: str,
    base_seed: int = 12345,
    confidence_level: float = 0.95,
    episode_length_seconds: float = 20.0,
    recovery_criteria: RecoveryCriteria | None = None,
    maximum_steps: int | None = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    criteria = recovery_criteria or RecoveryCriteria()
    criteria.validate()
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_studies = tuple(map(validate_study_type, study_types))
    summaries = tuple(
        map(
            lambda study_type: (
                study_type,
                execute_single_controller_study(
                    study_type=study_type,
                    total_trials=total_trials,
                    environment_factory=environment_factory,
                    action_factory=action_factory,
                    controller_name=controller_name,
                    controller_configuration=controller_configuration,
                    output_directory=output_path,
                    base_seed=base_seed,
                    confidence_level=confidence_level,
                    episode_length_seconds=episode_length_seconds,
                    recovery_criteria=criteria,
                    maximum_steps=maximum_steps,
                    show_plots=show_plots,
                ),
            ),
            normalized_studies,
        )
    )
    rows = tuple(map(lambda item: _comparison_row(*item), summaries))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_path / "study_comparison_summary.csv", index=False)
    save_json(
        {
            "studies": rows,
            "shared_configuration": {
                "controller": controller_name,
                "controller_configuration": dict(controller_configuration),
                "total_trials_per_study": int(total_trials),
                "base_seed": int(base_seed),
                "confidence_level": float(confidence_level),
                "episode_length_seconds": float(episode_length_seconds),
                "recovery_criteria": criteria_dictionary(criteria),
            },
        },
        output_path / "study_comparison_summary.json",
    )
    return comparison
