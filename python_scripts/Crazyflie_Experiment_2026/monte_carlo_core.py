"""Core utilities for Crazyflie Monte Carlo recovery/stabilization studies.

This module deliberately separates three concepts:

1. hard failure: a physical/numerical event that terminates the rollout,
2. training-envelope violation: a diagnostic event that is logged but does not
   determine recovery success,
3. recovery: entry into a configurable stabilization set for a required dwell
   time before the evaluation horizon ends.

The corresponding MonteCarloEnv is responsible for hard-failure termination and
for reporting training-envelope violations through ``info``. This module is
responsible for deciding whether the closed loop actually recovered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, sqrt
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Evaluation configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryCriteria:
    """Definition of the stabilization set used to declare recovery.

    The defaults are evaluation choices, not simulator safety limits. They can
    be changed from the study driver without modifying this module.
    """

    position_tolerance_m: float = 0.10
    speed_tolerance_m_s: float = 0.20
    attitude_tolerance_rad: float = np.deg2rad(5.0)
    angular_speed_tolerance_rad_s: float = 0.30
    dwell_time_s: float = 1.0
    terminal_window_s: float = 1.0

    def validate(self) -> None:
        positive_values = {
            "position_tolerance_m": self.position_tolerance_m,
            "speed_tolerance_m_s": self.speed_tolerance_m_s,
            "attitude_tolerance_rad": self.attitude_tolerance_rad,
            "angular_speed_tolerance_rad_s": self.angular_speed_tolerance_rad_s,
            "dwell_time_s": self.dwell_time_s,
            "terminal_window_s": self.terminal_window_s,
        }
        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")


# -----------------------------------------------------------------------------
# Per-rollout data
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceResult:
    position_errors: np.ndarray
    stable_history: np.ndarray

    recovered: bool
    outcome: str
    hard_failure: bool
    hard_failure_reasons: Tuple[str, ...]
    training_envelope_violations: Tuple[str, ...]

    terminated: bool
    truncated: bool
    max_steps_reached: bool
    steps: int
    simulation_time_s: float

    settling_time_s: float
    terminal_rmse_position: float
    terminal_mean_position_error: float
    terminal_max_position_error: float

    final_position_error: float
    final_speed: float
    final_angular_speed: float
    final_attitude_error_rad: float

    max_speed: float
    max_angular_speed: float
    max_abs_roll_rad: float
    max_abs_pitch_rad: float
    max_attitude_error_rad: float

    @property
    def training_envelope_violated(self) -> bool:
        return bool(self.training_envelope_violations)


@dataclass(frozen=True)
class RolloutResult:
    trial_index: int
    seed: int

    recovered: bool
    outcome: str
    hard_failure: bool
    hard_failure_reasons: Tuple[str, ...]
    training_envelope_violations: Tuple[str, ...]

    terminated: bool
    truncated: bool
    max_steps_reached: bool
    steps: int
    simulation_time_s: float

    position_errors: np.ndarray
    stable_history: np.ndarray
    initial_conditions: Dict[str, Any]
    dr_parameters: Dict[str, Any]

    settling_time_s: float
    terminal_rmse_position: float
    terminal_mean_position_error: float
    terminal_max_position_error: float

    final_position_error: float
    final_speed: float
    final_angular_speed: float
    final_attitude_error_rad: float

    max_speed: float
    max_angular_speed: float
    max_abs_roll_rad: float
    max_abs_pitch_rad: float
    max_attitude_error_rad: float

    @property
    def training_envelope_violated(self) -> bool:
        return bool(self.training_envelope_violations)

    @property
    def recovered_cleanly(self) -> bool:
        return self.outcome == "recovered_cleanly"

    @property
    def recovered_with_training_envelope_violation(self) -> bool:
        return self.outcome == "recovered_with_training_envelope_violation"

    @property
    def initial_error(self) -> float:
        return float(self.position_errors[0]) if self.position_errors.size else float("nan")

    @property
    def whole_episode_rmse_position(self) -> float:
        if self.position_errors.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(self.position_errors))))

    @property
    def whole_episode_mean_position_error(self) -> float:
        if self.position_errors.size == 0:
            return float("nan")
        return float(np.mean(self.position_errors))

    @property
    def maximum_position_error(self) -> float:
        if self.position_errors.size == 0:
            return float("nan")
        return float(np.max(self.position_errors))

    # Temporary compatibility aliases for the existing study/plot code.
    # They can be removed once monte_carlo_study.py and monte_carlo_plots.py
    # have been refactored to use recovery terminology directly.
    @property
    def survived(self) -> bool:
        return self.recovered

    @property
    def tracking_errors(self) -> np.ndarray:
        return self.position_errors

    @property
    def rmse_position(self) -> float:
        return self.whole_episode_rmse_position

    @property
    def mean_position_error(self) -> float:
        return self.whole_episode_mean_position_error


@dataclass(frozen=True)
class MonteCarloSummary:
    total_trials: int
    recovered_trials: int
    recovered_cleanly_trials: int
    recovered_with_envelope_violation_trials: int
    not_recovered_by_horizon_trials: int
    hard_failure_trials: int
    incomplete_trials: int

    recovery_rate: float
    recovery_ci_level: float
    recovery_ci_lower: float
    recovery_ci_upper: float

    training_envelope_violation_trials: int
    training_envelope_violation_rate: float

    recovered_median_settling_time_s: float
    recovered_p95_settling_time_s: float
    recovered_mean_terminal_rmse: float
    recovered_median_terminal_rmse: float
    recovered_p95_terminal_rmse: float

    recovered_mean_whole_episode_rmse: float
    recovered_median_whole_episode_rmse: float
    recovered_p95_whole_episode_rmse: float
    recovered_p95_maximum_position_error: float

    # Compatibility properties for the current monte_carlo_study.py.
    @property
    def successful_trials(self) -> int:
        return self.recovered_trials

    @property
    def failed_trials(self) -> int:
        return self.total_trials - self.recovered_trials

    @property
    def survival_rate(self) -> float:
        return self.recovery_rate

    @property
    def survival_ci_level(self) -> float:
        return self.recovery_ci_level

    @property
    def survival_ci_lower(self) -> float:
        return self.recovery_ci_lower

    @property
    def survival_ci_upper(self) -> float:
        return self.recovery_ci_upper

    @property
    def survivor_mean_rmse(self) -> float:
        return self.recovered_mean_whole_episode_rmse

    @property
    def survivor_median_rmse(self) -> float:
        return self.recovered_median_whole_episode_rmse

    @property
    def survivor_p95_rmse(self) -> float:
        return self.recovered_p95_whole_episode_rmse

    @property
    def survivor_p95_maximum_error(self) -> float:
        return self.recovered_p95_maximum_position_error

    @property
    def mean_failure_time_s(self) -> float:
        # Kept only so the existing display code does not break. Failure time is
        # not a primary metric in the recovery formulation and should disappear
        # when monte_carlo_study.py is refactored.
        return float("nan")


# -----------------------------------------------------------------------------
# State metrics and stabilization logic
# -----------------------------------------------------------------------------


def compute_position_error(current_state: np.ndarray, target_position: np.ndarray) -> float:
    current_position = np.asarray(current_state[0:3], dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    return float(np.linalg.norm(current_position - target_position))


def compute_speed(current_state: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(current_state[10:13], dtype=float)))


def compute_angular_speed(current_state: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(current_state[13:16], dtype=float)))


def _normalize_quaternion_xyzw(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < eps:
        return np.full(4, np.nan, dtype=float)
    return q / norm


def compute_attitude_error_rad(current_state: np.ndarray, target_quaternion_xyzw: np.ndarray) -> float:
    """Quaternion geodesic attitude error in radians, in [0, pi]."""
    q_current = _normalize_quaternion_xyzw(current_state[3:7])
    q_target = _normalize_quaternion_xyzw(target_quaternion_xyzw)
    if not np.all(np.isfinite(q_current)) or not np.all(np.isfinite(q_target)):
        return float("nan")

    # q and -q represent the same attitude, so use |dot|.
    dot = float(np.clip(abs(np.dot(q_current, q_target)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def state_is_stable(
    current_state: np.ndarray,
    target_position: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    criteria: RecoveryCriteria,
) -> bool:
    metrics = (
        compute_position_error(current_state, target_position),
        compute_speed(current_state),
        compute_attitude_error_rad(current_state, target_quaternion_xyzw),
        compute_angular_speed(current_state),
    )
    if not np.all(np.isfinite(metrics)):
        return False

    position_error, speed, attitude_error, angular_speed = metrics
    return bool(
        position_error <= criteria.position_tolerance_m
        and speed <= criteria.speed_tolerance_m_s
        and attitude_error <= criteria.attitude_tolerance_rad
        and angular_speed <= criteria.angular_speed_tolerance_rad_s
    )


def _control_frequency(environment: Any) -> float:
    if hasattr(environment, "CTRL_FREQ"):
        return float(environment.CTRL_FREQ)
    if hasattr(environment, "ctrl_freq"):
        return float(environment.ctrl_freq)
    raise AttributeError("Environment must expose CTRL_FREQ for recovery timing metrics")


def _simulation_time_seconds(environment: Any, fallback_steps: int) -> float:
    if hasattr(environment, "step_counter") and hasattr(environment, "PYB_FREQ"):
        return float(environment.step_counter) / float(environment.PYB_FREQ)
    return float(fallback_steps) / _control_frequency(environment)


def _required_samples(duration_s: float, control_frequency_hz: float) -> int:
    return max(1, int(ceil(float(duration_s) * float(control_frequency_hz))))


def _settling_time_from_stable_history(
    stable_history: Sequence[bool],
    control_frequency_hz: float,
    dwell_time_s: float,
) -> float:
    """Return the first time after which the state remains stable until rollout end.

    Recovery is accepted only if the final continuous stable run is at least the
    requested dwell time. The returned settling time is the start of that final
    continuous stable run.
    """
    stable = np.asarray(stable_history, dtype=bool)
    required = _required_samples(dwell_time_s, control_frequency_hz)
    if stable.size < required or not np.all(stable[-required:]):
        return float("nan")

    false_indices = np.flatnonzero(~stable)
    first_stable_index = int(false_indices[-1] + 1) if false_indices.size else 0

    if stable.size - first_stable_index < required:
        return float("nan")

    return float(first_stable_index / control_frequency_hz)


def _terminal_position_statistics(
    position_errors: Sequence[float],
    control_frequency_hz: float,
    terminal_window_s: float,
) -> Tuple[float, float, float]:
    errors = np.asarray(position_errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return float("nan"), float("nan"), float("nan")

    samples = _required_samples(terminal_window_s, control_frequency_hz)
    terminal = errors[-min(samples, errors.size):]
    return (
        float(np.sqrt(np.mean(np.square(terminal)))),
        float(np.mean(terminal)),
        float(np.max(terminal)),
    )


def _reasons_from_info(info: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    raw = info.get(key, ())
    if raw in (None, ""):
        return tuple()
    if isinstance(raw, str):
        return (raw,)
    try:
        return tuple(str(value) for value in raw)
    except TypeError:
        return (str(raw),)


def _classify_outcome(
    recovered: bool,
    hard_failure: bool,
    training_envelope_violated: bool,
    truncated: bool,
    max_steps_reached: bool,
) -> str:
    if hard_failure:
        return "hard_failure"
    if max_steps_reached and not truncated:
        return "evaluation_guard_reached"
    if not recovered:
        return "not_recovered_by_horizon"
    if training_envelope_violated:
        return "recovered_with_training_envelope_violation"
    return "recovered_cleanly"


# -----------------------------------------------------------------------------
# Rollout execution
# -----------------------------------------------------------------------------


def execute_inference_loop(
    environment: Any,
    policy: Any,
    initial_observation: np.ndarray,
    max_steps: int,
    recovery_criteria: Optional[RecoveryCriteria] = None,
    initial_info: Optional[Mapping[str, Any]] = None,
) -> InferenceResult:
    """Run one deterministic policy rollout and evaluate actual recovery.

    Training-envelope violations are accumulated as diagnostics. They never
    directly determine ``recovered``. A rollout ends only through the environment
    hard-failure signal, the environment time horizon, or ``max_steps``.
    """
    criteria = recovery_criteria or RecoveryCriteria()
    criteria.validate()

    control_frequency = _control_frequency(environment)
    current_observation = initial_observation
    current_state = np.asarray(environment._getDroneStateVector(0), dtype=float)

    position_errors: List[float] = [
        compute_position_error(current_state, environment.TARGET_POS)
    ]
    stable_history: List[bool] = [
        state_is_stable(
            current_state,
            environment.TARGET_POS,
            environment.TARGET_QUATERNION,
            criteria,
        )
    ]

    max_speed = compute_speed(current_state)
    max_angular_speed = compute_angular_speed(current_state)
    max_abs_roll = float(abs(current_state[7]))
    max_abs_pitch = float(abs(current_state[8]))
    max_attitude_error = compute_attitude_error_rad(
        current_state, environment.TARGET_QUATERNION
    )

    envelope_violations = set(
        _reasons_from_info(initial_info or {}, "training_envelope_violations")
    )
    hard_failure_reasons = set(
        _reasons_from_info(initial_info or {}, "hard_failure_reasons")
    )

    terminated = False
    truncated = False
    steps = 0
    last_info: Mapping[str, Any] = initial_info or {}

    for _ in range(max_steps):
        action, _ = policy.predict(current_observation, deterministic=True)
        current_observation, _, terminated, truncated, info = environment.step(action)
        last_info = info or {}
        steps += 1

        current_state = np.asarray(environment._getDroneStateVector(0), dtype=float)
        position_errors.append(compute_position_error(current_state, environment.TARGET_POS))
        stable_history.append(
            state_is_stable(
                current_state,
                environment.TARGET_POS,
                environment.TARGET_QUATERNION,
                criteria,
            )
        )

        speed = compute_speed(current_state)
        angular_speed = compute_angular_speed(current_state)
        attitude_error = compute_attitude_error_rad(
            current_state, environment.TARGET_QUATERNION
        )

        max_speed = max(max_speed, speed)
        max_angular_speed = max(max_angular_speed, angular_speed)
        max_abs_roll = max(max_abs_roll, float(abs(current_state[7])))
        max_abs_pitch = max(max_abs_pitch, float(abs(current_state[8])))
        if np.isfinite(attitude_error):
            max_attitude_error = max(max_attitude_error, attitude_error)

        envelope_violations.update(
            _reasons_from_info(last_info, "training_envelope_violations")
        )
        hard_failure_reasons.update(
            _reasons_from_info(last_info, "hard_failure_reasons")
        )

        if terminated or truncated:
            break

    max_steps_reached = bool(not terminated and not truncated and steps >= max_steps)
    hard_failure = bool(terminated or hard_failure_reasons)

    settling_time_s = _settling_time_from_stable_history(
        stable_history=stable_history,
        control_frequency_hz=control_frequency,
        dwell_time_s=criteria.dwell_time_s,
    )
    recovered = bool(
        not hard_failure
        and not max_steps_reached
        and np.isfinite(settling_time_s)
    )

    terminal_rmse, terminal_mean, terminal_max = _terminal_position_statistics(
        position_errors=position_errors,
        control_frequency_hz=control_frequency,
        terminal_window_s=criteria.terminal_window_s,
    )

    final_state = current_state
    final_position_error = compute_position_error(final_state, environment.TARGET_POS)
    final_speed = compute_speed(final_state)
    final_angular_speed = compute_angular_speed(final_state)
    final_attitude_error = compute_attitude_error_rad(
        final_state, environment.TARGET_QUATERNION
    )

    outcome = _classify_outcome(
        recovered=recovered,
        hard_failure=hard_failure,
        training_envelope_violated=bool(envelope_violations),
        truncated=bool(truncated),
        max_steps_reached=max_steps_reached,
    )

    return InferenceResult(
        position_errors=np.asarray(position_errors, dtype=float),
        stable_history=np.asarray(stable_history, dtype=bool),
        recovered=recovered,
        outcome=outcome,
        hard_failure=hard_failure,
        hard_failure_reasons=tuple(sorted(hard_failure_reasons)),
        training_envelope_violations=tuple(sorted(envelope_violations)),
        terminated=bool(terminated),
        truncated=bool(truncated),
        max_steps_reached=max_steps_reached,
        steps=steps,
        simulation_time_s=_simulation_time_seconds(environment, steps),
        settling_time_s=settling_time_s,
        terminal_rmse_position=terminal_rmse,
        terminal_mean_position_error=terminal_mean,
        terminal_max_position_error=terminal_max,
        final_position_error=final_position_error,
        final_speed=final_speed,
        final_angular_speed=final_angular_speed,
        final_attitude_error_rad=final_attitude_error,
        max_speed=max_speed,
        max_angular_speed=max_angular_speed,
        max_abs_roll_rad=max_abs_roll,
        max_abs_pitch_rad=max_abs_pitch,
        max_attitude_error_rad=max_attitude_error,
    )


def evaluate_single_trial(
    trial_index: int,
    environment: Any,
    policy: Any,
    max_steps: int,
    base_seed: int,
    recovery_criteria: Optional[RecoveryCriteria] = None,
) -> RolloutResult:
    seed = int(base_seed + trial_index)
    initial_observation, info = environment.reset(seed=seed)

    dr_parameters = dict(info.get("dr_params", {}) or {})
    initial_conditions = dict(info.get("initial_conditions", {}) or {})

    inference = execute_inference_loop(
        environment=environment,
        policy=policy,
        initial_observation=initial_observation,
        max_steps=max_steps,
        recovery_criteria=recovery_criteria,
        initial_info=info,
    )

    return RolloutResult(
        trial_index=trial_index,
        seed=seed,
        recovered=inference.recovered,
        outcome=inference.outcome,
        hard_failure=inference.hard_failure,
        hard_failure_reasons=inference.hard_failure_reasons,
        training_envelope_violations=inference.training_envelope_violations,
        terminated=inference.terminated,
        truncated=inference.truncated,
        max_steps_reached=inference.max_steps_reached,
        steps=inference.steps,
        simulation_time_s=inference.simulation_time_s,
        position_errors=inference.position_errors,
        stable_history=inference.stable_history,
        initial_conditions=initial_conditions,
        dr_parameters=dr_parameters,
        settling_time_s=inference.settling_time_s,
        terminal_rmse_position=inference.terminal_rmse_position,
        terminal_mean_position_error=inference.terminal_mean_position_error,
        terminal_max_position_error=inference.terminal_max_position_error,
        final_position_error=inference.final_position_error,
        final_speed=inference.final_speed,
        final_angular_speed=inference.final_angular_speed,
        final_attitude_error_rad=inference.final_attitude_error_rad,
        max_speed=inference.max_speed,
        max_angular_speed=inference.max_angular_speed,
        max_abs_roll_rad=inference.max_abs_roll_rad,
        max_abs_pitch_rad=inference.max_abs_pitch_rad,
        max_attitude_error_rad=inference.max_attitude_error_rad,
    )


def evaluate_all_trials(
    trial_indices: Sequence[int],
    environment: Any,
    policy: Any,
    max_steps: int,
    base_seed: int = 12345,
    recovery_criteria: Optional[RecoveryCriteria] = None,
) -> List[RolloutResult]:
    return [
        evaluate_single_trial(
            trial_index=index,
            environment=environment,
            policy=policy,
            max_steps=max_steps,
            base_seed=base_seed,
            recovery_criteria=recovery_criteria,
        )
        for index in trial_indices
    ]


# -----------------------------------------------------------------------------
# Statistical summary
# -----------------------------------------------------------------------------


def wilson_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    p_hat = successes / total
    denominator = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denominator
    half_width = (
        z
        * sqrt(
            (p_hat * (1.0 - p_hat) / total)
            + (z * z) / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _safe_stat(values: Iterable[float], statistic: str) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")

    if statistic == "mean":
        return float(np.mean(array))
    if statistic == "median":
        return float(np.median(array))
    if statistic == "p95":
        return float(np.quantile(array, 0.95))
    raise ValueError(f"Unsupported statistic: {statistic}")


def summarize_monte_carlo_results(
    results: Sequence[RolloutResult],
    confidence_level: float = 0.95,
) -> MonteCarloSummary:
    total = len(results)
    recovered = [result for result in results if result.recovered]
    clean = [result for result in results if result.outcome == "recovered_cleanly"]
    recovered_with_violation = [
        result
        for result in results
        if result.outcome == "recovered_with_training_envelope_violation"
    ]
    not_recovered = [
        result for result in results if result.outcome == "not_recovered_by_horizon"
    ]
    hard_failures = [result for result in results if result.outcome == "hard_failure"]
    incomplete = [
        result for result in results if result.outcome == "evaluation_guard_reached"
    ]
    envelope_violations = [
        result for result in results if result.training_envelope_violated
    ]

    recovered_count = len(recovered)
    recovery_rate = recovered_count / total if total else float("nan")
    ci_lower, ci_upper = wilson_interval(recovered_count, total, confidence_level)

    return MonteCarloSummary(
        total_trials=total,
        recovered_trials=recovered_count,
        recovered_cleanly_trials=len(clean),
        recovered_with_envelope_violation_trials=len(recovered_with_violation),
        not_recovered_by_horizon_trials=len(not_recovered),
        hard_failure_trials=len(hard_failures),
        incomplete_trials=len(incomplete),
        recovery_rate=recovery_rate,
        recovery_ci_level=confidence_level,
        recovery_ci_lower=ci_lower,
        recovery_ci_upper=ci_upper,
        training_envelope_violation_trials=len(envelope_violations),
        training_envelope_violation_rate=(
            len(envelope_violations) / total if total else float("nan")
        ),
        recovered_median_settling_time_s=_safe_stat(
            (result.settling_time_s for result in recovered), "median"
        ),
        recovered_p95_settling_time_s=_safe_stat(
            (result.settling_time_s for result in recovered), "p95"
        ),
        recovered_mean_terminal_rmse=_safe_stat(
            (result.terminal_rmse_position for result in recovered), "mean"
        ),
        recovered_median_terminal_rmse=_safe_stat(
            (result.terminal_rmse_position for result in recovered), "median"
        ),
        recovered_p95_terminal_rmse=_safe_stat(
            (result.terminal_rmse_position for result in recovered), "p95"
        ),
        recovered_mean_whole_episode_rmse=_safe_stat(
            (result.whole_episode_rmse_position for result in recovered), "mean"
        ),
        recovered_median_whole_episode_rmse=_safe_stat(
            (result.whole_episode_rmse_position for result in recovered), "median"
        ),
        recovered_p95_whole_episode_rmse=_safe_stat(
            (result.whole_episode_rmse_position for result in recovered), "p95"
        ),
        recovered_p95_maximum_position_error=_safe_stat(
            (result.maximum_position_error for result in recovered), "p95"
        ),
    )


def run_monte_carlo_study(
    total_trials: int,
    environment: Any,
    policy: Any,
    max_steps: int,
    base_seed: int = 12345,
    confidence_level: float = 0.95,
    recovery_criteria: Optional[RecoveryCriteria] = None,
) -> Tuple[List[RolloutResult], MonteCarloSummary]:
    results = evaluate_all_trials(
        trial_indices=list(range(total_trials)),
        environment=environment,
        policy=policy,
        max_steps=max_steps,
        base_seed=base_seed,
        recovery_criteria=recovery_criteria,
    )
    return results, summarize_monte_carlo_results(results, confidence_level)


# -----------------------------------------------------------------------------
# Data export
# -----------------------------------------------------------------------------


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}

    for key, value in mapping.items():
        column = f"{prefix}{key}"

        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, prefix=f"{column}_"))
            continue

        array = np.asarray(value)
        if array.ndim == 0:
            scalar = array.item()
            flattened[column] = scalar
        else:
            for index, scalar in enumerate(array.reshape(-1)):
                flattened[f"{column}_{index}"] = (
                    scalar.item() if hasattr(scalar, "item") else scalar
                )

    return flattened


def _violation_flag(result: RolloutResult, name: str) -> bool:
    return name in result.training_envelope_violations


def map_rollout_result_to_dictionary(result: RolloutResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trial_index": result.trial_index,
        "seed": result.seed,
        "outcome": result.outcome,
        "recovered": result.recovered,
        "recovered_cleanly": result.recovered_cleanly,
        "recovered_with_training_envelope_violation": (
            result.recovered_with_training_envelope_violation
        ),
        "hard_failure": result.hard_failure,
        "hard_failure_reasons": ";".join(result.hard_failure_reasons),
        "training_envelope_violated": result.training_envelope_violated,
        "training_envelope_violations": ";".join(
            result.training_envelope_violations
        ),
        "violated_altitude_training_limit": _violation_flag(
            result, "altitude_below_training_limit"
        ),
        "violated_position_training_limit": _violation_flag(
            result, "position_error_training_limit"
        ),
        "violated_roll_training_limit": _violation_flag(
            result, "roll_training_limit"
        ),
        "violated_pitch_training_limit": _violation_flag(
            result, "pitch_training_limit"
        ),
        "violated_linear_speed_training_limit": _violation_flag(
            result, "linear_speed_training_limit"
        ),
        "violated_angular_speed_training_limit": _violation_flag(
            result, "angular_speed_training_limit"
        ),
        "terminated": result.terminated,
        "truncated": result.truncated,
        "max_steps_reached": result.max_steps_reached,
        "steps": result.steps,
        "simulation_time_s": result.simulation_time_s,
        "initial_error": result.initial_error,
        "whole_episode_rmse_position": result.whole_episode_rmse_position,
        "whole_episode_mean_position_error": result.whole_episode_mean_position_error,
        "maximum_position_error": result.maximum_position_error,
        "settling_time_s": result.settling_time_s,
        "terminal_rmse_position": result.terminal_rmse_position,
        "terminal_mean_position_error": result.terminal_mean_position_error,
        "terminal_max_position_error": result.terminal_max_position_error,
        "final_position_error": result.final_position_error,
        "final_speed": result.final_speed,
        "final_angular_speed": result.final_angular_speed,
        "final_attitude_error_rad": result.final_attitude_error_rad,
        "final_attitude_error_deg": np.rad2deg(result.final_attitude_error_rad),
        "max_speed": result.max_speed,
        "max_angular_speed": result.max_angular_speed,
        "max_abs_roll_rad": result.max_abs_roll_rad,
        "max_abs_pitch_rad": result.max_abs_pitch_rad,
        "max_abs_roll_deg": np.rad2deg(result.max_abs_roll_rad),
        "max_abs_pitch_deg": np.rad2deg(result.max_abs_pitch_rad),
        "max_attitude_error_rad": result.max_attitude_error_rad,
        "max_attitude_error_deg": np.rad2deg(result.max_attitude_error_rad),
        # Compatibility columns for the current plot code.
        "survived": result.recovered,
        "rmse_position": result.whole_episode_rmse_position,
        "mean_position_error": result.whole_episode_mean_position_error,
    }

    row.update(_flatten_mapping(result.initial_conditions, prefix="ic_"))
    row.update(_flatten_mapping(result.dr_parameters, prefix="dr_"))
    return row


def convert_results_to_dataframe(results: Sequence[RolloutResult]) -> pd.DataFrame:
    return pd.DataFrame(map_rollout_result_to_dictionary(result) for result in results)


def export_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    dataframe.to_csv(file_path, index=False)


def summary_to_dictionary(summary: MonteCarloSummary) -> Dict[str, Any]:
    return asdict(summary)
