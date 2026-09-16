import numpy as np
import pybullet as p


class DomainRandomizationMixin:
    def __init__(
            self,
            *args,
            dr_on_reset: bool = True,
            dr_mass_range: tuple = (0.95, 1.1),
            dr_kf_range: tuple = (0.5, 2.0),
            dr_km_range: tuple = (0.5, 2.0),
            dr_arm_length_range: tuple = (0.95, 1.05),
            dr_scale_inertia: bool = True,
            dr_seed: int | None = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "M") and hasattr(self, "MASS"):
            self.M = float(self.MASS)

        self._dr_nominal_mass = float(self.M)
        self._dr_nominal_kf = float(self.KF)
        self._dr_nominal_km = float(self.KM)
        self._dr_nominal_arm_length = float(getattr(self, "L", 0.0))

        self._dr_on_reset = bool(dr_on_reset)
        self._dr_mass_range = tuple(dr_mass_range)
        self._dr_kf_range = tuple(dr_kf_range)
        self._dr_km_range = tuple(dr_km_range)
        self._dr_arm_length_range = tuple(dr_arm_length_range)
        self._dr_scale_inertia = bool(dr_scale_inertia)
        if dr_seed is not None:
            self._np_random = np.random.default_rng(dr_seed)
            self._np_random_seed = dr_seed

        self._recompute_weight_and_hover()
        self._dr_last = None

    def _set_mass_in_bullet(self, mass: float, scale_inertia: float | None = None):
        for uid in getattr(self, "DRONE_IDS", []):
            p.changeDynamics(uid, -1, mass=float(mass), physicsClientId=self.CLIENT)
            if scale_inertia is not None:
                dynamics_info = p.getDynamicsInfo(uid, -1, physicsClientId=self.CLIENT)[2]
                p.changeDynamics(
                    uid, -1,
                    localInertiaDiagonal = (np.asarray(dynamics_info) * float(scale_inertia)).tolist(),
                    physicsClientId = self.CLIENT
                )

    def _refresh_arm_dependent_terms(self, old_L: float | None = None):
        if not hasattr(self, "L") or (old_L is None) or old_L == 0.0:
            old_L = float(self._dr_nominal_arm_length) if getattr(self, "_dr_nominal_arm_length", 0.0) > 0.0 else None

        if old_L is not None or old_L == 0.0:
            ratio = None
        else:
            ratio = float(self.L) / float(old_L) if float(old_L) != 0.0 else None

    def set_dr_params(
            self,
            *,
            mass: float | None = None,
            kf: float | None = None,
            km: float | None = None,
            arm_length: float | None = None,
            scale_inertia: float | None = None,
            update_log: bool = True
    ):
        if mass is not None:
            self._set_mass_in_bullet(mass, scale_inertia=scale_inertia)
            self.M = float(mass)
        if kf is not None:
            self.KF = float(kf)
        if km is not None:
            self.KM = float(km)
        if arm_length is not None and hasattr(self, "L"):
            self.L = float(arm_length)

        self._dr_after_change()
        if update_log:
            self._dr_last = {
                "mass": float(self.M),
                "kf": float(self.KF),
                "km": float(self.KM),
                "arm_length": float(self.L) if hasattr(self, "L") else self._dr_nominal_arm_length,
                "hover_rpm": float(self.HOVER_RPM),
                "action_hover_rpm": float(self.ACTION_HOVER_RPM) if hasattr(self, "ACTION_HOVER_RPM") else None
            }

    def get_last_dr_params(self):
        return dict(self._dr_last) if self._dr_last is not None else None

    def _recompute_weight_and_hover(self):
        if hasattr(self, "G"):
            self.GRAVITY = float(self.G) * float(self.M)

        self.HOVER_RPM = float(np.sqrt(float(self.GRAVITY) / float((4.0 * self.KF))))

    def _dr_sample_scales(self) -> tuple[float, float, float, float]:
        sample_mass = float(self.np_random.uniform(*self._dr_mass_range))
        sample_kf = float(self.np_random.uniform(*self._dr_kf_range))
        sample_km = float(self.np_random.uniform(*self._dr_km_range))
        sample_arm_length = float(self.np_random.uniform(*self._dr_arm_length_range))
        return sample_mass, sample_kf, sample_km, sample_arm_length

    def _dr_after_change(self):
        self._recompute_weight_and_hover()
        if hasattr(self, "RPM2FORCE"):
            self.RPM2FORCE = self.KF

        if hasattr(self, "_rpm2force"):
            self._rpm2force = self.KF

    def _apply_domain_randomization(self):
        sm, skf, skm, sal = self._dr_sample_scales()
        scaled_mass = self._dr_nominal_mass * sm
        scaled_kf = self._dr_nominal_kf * skf
        scaled_km = self._dr_nominal_km * skm
        scaled_arm_length = (self._dr_nominal_arm_length * sal) if (hasattr(self, "L") and self._dr_nominal_arm_length
                                                                    > 0.0) else None

        self.set_dr_params(
            mass=scaled_mass,
            kf=scaled_kf,
            km=scaled_km,
            arm_length=scaled_arm_length,
            scale_inertia=(sm if self._dr_scale_inertia else None),
            update_log=True
        )

    def _resetModelState(self, options: dict | None = None) -> None:
        """Applies model randomization before the reset observation is computed."""
        super()._resetModelState(options=options)
        if self._dr_on_reset:
            self._apply_domain_randomization()

    def _computeResetInfo(self) -> dict:
        """Adds the sampled domain-randomization parameters to reset metadata."""
        info = dict(super()._computeResetInfo())
        if self._dr_on_reset and self._dr_last is not None:
            info["dr_params"] = dict(self._dr_last)
        return info


def validate_scale_range(scale_range: tuple[float, float]) -> tuple[float, float]:
    """Validates and normalizes a multiplicative domain-randomization range."""
    lower_scale, upper_scale = map(float, scale_range)
    if lower_scale <= 0.0:
        raise ValueError("Domain-randomization scales must be strictly positive.")
    if lower_scale > upper_scale:
        raise ValueError("Domain-randomization scale range must be ordered as (low, high).")
    return lower_scale, upper_scale


def sample_uniform_scale(
        rng: np.random.Generator,
        scale_range: tuple[float, float]
) -> float:
    """Samples one multiplicative scale from a validated uniform range."""
    lower_scale, upper_scale = validate_scale_range(scale_range)
    return float(rng.uniform(lower_scale, upper_scale))


def calculate_hover_rpm_for_mass(
        gravity_acceleration: float,
        mass: float,
        thrust_coefficient: float
) -> float:
    """Calculates the physical hover RPM for a given mass and thrust coefficient."""
    return float(np.sqrt(
        float(gravity_acceleration) * float(mass)
        / (4.0 * float(thrust_coefficient))
    ))


class MassRandomizationMixin:
    """Stage-5 mass-only domain randomization.

    The sampled mass modifies the physical PyBullet plant while leaving inertia,
    thrust/torque coefficients, geometry, actuator mapping, and battery voltage
    unchanged. The RL action-to-RPM mapping remains immutable because it is owned
    by BaseRLAviary.ACTION_HOVER_RPM and is never recomputed here.
    """

    def __init__(
            self,
            *args,
            mass_dr_on_reset: bool = True,
            mass_scale_range: tuple[float, float] = (0.90, 1.10),
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mass_dr_on_reset = bool(mass_dr_on_reset)
        self._mass_scale_range = validate_scale_range(mass_scale_range)
        self._mass_dr_nominal_mass = float(self.M)
        self._mass_dr_last: dict[str, float] | None = None

    def _apply_mass_to_plant(self, mass: float) -> None:
        for drone_id in getattr(self, "DRONE_IDS", []):
            p.changeDynamics(
                drone_id,
                -1,
                mass=float(mass),
                physicsClientId=self.CLIENT,
            )

        self.M = float(mass)
        self.GRAVITY = float(self.G) * self.M
        self.HOVER_RPM = calculate_hover_rpm_for_mass(
            gravity_acceleration=float(self.G),
            mass=self.M,
            thrust_coefficient=float(self.KF),
        )

    def _sample_and_apply_mass_randomization(self) -> None:
        mass_scale = sample_uniform_scale(
            rng=self.np_random,
            scale_range=self._mass_scale_range,
        )
        randomized_mass = self._mass_dr_nominal_mass * mass_scale
        self._apply_mass_to_plant(randomized_mass)
        self._mass_dr_last = {
            "mass": float(randomized_mass),
            "mass_scale": float(mass_scale),
            "physical_hover_rpm": float(self.HOVER_RPM),
            "action_hover_rpm": float(self.ACTION_HOVER_RPM),
        }

    def _resetModelState(self, options: dict | None = None) -> None:
        super()._resetModelState(options=options)
        if self._mass_dr_on_reset:
            self._sample_and_apply_mass_randomization()

    def _computeResetInfo(self) -> dict:
        reset_info = dict(super()._computeResetInfo())
        if self._mass_dr_on_reset and self._mass_dr_last is not None:
            reset_info["mass_dr"] = dict(self._mass_dr_last)
        return reset_info



def validate_inertia_diagonal(inertia_diagonal: np.ndarray) -> np.ndarray:
    """Validates a positive physically admissible diagonal inertia tensor."""
    validated_inertia = np.asarray(inertia_diagonal, dtype=np.float64).reshape(3)
    if np.any(validated_inertia <= 0.0):
        raise ValueError("Principal moments of inertia must be strictly positive.")

    inertia_x, inertia_y, inertia_z = validated_inertia
    satisfies_triangle_inequalities = (
        inertia_x <= inertia_y + inertia_z
        and inertia_y <= inertia_x + inertia_z
        and inertia_z <= inertia_x + inertia_y
    )
    if not satisfies_triangle_inequalities:
        raise ValueError("Principal moments of inertia must satisfy the triangle inequalities.")

    return validated_inertia


def sample_independent_inertia_scales(
        rng: np.random.Generator,
        scale_range: tuple[float, float]
) -> np.ndarray:
    """Samples independent multiplicative scales for Jx, Jy, and Jz."""
    lower_scale, upper_scale = validate_scale_range(scale_range)
    return rng.uniform(lower_scale, upper_scale, size=3).astype(np.float64)


def calculate_randomized_inertia_diagonal(
        nominal_inertia_diagonal: np.ndarray,
        inertia_scales: np.ndarray
) -> np.ndarray:
    """Applies independent multiplicative scales to a nominal inertia diagonal."""
    nominal_inertia = validate_inertia_diagonal(nominal_inertia_diagonal)
    scales = np.asarray(inertia_scales, dtype=np.float64).reshape(3)
    if np.any(scales <= 0.0):
        raise ValueError("Inertia scales must be strictly positive.")
    return validate_inertia_diagonal(nominal_inertia * scales)


class InertiaRandomizationMixin:
    """Stage-6 independent inertia domain randomization.

    Jx, Jy, and Jz are sampled independently around the nominal URDF inertia.
    Mass randomization remains a separate concern owned by MassRandomizationMixin.
    This mixin does not modify mass, KF, KM, geometry, battery voltage, or the
    immutable RL action-to-RPM mapping.
    """

    def __init__(
            self,
            *args,
            inertia_dr_on_reset: bool = True,
            inertia_scale_range: tuple[float, float] = (0.90, 1.10),
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._inertia_dr_on_reset = bool(inertia_dr_on_reset)
        self._inertia_scale_range = validate_scale_range(inertia_scale_range)
        self._inertia_dr_nominal_diagonal = validate_inertia_diagonal(np.diag(self.J))
        self._inertia_dr_last: dict[str, list[float]] | None = None

    def _apply_inertia_to_plant(self, inertia_diagonal: np.ndarray) -> None:
        validated_inertia = validate_inertia_diagonal(inertia_diagonal)

        for drone_id in getattr(self, "DRONE_IDS", []):
            p.changeDynamics(
                drone_id,
                -1,
                localInertiaDiagonal=validated_inertia.tolist(),
                physicsClientId=self.CLIENT,
            )

        self.J = np.diag(validated_inertia)
        self.J_INV = np.diag(1.0 / validated_inertia)

    def _sample_and_apply_inertia_randomization(self) -> None:
        inertia_scales = sample_independent_inertia_scales(
            rng=self.np_random,
            scale_range=self._inertia_scale_range,
        )
        randomized_inertia = calculate_randomized_inertia_diagonal(
            nominal_inertia_diagonal=self._inertia_dr_nominal_diagonal,
            inertia_scales=inertia_scales,
        )
        self._apply_inertia_to_plant(randomized_inertia)
        self._inertia_dr_last = {
            "nominal_inertia_diagonal": self._inertia_dr_nominal_diagonal.tolist(),
            "inertia_diagonal": randomized_inertia.tolist(),
            "inertia_scales": inertia_scales.tolist(),
        }

    def _resetModelState(self, options: dict | None = None) -> None:
        super()._resetModelState(options=options)
        if self._inertia_dr_on_reset:
            self._sample_and_apply_inertia_randomization()

    def _computeResetInfo(self) -> dict:
        reset_info = dict(super()._computeResetInfo())
        if self._inertia_dr_on_reset and self._inertia_dr_last is not None:
            reset_info["inertia_dr"] = dict(self._inertia_dr_last)
        return reset_info
