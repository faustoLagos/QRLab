"""Per-episode physical center-of-mass (COM) randomization for QRLab.

The COM is moved by editing the base link's URDF *inertial origin* before
PyBullet loads the drone. This changes the effective motor thrust lever arms,
including ground-effect forces, without introducing an artificial controller
or external torque. The base-link inertia tensor about its COM is kept nominal.

Only ICRA27Env's one-drone setup is supported; other environments are unchanged.
"""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p


DEFAULT_COM_OFFSET_RANGE_M = (
    (-0.003, 0.003),  # x: +/-3 mm
    (-0.003, 0.003),  # y: +/-3 mm
    (-0.002, 0.002),  # z: +/-2 mm
)


def validate_com_offset_range_m(offset_range) -> np.ndarray:
    """Validate signed (min,max) ranges for x,y,z in body-frame metres."""
    bounds = np.asarray(offset_range, dtype=np.float64)
    if bounds.shape != (3, 2):
        raise ValueError("com_offset_range_m must contain three (min, max) pairs for x, y, z.")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("COM offset bounds must be finite.")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("COM offset bounds must satisfy min <= max for each axis.")
    return bounds


def _read_nominal_com_origin(urdf_path: Path) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    origin = root.find("./link[@name='base_link']/inertial/origin")
    if origin is None or "xyz" not in origin.attrib:
        raise ValueError(f"URDF missing base_link inertial origin: {urdf_path}")
    origin_xyz = np.fromstring(origin.attrib["xyz"], sep=" ", dtype=np.float64)
    if origin_xyz.shape != (3,) or not np.all(np.isfinite(origin_xyz)):
        raise ValueError(f"Invalid base_link inertial origin in {urdf_path}")
    return origin_xyz


def write_urdf_with_com(urdf_path: Path, output_path: Path, com_origin_xyz) -> None:
    """Write a fresh URDF with a physically displaced base-link inertial frame.

    Make relative mesh paths absolute: the randomized URDF resides in a temporary
    directory instead of the original assets directory.
    """
    urdf_path = Path(urdf_path).resolve()
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    origin = root.find("./link[@name='base_link']/inertial/origin")
    if origin is None:
        raise ValueError(f"URDF missing base_link inertial origin: {urdf_path}")
    xyz = np.asarray(com_origin_xyz, dtype=np.float64)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("COM origin must be three finite coordinates in metres.")
    origin.set("xyz", " ".join(f"{component:.12g}" for component in xyz))

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename is not None and not filename.startswith(("package://", "file://")):
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh.set("filename", str((urdf_path.parent / mesh_path).resolve()))
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


class COMRandomizationMixin:
    """Load a genuinely COM-shifted single-drone PyBullet model every reset.

    Cooperative MRO: this mixin's `_housekeeping` selects a modified URDF while
    BaseAviary constructs the drone. MassRandomizationMixin later applies mass
    randomization to that *same* freshly loaded body via `_resetModelState`.

    The constructor must not randomize during BaseAviary's initial `_housekeeping`
    call, which runs before mixin initialization finishes.
    """

    def __init__(
        self,
        *args,
        com_dr_on_reset: bool = True,
        com_offset_range_m=DEFAULT_COM_OFFSET_RANGE_M,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._com_dr_on_reset = bool(com_dr_on_reset)
        self._com_offset_bounds_m = validate_com_offset_range_m(com_offset_range_m)
        self._com_dr_nominal_urdf_path = Path(self.URDF_PATH).resolve()
        self._com_dr_nominal_origin_m = _read_nominal_com_origin(self._com_dr_nominal_urdf_path)
        self._com_dr_last: dict | None = None

    def _housekeeping(self):
        # BaseAviary calls this once during __init__, before this mixin is ready.
        if not getattr(self, "_com_dr_on_reset", False):
            return super()._housekeeping()

        if self.NUM_DRONES != 1:
            raise NotImplementedError("COMRandomizationMixin currently supports one drone.")

        bounds = self._com_offset_bounds_m
        offset_m = self.np_random.uniform(bounds[:, 0], bounds[:, 1])
        com_origin_m = self._com_dr_nominal_origin_m + offset_m

        # A unique temporary URDF avoids collisions across parallel PPO workers.
        with tempfile.TemporaryDirectory(prefix="qrlab-com-") as tmp:
            altered_urdf_path = Path(tmp) / self._com_dr_nominal_urdf_path.name
            write_urdf_with_com(
                self._com_dr_nominal_urdf_path,
                altered_urdf_path,
                com_origin_m,
            )
            original_urdf_path = self.URDF_PATH
            try:
                self.URDF_PATH = altered_urdf_path
                super()._housekeeping()
            finally:
                self.URDF_PATH = original_urdf_path

        # PyBullet's getDynamicsInfo(...)[3] is the local inertial-frame origin.
        # Saving both offsets and the loaded frame makes verification easy.
        bullet_origin_m = np.asarray(
            p.getDynamicsInfo(self.DRONE_IDS[0], -1, physicsClientId=self.CLIENT)[3],
            dtype=np.float64,
        )
        self._com_dr_last = {
            "offset_m": offset_m.tolist(),
            "nominal_origin_m": self._com_dr_nominal_origin_m.tolist(),
            "com_origin_m": com_origin_m.tolist(),
            "bullet_inertial_origin_m": bullet_origin_m.tolist(),
        }

    def _computeResetInfo(self) -> dict:
        info = dict(super()._computeResetInfo())
        if self._com_dr_on_reset and self._com_dr_last is not None:
            info["com_dr"] = dict(self._com_dr_last)
        return info
