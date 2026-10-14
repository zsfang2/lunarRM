"""Explicit experiment profiles and ordered, label-free feature names."""

from dataclasses import dataclass, replace
from types import MappingProxyType


BASE_FEATURES = ("height", "tx", "band")
PHYSICS_FEATURES = BASE_FEATURES + ("fspl",)
GEOMETRY_FEATURES = PHYSICS_FEATURES + (
    "relative_row", "relative_col", "relative_height", "slope", "curvature_proxy",
)
FULL_FEATURES = GEOMETRY_FEATURES + ("los", "fresnel")
FEATURES = frozenset(FULL_FEATURES)


@dataclass(frozen=True)
class Experiment:
    model: str
    head: str
    features: tuple[str, ...]
    loss: str = "mae"
    nlos_alpha: float = 0.0


_profiles = {
    "fspl": Experiment("fspl", "baseline", ()),
    "tiny_unet": Experiment("tiny", "direct", BASE_FEATURES),
    "unet": Experiment("unet", "direct", BASE_FEATURES),
    "fspl_residual": Experiment("tiny", "residual", PHYSICS_FEATURES),
    "fspl_direct": Experiment("tiny", "direct", PHYSICS_FEATURES),
    "geometry_unet": Experiment("tiny", "residual", GEOMETRY_FEATURES),
    "physics_unet": Experiment("tiny", "residual", FULL_FEATURES),
}
for _suffix, _feature in (("no_los", "los"), ("no_fresnel", "fresnel"),
                         ("no_curvature", "curvature_proxy")):
    _profiles[f"physics_unet_{_suffix}"] = replace(
        _profiles["physics_unet"], features=tuple(f for f in FULL_FEATURES if f != _feature))
_profiles["physics_unet_mae"] = _profiles["physics_unet"]
_profiles["physics_unet_mse"] = replace(_profiles["physics_unet"], loss="mse")
_profiles["physics_unet_alpha0"] = _profiles["physics_unet"]
_profiles["physics_unet_alpha01"] = replace(_profiles["physics_unet"], nlos_alpha=0.1)
_profiles["physics_unet_alpha05"] = replace(_profiles["physics_unet"], nlos_alpha=0.5)
_profiles["physics_unet_direct"] = replace(_profiles["physics_unet"], head="direct")
EXPERIMENTS = MappingProxyType(_profiles)

# Compatibility for saved configurations from earlier repository versions.
LEGACY_NAMES = {'A0': 'fspl', 'N0': 'tiny_unet', 'N1': 'unet', 'R0': 'fspl_residual', 'G0': 'fspl_direct', 'R1': 'geometry_unet', 'R2': 'physics_unet', 'R2_no_los': 'physics_unet_no_los', 'R2_no_fresnel': 'physics_unet_no_fresnel', 'R2_no_curvature': 'physics_unet_no_curvature', 'R2_mae': 'physics_unet_mae', 'R2_mse': 'physics_unet_mse', 'R2_alpha0': 'physics_unet_alpha0', 'R2_alpha01': 'physics_unet_alpha01', 'R2_alpha05': 'physics_unet_alpha05', 'R2_direct': 'physics_unet_direct'}
