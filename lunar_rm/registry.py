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
    "A0": Experiment("fspl", "baseline", ()),
    "N0": Experiment("tiny", "direct", BASE_FEATURES),
    "N1": Experiment("unet", "direct", BASE_FEATURES),
    "R0": Experiment("tiny", "residual", PHYSICS_FEATURES),
    "G0": Experiment("tiny", "direct", PHYSICS_FEATURES),
    "R1": Experiment("tiny", "residual", GEOMETRY_FEATURES),
    "R2": Experiment("tiny", "residual", FULL_FEATURES),
}
for _suffix, _feature in (("no_los", "los"), ("no_fresnel", "fresnel"),
                         ("no_curvature", "curvature_proxy")):
    _profiles[f"R2_{_suffix}"] = replace(
        _profiles["R2"], features=tuple(f for f in FULL_FEATURES if f != _feature))
_profiles["R2_mae"] = _profiles["R2"]
_profiles["R2_mse"] = replace(_profiles["R2"], loss="mse")
_profiles["R2_alpha0"] = _profiles["R2"]
_profiles["R2_alpha01"] = replace(_profiles["R2"], nlos_alpha=0.1)
_profiles["R2_alpha05"] = replace(_profiles["R2"], nlos_alpha=0.5)
_profiles["R2_direct"] = replace(_profiles["R2"], head="direct")
EXPERIMENTS = MappingProxyType(_profiles)
