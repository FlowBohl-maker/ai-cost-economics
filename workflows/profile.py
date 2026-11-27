"""Workflow profiles: assumed token volumes per task, kept separate from the token price curve.

A profile is one YAML file (see workflows/profiles/example_placeholder.yaml and the README):

  name: str                         required, unique
  description: str                  required
  example: bool                     optional; true = made-up volumes, flagged in every output
  tokens_per_task:                  required; per task, summed over all model calls
    input:      N or {low, central, high}   uncached input tokens            (required)
    output:     N or {low, central, high}   visible output tokens            (required)
    reasoning:  N or {low, central, high}   reasoning tokens, billed as output (default 0)
    cache_read: N or {low, central, high}   input tokens read from a prompt cache (default 0)
    cache_write: N or {low, central, high}  input tokens written to a prompt cache (default 0)
  model:                            required; exactly one of
    band: <capability band>         floor | small | mid | frontier (edges in config/capability_bands.yaml); recommended
    min_eci: <number>               exact: cheapest model with Epoch ECI >= this, month by month
    tier: <curve tier>              a curve_v4 tier: eci>=120 ... eci>=160, mid, frontier (tier median for mid/frontier)
  effort: none | low | medium | high   optional label (default none); the token volumes should match it
  runs_per_month: N or {low, central, high}   optional; adds monthly cost
  notes, calibration: free-form     optional, ignored by the cost model
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .bands import load_bands

TOKEN_TYPES = ("input", "output", "reasoning", "cache_read", "cache_write")
MODEL_KEYS = ("band", "min_eci", "tier")
REQUIRED_TOKENS = ("input", "output")
LEVELS = ("low", "central", "high")
EFFORTS = ("none", "low", "medium", "high")
ALLOWED_KEYS = {"name", "description", "example", "tokens_per_task", "model", "effort", "runs_per_month", "notes", "calibration"}


class ProfileError(ValueError):
    pass


class EdgeWarning(UserWarning):
    """An exact min_eci close to a capability band edge."""


@dataclass(frozen=True)
class Range:
    low: float
    central: float
    high: float

    def __getitem__(self, level: str) -> float:
        return getattr(self, level)

    @classmethod
    def parse(cls, v, what: str) -> "Range":
        if isinstance(v, bool):
            raise ProfileError(f"{what}: expected a number or {{low, central, high}}, got {v!r}")
        if isinstance(v, (int, float)):
            r = cls(float(v), float(v), float(v))
        elif isinstance(v, dict):
            extra = set(v) - set(LEVELS)
            missing = [k for k in LEVELS if k not in v]
            if extra or missing:
                raise ProfileError(f"{what}: needs exactly low, central, high (missing {missing}, unknown {sorted(extra)})")
            if not all(isinstance(v[k], (int, float)) and not isinstance(v[k], bool) for k in LEVELS):
                raise ProfileError(f"{what}: low, central and high must be numbers")
            r = cls(*(float(v[k]) for k in LEVELS))
        else:
            raise ProfileError(f"{what}: expected a number or {{low, central, high}}, got {v!r}")
        if r.low < 0:
            raise ProfileError(f"{what}: values must be >= 0")
        if not r.low <= r.central <= r.high:
            raise ProfileError(f"{what}: needs low <= central <= high, got {r.low}, {r.central}, {r.high}")
        return r


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    tokens: dict[str, Range]
    band: str | None = None
    tier: str | None = None
    min_eci: float | None = None
    effort: str = "none"
    runs_per_month: Range | None = None
    example: bool = False
    source: str | None = field(default=None, compare=False)

    @property
    def price_basis(self) -> str:
        return f"band {self.band}" if self.band else f"tier {self.tier}" if self.tier else f"ECI >= {self.min_eci:g}"


def parse_profile(d: dict, source: str | None = None) -> Profile:
    where = source or d.get("name", "profile") if isinstance(d, dict) else source
    if not isinstance(d, dict):
        raise ProfileError(f"{where}: a profile must be a mapping")
    unknown = set(d) - ALLOWED_KEYS
    if unknown:
        raise ProfileError(f"{where}: unknown keys {sorted(unknown)}")
    for k in ("name", "description"):
        if not isinstance(d.get(k), str) or not d[k].strip():
            raise ProfileError(f"{where}: '{k}' is required")
    t = d.get("tokens_per_task")
    if not isinstance(t, dict):
        raise ProfileError(f"{where}: 'tokens_per_task' is required")
    if set(t) - set(TOKEN_TYPES):
        raise ProfileError(f"{where}: unknown token types {sorted(set(t) - set(TOKEN_TYPES))}; use {', '.join(TOKEN_TYPES)}")
    for k in REQUIRED_TOKENS:
        if k not in t:
            raise ProfileError(f"{where}: tokens_per_task.{k} is required")
    tokens = {k: Range.parse(t.get(k, 0), f"{where}: tokens_per_task.{k}") for k in TOKEN_TYPES}
    m = d.get("model")
    if not isinstance(m, dict) or len(set(MODEL_KEYS) & set(m)) != 1 or set(m) - set(MODEL_KEYS):
        raise ProfileError(f"{where}: 'model' needs exactly one of 'band', 'min_eci' or 'tier'")
    band, tier, min_eci = m.get("band"), m.get("tier"), m.get("min_eci")
    if tier is not None and not isinstance(tier, str):
        raise ProfileError(f"{where}: model.tier must be a tier name")
    if band is not None:
        bands = load_bands()
        if band not in bands.names:
            raise ProfileError(f"{where}: unknown band {band!r}; bands: {', '.join(bands.names)}")
    if min_eci is not None and (isinstance(min_eci, bool) or not isinstance(min_eci, (int, float))):
        raise ProfileError(f"{where}: model.min_eci must be a number")
    effort = str(d.get("effort", "none")).lower()
    if effort not in EFFORTS:
        raise ProfileError(f"{where}: effort must be one of {', '.join(EFFORTS)}")
    runs = Range.parse(d["runs_per_month"], f"{where}: runs_per_month") if d.get("runs_per_month") is not None else None
    if min_eci is not None:
        for w in load_bands().edge_warnings(float(min_eci)):
            warnings.warn(f"{where}: {w}", EdgeWarning, stacklevel=2)
    return Profile(name=d["name"].strip(), description=d["description"].strip(), tokens=tokens, band=band, tier=tier,
                   min_eci=None if min_eci is None else float(min_eci), effort=effort, runs_per_month=runs,
                   example=bool(d.get("example", False)), source=source)


def load_profile(path: str | Path) -> Profile:
    path = Path(path)
    return parse_profile(yaml.safe_load(path.read_text()), source=str(path))


def load_profiles(folder: str | Path) -> list[Profile]:
    files = sorted(Path(folder).glob("*.y*ml"))
    profiles = [load_profile(f) for f in files]
    names = [p.name for p in profiles]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ProfileError(f"duplicate profile names: {sorted(dup)}")
    return profiles
