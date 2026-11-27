"""Capability bands (floor, small, mid, frontier), read from config/capability_bands.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "capability_bands.yaml"


@dataclass(frozen=True)
class Band:
    name: str
    min_eci: float


@dataclass(frozen=True)
class Bands:
    bands: tuple[Band, ...]
    edge_warning_points: float = 2.0
    as_of: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Bands":
        bands = tuple(Band(str(b["name"]), float(b["min_eci"])) for b in d["bands"])
        if not bands:
            raise ValueError("capability bands: at least one band is required")
        if len({b.name for b in bands}) != len(bands):
            raise ValueError("capability bands: names must be unique")
        if any(a.min_eci >= b.min_eci for a, b in zip(bands, bands[1:])):
            raise ValueError("capability bands: min_eci must increase from band to band")
        return cls(bands, float(d.get("edge_warning_points", 2)), str(d.get("as_of", "")))

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.bands]

    def get(self, name: str) -> Band:
        for b in self.bands:
            if b.name == name:
                return b
        raise KeyError(f"unknown band {name!r}; bands: {', '.join(self.names)}")

    def containing(self, eci: float) -> Band:
        """The band an ECI score falls in; scores below the lowest edge count as the lowest band."""
        inside = [b for b in self.bands if b.min_eci <= eci]
        return inside[-1] if inside else self.bands[0]

    def up(self, name: str) -> Band | None:
        i = self.names.index(name)
        return self.bands[i + 1] if i + 1 < len(self.bands) else None

    def edge_warnings(self, eci: float) -> list[str]:
        out = []
        for b in self.bands:
            if abs(eci - b.min_eci) <= self.edge_warning_points:
                side = "below" if eci < b.min_eci else "at or above"
                out.append(f"min_eci {eci:g} is within {self.edge_warning_points:g} ECI points of the '{b.name}' band edge "
                           f"({b.min_eci:g}, {side} it): the price can change a lot with a small change in the threshold")
        return out


@lru_cache(maxsize=None)
def load_bands(path: str | Path = CONFIG) -> Bands:
    return Bands.from_dict(yaml.safe_load(Path(path).read_text()))
