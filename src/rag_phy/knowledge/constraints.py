"""Exact and range-indexed lookup for cited material temperature constraints."""

from __future__ import annotations

import csv
import logging
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class ConstraintDataError(ValueError):
    """Raised when a curated structured-constraint file is malformed."""


class ConstraintNotFoundError(LookupError):
    """Raised when an exact material constraint is absent from the curated store."""


@dataclass(frozen=True)
class MaterialConstraint:
    """One sourced material rating; service temperature is in kelvin."""

    material_name: str
    max_service_temperature_k: float
    source_id: str


class MaterialConstraintStore:
    """O(1) exact material index plus sorted temperature range index."""

    REQUIRED_COLUMNS = frozenset(
        {"material_name", "max_service_temperature_k", "source_id"}
    )

    def __init__(self, records: Iterable[MaterialConstraint]) -> None:
        """Index validated material records and reject ambiguous duplicate names."""
        self._by_material: dict[str, MaterialConstraint] = {}
        for record in records:
            key = self._normalize_name(record.material_name)
            if not key:
                raise ConstraintDataError("Material names must not be empty")
            if not record.source_id.strip():
                raise ConstraintDataError(
                    f"Material constraint {record.material_name!r} requires a source ID"
                )
            if not math.isfinite(record.max_service_temperature_k) or record.max_service_temperature_k <= 0:
                raise ConstraintDataError(
                    f"Material {record.material_name!r} has an invalid service temperature"
                )
            if key in self._by_material:
                raise ConstraintDataError(f"Duplicate material constraint for {record.material_name}")
            self._by_material[key] = record
        self._by_temperature = sorted(
            (record.max_service_temperature_k, key)
            for key, record in self._by_material.items()
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> MaterialConstraintStore:
        """Load sourced material ratings from a CSV with the documented column schema.

        Args:
            path: Curated CSV with material_name, max_service_temperature_k, and source_id.

        Returns:
            Indexed material constraint store.

        Raises:
            FileNotFoundError: If the curated CSV is missing.
            ConstraintDataError: If schema, values, or record provenance are invalid.
        """
        csv_path = Path(path)
        records: list[MaterialConstraint] = []
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or not cls.REQUIRED_COLUMNS.issubset(
                    set(reader.fieldnames)
                ):
                    raise ConstraintDataError(
                        f"{csv_path} must contain columns: "
                        f"{', '.join(sorted(cls.REQUIRED_COLUMNS))}"
                    )
                for line_number, row in enumerate(reader, start=2):
                    try:
                        material_name = (row["material_name"] or "").strip()
                        source_id = (row["source_id"] or "").strip()
                        if not material_name or not source_id:
                            raise ValueError("Material name and source ID are required")
                        records.append(
                            MaterialConstraint(
                                material_name=material_name,
                                max_service_temperature_k=float(
                                    row["max_service_temperature_k"]
                                ),
                                source_id=source_id,
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        raise ConstraintDataError(
                            f"Invalid material constraint at {csv_path}:{line_number}"
                        ) from exc
        except ConstraintDataError:
            raise
        except OSError as exc:
            logger.exception("Material constraint file read failed", extra={"path": str(csv_path)})
            raise ConstraintDataError(f"Could not read material constraints from {csv_path}") from exc
        if not records:
            raise ConstraintDataError(f"Material constraint file contains no records: {csv_path}")
        return cls(records)

    def get_material(self, material_name: str) -> MaterialConstraint:
        """Return one exact material record or raise when no source-backed record exists."""
        key = self._normalize_name(material_name)
        try:
            return self._by_material[key]
        except KeyError as exc:
            raise ConstraintNotFoundError(
                f"No curated constraint record for material {material_name!r}"
            ) from exc

    def materials_rated_at_least(self, temperature_k: float) -> list[MaterialConstraint]:
        """Return materials with service limits >= temperature_k using binary search."""
        if not math.isfinite(temperature_k) or temperature_k <= 0:
            raise ValueError("Temperature query must be positive kelvin")
        start = bisect_left(self._by_temperature, (temperature_k, ""))
        return [
            self._by_material[material_key]
            for _, material_key in self._by_temperature[start:]
        ]

    @staticmethod
    def _normalize_name(material_name: str) -> str:
        """Normalize material keys for case-insensitive exact lookup."""
        return " ".join(material_name.casefold().split())


__all__ = [
    "ConstraintDataError",
    "ConstraintNotFoundError",
    "MaterialConstraint",
    "MaterialConstraintStore",
]
