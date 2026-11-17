from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    """DAG structure, core allocation and workload parameters."""
    dag_id: str
    length: int
    parallelism: int
    num_cores: int
    period_ms: int  # Ti in [1000, 3000]
    utilization: float = 0.5  # Fixed task utilization

    def total_budget_ms(self) -> float:
        # Cτ = Uτ × Tτ
        return float(self.period_ms) * float(self.utilization)


@dataclass(frozen=True)
class SweepRanges:
    length_range: Tuple[int, int] = (4, 15)
    parallelism_range: Tuple[int, int] = (4, 15)
    cores_range: Tuple[int, int] = (2, 16)
    period_range_ms: Tuple[int, int] = (1000, 3000)
