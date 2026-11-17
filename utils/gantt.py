from __future__ import annotations

import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union


@dataclass(frozen=True)
class GanttTask:
    """
    One scheduled interval on a specific core.
    """
    task_id: str
    start: float
    finish: float
    core: int


def build_gantt_tasks(
    start_times: Dict[str, float],
    finish_times: Dict[str, float],
    core_of: Optional[Dict[str, int]] = None,
    num_cores: Optional[int] = None,
    tie_break: str = "task_id",
) -> List[GanttTask]:
    """Build plotting intervals from start and finish times.
    
    Use core_of when supplied; otherwise reconstruct a feasible assignment
    using num_cores or the minimum number implied by overlapping intervals."""
    if num_cores is not None and num_cores < 1:
        raise ValueError("num_cores must be positive")
    intervals: List[Tuple[str, float, float]] = []
    for tid, st in start_times.items():
        if tid not in finish_times:
            raise KeyError(f"Missing finish time for task {tid}")
        ft = finish_times[tid]
        if ft < st:
            raise ValueError(f"Invalid interval for {tid}: finish < start ({ft} < {st})")
        intervals.append((tid, st, ft))

    if tie_break == "finish":
        intervals.sort(key=lambda x: (x[1], x[2], x[0]))
    elif tie_break == "task_id":
        intervals.sort(key=lambda x: (x[1], x[0], x[2]))
    else:
        raise ValueError("tie_break must be 'task_id' or 'finish'")

    if core_of is not None:
        tasks: List[GanttTask] = []
        for tid, st, ft in intervals:
            if tid not in core_of:
                raise KeyError(f"Missing core_id for task {tid} in core_of")
            c = core_of[tid]
            if c < 0:
                raise ValueError(f"Invalid core id for {tid}: {c}")
            tasks.append(GanttTask(task_id=tid, start=st, finish=ft, core=c))
        return tasks

    inferred_cores = num_cores if num_cores is not None else _infer_min_cores(intervals)
    core_end: List[float] = [0.0 for _ in range(inferred_cores)]
    tasks: List[GanttTask] = []

    for tid, st, ft in intervals:
        chosen: Optional[int] = None
        for c in range(inferred_cores):
            if core_end[c] <= st:
                chosen = c
                break
        if chosen is None:
            raise ValueError("Intervals overlap on more cores than requested")
        core_end[chosen] = max(core_end[chosen], ft)
        tasks.append(GanttTask(task_id=tid, start=st, finish=ft, core=chosen))

    return tasks


def plot_gantt(
    tasks: List[GanttTask],
    title: str = "Gantt Chart",
    xlabel: str = "Time",
    show_task_labels: bool = True,
    core_label_prefix: str = "Core",
    figsize: Tuple[float, float] = (10, 4),
    output_path: Optional[str] = None,
) -> None:
    """
    Plot a Gantt chart from a list of GanttTask.

    If output_path is provided, saves the figure (png/pdf based on extension).
    """
    if not tasks:
        raise ValueError("No tasks to plot.")

    max_core = max(t.core for t in tasks)

    fig, ax = plt.subplots(figsize=figsize)

    for t in tasks:
        ax.barh(
            y=t.core,
            width=(t.finish - t.start),
            left=t.start,
            height=0.6,
            align="center",
            edgecolor="black",
        )
        if show_task_labels:
            ax.text(
                x=t.start + (t.finish - t.start) / 2.0,
                y=t.core,
                s=t.task_id,
                va="center",
                ha="center",
                fontsize=9,
            )

    ax.set_yticks(list(range(max_core + 1)))
    ax.set_yticklabels([f"{core_label_prefix} {i}" for i in range(max_core + 1)])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if output_path:
        plt.close(fig)
    else:
        plt.show()


def _infer_min_cores(intervals: List[Tuple[str, float, float]]) -> int:
    """
    Infer the minimum number of cores needed from overlaps (sweep-line).
    """
    events: List[Tuple[float, int]] = []
    for _, st, ft in intervals:
        if st == ft:
            continue
        events.append((st, +1))
        events.append((ft, -1))

    # End before start at same time so back-to-back doesn't count as overlap
    events.sort(key=lambda x: (x[0], x[1]))

    cur = 0
    peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return max(1, peak)
