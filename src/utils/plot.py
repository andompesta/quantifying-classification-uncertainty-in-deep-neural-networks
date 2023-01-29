from collections import namedtuple
from typing import List, Optional
from itertools import cycle
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt

from src.utils import ensure_dir


Timeseries = namedtuple("Timeseries", ["y", "x", "name"])

def create_timeseries(
        series: List[float],
        name: str,
        steps: Optional[List[int]] = None
) -> Timeseries:
    if steps is None:
        steps = list(range(len(series)))
    else:
        assert len(series) == len(steps), f"series: {len(series)} \t vs. \t steps: {len(steps)}"

    return Timeseries(y=series, x=steps, name=name)

def plot_scalars(
        path_: str,
        timeseries: List[Timeseries]
) -> None:
    fig = plt.figure(figsize=(8, 6))
    # fig, ax = plt.subplot(1, 1, 1, figsize=(8, 6))

    for ts, color in zip(timeseries, cycle(TABLEAU_COLORS)):
        if len(ts.x) > 10:
            markevery = int(len(ts.x) / 10)
        else:
            markevery = 1

        plt.plot(
            ts.x, ts.y,
            marker='o',
            label=ts.name,
            markevery=markevery,
            color=color
        )

    plt.grid(True)
    fig.tight_layout()
    plt.legend()
    plt.savefig(ensure_dir(path_))
    plt.close()