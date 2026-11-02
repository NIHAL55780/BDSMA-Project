"""Run the mechanism × task × seed grid and write results/runs.csv."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from lpdp.config import load_config, resolve_path  # noqa: E402
from lpdp.pipeline import append_run, run_one  # noqa: E402


def run_sweep(config: str = "sweep", dataset: str | None = None, seed: int | None = None) -> Path:
    sweep = load_config(config)
    datasets = [dataset] if dataset else list(sweep.get("datasets", ["synthetic"]))
    seeds = [seed] if seed is not None else list(sweep.get("seeds", [0]))
    grid = sweep["grid"]
    csv_path = resolve_path(load_config(datasets[0]), "results") / "runs.csv"
    n = 0
    for ds in datasets:
        for task, jobs in grid.items():
            for job in jobs:
                for s in seeds:
                    print(
                        f"-> {ds} {task} {job['mechanism']} "
                        f"{job['param_name']}={job['param_value']} seed={s}",
                        flush=True,
                    )
                    try:
                        row = run_one(
                            dataset=ds,
                            task=task,
                            mechanism=job["mechanism"],
                            param_name=job["param_name"],
                            param_value=float(job["param_value"]),
                            seed=int(s),
                        )
                        append_run(row, csv_path)
                        n += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"   FAILED: {type(exc).__name__}: {exc}", flush=True)
                    finally:
                        gc.collect()
    print(f"Wrote {n} runs to {csv_path}")
    return csv_path


if __name__ == "__main__":
    run_sweep()
