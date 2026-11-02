"""Typer CLI: `python -m lpdp --help`."""

from __future__ import annotations

from pathlib import Path

import typer

from lpdp.config import REPO_ROOT, load_config, resolve_path

app = typer.Typer(help="Location privacy x DPDP experiment pipeline")


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def preprocess(
    dataset: str = typer.Option("synthetic", help="synthetic | foursquare_nyc | gowalla"),
    seed: int = typer.Option(0),
) -> None:
    """Download (if needed), filter, split and write Parquet + dataset_stats.json."""
    from lpdp.data.preprocess import preprocess_dataset

    out = preprocess_dataset(dataset, seed=seed)
    typer.echo(f"Wrote processed dataset to {out}")


@app.command()
def run(
    dataset: str = typer.Option("synthetic"),
    task: str = typer.Option("next_poi"),
    mechanism: str = typer.Option("M0"),
    param_name: str = typer.Option("none"),
    param_value: float = typer.Option(0.0),
    seed: int = typer.Option(0),
) -> None:
    """Run a single mechanism x task x seed and append to results/runs.csv."""
    from lpdp.pipeline import append_run, run_one

    cfg = load_config(dataset)
    csv_path = resolve_path(cfg, "results") / "runs.csv"
    row = run_one(dataset, task, mechanism, param_name, param_value, seed)
    append_run(row, csv_path)
    typer.echo(f"{row['run_id']}  acc10={row.get('acc10')}  hotspot_f1={row.get('hotspot_f1')}")


@app.command()
def sweep(
    config: str = typer.Option("sweep"),
    dataset: str | None = typer.Option(None, help="Override datasets in sweep.yaml"),
    seed: int | None = typer.Option(None, help="Run a single seed"),
) -> None:
    """Sweep the grid in experiments/configs/sweep.yaml."""
    from experiments.scripts.run_sweep import run_sweep

    run_sweep(config=config, dataset=dataset, seed=seed)


@app.command("make-figures")
def make_figures_cmd(
    dataset: str = typer.Option("synthetic"),
) -> None:
    from experiments.scripts.make_figures import make_all

    make_all(dataset)


@app.command()
def compliance(
    dataset: str = typer.Option("synthetic"),
) -> None:
    from lpdp.compliance.evaluate import evaluate_compliance

    df = evaluate_compliance(dataset)
    typer.echo(df.head(12).to_string(index=False))


@app.command()
def demo(seed: int = typer.Option(0)) -> None:
    """Generate synthetic data, run a compact sweep, write figures and the checklist."""
    from experiments.scripts.make_figures import make_all
    from experiments.scripts.run_sweep import run_sweep
    from lpdp.compliance.evaluate import evaluate_compliance
    from lpdp.data.preprocess import preprocess_dataset

    typer.echo("Preprocessing synthetic data...")
    preprocess_dataset("synthetic", seed=seed)
    typer.echo("Running compact sweep (this trains several tiny GRUs on CPU)...")
    run_sweep(config="sweep", dataset="synthetic", seed=seed)
    evaluate_compliance("synthetic")
    make_all("synthetic")
    typer.echo("Done. See results/runs.csv and figures/.")


@app.command()
def erase(
    dataset: str = typer.Option("synthetic"),
    user_id: int = typer.Option(..., "--user-id"),
) -> None:
    from lpdp.data.consent import erase_user

    cfg = load_config(dataset)
    erase_user(resolve_path(cfg, "processed") / dataset, user_id)
    typer.echo(f"Erased user {user_id} from {dataset}")


def _patch_scripts_import() -> None:
    """Allow `from experiments.scripts...` when repo root is on sys.path."""
    import sys

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_patch_scripts_import()
