"""Run a batch of MLflow experiments from YAML configs."""

from pathlib import Path

import click

from src.config import DATA_PATH
from src.config_loader import load_config
from src.models.trainer import train
from src.utils import get_logger


logger = get_logger(__name__)


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default=str(DATA_PATH),
    show_default=True,
)
@click.option(
    "--configs-dir",
    type=click.Path(exists=True),
    default="configs/experiments",
    show_default=True,
    help="Directory with experiment YAML files.",
)
@click.option(
    "--pattern",
    default="*.yaml",
    show_default=True,
    help="Glob pattern for config files.",
)
def main(data_path, configs_dir, pattern):
    config_paths = sorted(Path(configs_dir).glob(pattern))

    if not config_paths:
        raise click.ClickException(
            f"No configs found in {configs_dir}/{pattern}"
        )

    results = []

    for config_path in config_paths:
        logger.info("=" * 60)
        logger.info("Starting experiment: %s", config_path.name)
        config = load_config(config_path)
        result = train(
            data_path=data_path,
            config=config,
            config_path=config_path,
        )
        results.append(
            {
                "config": config_path.name,
                "run_name": config.experiment.run_name or config_path.stem,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
            }
        )

    click.echo("\n=== Experiment summary ===")
    for row in sorted(results, key=lambda r: r["macro_f1"], reverse=True):
        click.echo(
            f"{row['run_name']:30s}  "
            f"acc={row['accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}"
        )

    best = max(results, key=lambda r: r["macro_f1"])
    click.echo(f"\nBest by macro_f1: {best['run_name']} ({best['macro_f1']:.4f})")


if __name__ == "__main__":
    main()
