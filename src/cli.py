import subprocess
import sys

import click

from src.config import DATA_PATH, DEFAULT_CONFIG_PATH
from src.config_loader import load_config
from src.models.registry import load_predictor
from src.models.trainer import train


@click.group()
def main():
    """Study MLOps CLI — data, training and inference."""


@main.command("data")
@click.argument("input_filepath", type=click.Path(), default="data/raw")
@click.argument("output_filepath", type=click.Path(), default="data/processed")
def data_cmd(input_filepath, output_filepath):
    """Prepare processed dataset from raw CSV."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data.make_dataset",
            input_filepath,
            output_filepath,
        ],
        check=True,
    )


@main.command("train")
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default=str(DATA_PATH),
    show_default=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=str(DEFAULT_CONFIG_PATH),
    show_default=True,
)
def train_cmd(data_path, config_path):
    """Train classifier and save artifacts."""
    config = load_config(config_path)
    train(data_path=data_path, config=config, config_path=config_path)


@main.command("experiments")
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
)
def experiments_cmd(data_path, configs_dir):
    """Run all experiment configs and log to MLflow."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.models.run_experiments",
            "--data-path",
            data_path,
            "--configs-dir",
            configs_dir,
        ],
        check=True,
    )


@main.command("predict")
@click.argument("text")
def predict_cmd(text):
    """Run inference for a single text."""
    import json

    result = load_predictor().predict(text)
    click.echo(json.dumps(result, indent=2))


@main.group("infra")
def infra_group():
    """Docker infrastructure helpers."""


@infra_group.command("status")
def infra_status():
    subprocess.run(["docker", "compose", "ps"], check=False)


if __name__ == "__main__":
    main()
