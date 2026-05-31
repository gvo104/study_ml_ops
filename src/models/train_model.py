import json
from pathlib import Path

import click

from src.config import DATA_PATH, DEFAULT_CONFIG_PATH
from src.config_loader import load_config
from src.models.registry import load_predictor
from src.models.trainer import train


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default=str(DATA_PATH),
    show_default=True,
    help="Path to the processed training CSV.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=str(DEFAULT_CONFIG_PATH),
    show_default=True,
    help="Path to experiment YAML config.",
)
def main(data_path, config_path):
    config = load_config(config_path)
    result = train(
        data_path=data_path,
        config=config,
        config_path=config_path,
    )
    click.echo(
        json.dumps(
            {
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "model_dir": str(result.model_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
