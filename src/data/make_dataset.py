import click
import logging
import shutil
from pathlib import Path


@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """Copy raw CSV data into the processed dataset location.

    Expected input can be either a CSV file or a directory containing
    fin_data.csv. The training pipeline reads data/processed/fin_data.csv.
    """
    logger = logging.getLogger(__name__)
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    if input_path.is_dir():
        source_path = input_path / 'fin_data.csv'
    else:
        source_path = input_path

    if output_path.suffix:
        target_path = output_path
    else:
        target_path = output_path / 'fin_data.csv'

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        if target_path.exists():
            logger.info('processed dataset already exists at %s', target_path)
            return

        project_dir = Path(__file__).resolve().parents[2]
        legacy_source_path = project_dir.parent / 'data' / 'fin_data.csv'

        if legacy_source_path.exists():
            source_path = legacy_source_path
            logger.info('using legacy dataset from %s', source_path)
        else:
            raise click.ClickException(
                'Dataset not found. Put fin_data.csv into data/raw/ or pass '
                'an explicit CSV path, for example: python -m '
                'src.data.make_dataset path/to/fin_data.csv data/processed'
            )

    shutil.copy2(source_path, target_path)

    logger.info('saved processed dataset to %s', target_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
