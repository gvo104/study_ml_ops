import click
import logging
import shutil
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
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

    if not source_path.exists():
        raise FileNotFoundError(f'Expected dataset at {source_path}')

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)

    logger.info('saved processed dataset to %s', target_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
