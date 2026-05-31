import json

import click

from src.models.registry import load_predictor


@click.command()
@click.argument("text")
def main(text):
    predictor = load_predictor()
    result = predictor.predict(text)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
