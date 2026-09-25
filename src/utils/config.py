from pathlib import Path
import yaml


def load_config(path):

    path = Path(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(
            f
        )

    return config