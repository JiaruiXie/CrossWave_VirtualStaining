from pathlib import Path
import sys


ROOT = Path(
    r"C:\Users\jiaru\Documents\code_temp_files\CrossWave_VirtualStaining"
)

sys.path.insert(
    0,
    str(ROOT),
)


from src.utils.config import (
    load_config
)


CONFIG_PATH = (
    ROOT
    / "configs"
    / "maf_teacher.yaml"
)


def main():

    config = load_config(
        CONFIG_PATH
    )

    print(
        "Experiment:",
        config["experiment"]["name"],
    )

    print(
        "Batch size:",
        config["data"]["batch_size"],
    )

    print(
        "Epochs:",
        config["training"]["epochs"],
    )

    print(
        "LR:",
        config["training"]["lr"],
    )

    print(
        "Lambda cycle:",
        config["loss"]["lambda_cyc"],
    )

    print(
        "Lambda feature:",
        config["loss"]["lambda_feat"],
    )

    print(
        "Lambda saliency:",
        config["loss"]["lambda_sal"],
    )

    print(
        "\nConfig test PASSED."
    )


if __name__ == "__main__":
    main()