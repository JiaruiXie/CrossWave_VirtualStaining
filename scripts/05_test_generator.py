from pathlib import Path
import sys

import torch


ROOT = Path(
    r"C:\Users\jiaru\Documents\code_temp_files\CrossWave_VirtualStaining"
)

sys.path.insert(
    0,
    str(ROOT),
)


from src.models.generator import (
    ResNetGenerator
)


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print(
        "PyTorch version:",
        torch.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # =====================================================
    # Teacher
    # MAF -> H&E
    # =====================================================

    teacher = ResNetGenerator(
        input_channels=3,
        output_channels=3,
        num_res_blocks=9,
    )

    maf = torch.randn(
        2,
        3,
        256,
        256,
    )

    teacher_output, teacher_features = (
        teacher(
            maf,
            return_features=True,
        )
    )

    print(
        "\nTeacher generator"
    )

    print(
        "Input:",
        tuple(maf.shape),
    )

    print(
        "Feature:",
        tuple(
            teacher_features.shape
        ),
    )

    print(
        "Output:",
        tuple(
            teacher_output.shape
        ),
    )

    print(
        "Parameters:",
        f"{count_parameters(teacher):,}",
    )

    # =====================================================
    # Student
    # DAPI -> H&E
    # =====================================================

    student = ResNetGenerator(
        input_channels=1,
        output_channels=3,
        num_res_blocks=9,
    )

    dapi = torch.randn(
        2,
        1,
        256,
        256,
    )

    student_output, student_features = (
        student(
            dapi,
            return_features=True,
        )
    )

    print(
        "\nStudent generator"
    )

    print(
        "Input:",
        tuple(dapi.shape),
    )

    print(
        "Feature:",
        tuple(
            student_features.shape
        ),
    )

    print(
        "Output:",
        tuple(
            student_output.shape
        ),
    )

    print(
        "Parameters:",
        f"{count_parameters(student):,}",
    )

    # =====================================================
    # Feature compatibility
    # =====================================================

    assert (
        teacher_features.shape
        == student_features.shape
    )

    assert (
        teacher_output.shape
        == student_output.shape
    )

    print(
        "\nFeature spaces match."
    )

    print(
        "Generator test PASSED."
    )


if __name__ == "__main__":
    main()