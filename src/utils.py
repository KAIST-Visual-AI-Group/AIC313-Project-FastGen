"""Parameter counting utilities kept in one place for student experiments."""

from collections.abc import Iterable
from torch import nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    parameters: Iterable = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def parameter_summary(module: nn.Module) -> dict[str, int]:
    total = count_parameters(module, trainable_only=False)
    trainable = count_parameters(module, trainable_only=True)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def print_parameter_summary(module: nn.Module) -> dict[str, int]:
    summary = parameter_summary(module)
    print(f"Total parameters: {summary['total']:,}")
    print(f"Trainable parameters: {summary['trainable']:,}")
    print(f"Frozen parameters: {summary['frozen']:,}")
    return summary
