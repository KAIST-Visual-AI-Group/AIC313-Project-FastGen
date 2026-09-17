"""Student-facing NFE-specific model definitions and checkpoint utilities."""


import torch
from torch import nn

from src.utils import count_parameters, parameter_summary


class Model(nn.Module):
    """DO NOT MODIFY: Common parent class for both NFE-specific models."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def count_parameters(self) -> int:
        """Return the total number of parameters, including frozen ones."""
        return count_parameters(self, trainable_only=False)

    @property
    def num_parameters(self) -> int:
        return self.count_parameters()

    def parameter_summary(self) -> dict[str, int]:
        return parameter_summary(self)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, **kwargs):
        return self.backbone(x, timestep, **kwargs)

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in the selected model class.")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        evaluate_mode: str,
        device="cpu",
        **kwargs,
    ) -> "Model":
        """Instantiate the selected model class and load its checkpoint."""
        model_classes = {
            "one_nfe": ModelOneNFE,
            "few_nfe": ModelFewNFE,
        }
        try:
            model_class = model_classes[evaluate_mode]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluate_mode={evaluate_mode!r}; "
                "choose 'one_nfe' or 'few_nfe'."
            ) from error

        model = model_class(device=device, **kwargs)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model.to(device).eval()





class ModelOneNFE(Model):
    """Student model evaluated with exactly one model function evaluation."""

    def __init__(self, device="cpu", **kwargs):
        raise NotImplementedError("Implement the one-NFE model constructor.")

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in ModelOneNFE.")


class ModelFewNFE(Model):
    """Student model evaluated with no more than four model evaluations."""

    def __init__(self, device="cpu", **kwargs):
        raise NotImplementedError("Implement the few-NFE model constructor.")
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in ModelFewNFE.")
