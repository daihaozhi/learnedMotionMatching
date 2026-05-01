from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 (input+output)")

        act_cls = nn.ReLU if activation.lower() == "relu" else nn.ELU
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 2) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Compressor(nn.Module):
    """
    Compressor C: [Y, Q] -> Z
    Paper-like default: 5 layers, 512 hidden, ELU.
    """

    def __init__(
        self,
        yq_dim: int,
        z_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 5,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=yq_dim,
            out_dim=z_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="elu",
        )

    def forward(self, yq: torch.Tensor) -> torch.Tensor:
        return self.mlp(yq)


class Decompressor(nn.Module):
    """
    Decompressor D: [X, Z] -> Y
    Paper-like default: 3 layers, 512 hidden, ReLU.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=x_dim + z_dim,
            out_dim=y_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="relu",
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x, z], dim=-1))


class Stepper(nn.Module):
    """
    Stepper S: [X_t, Z_t] -> [dX, dZ]
    Paper-like default: 4 layers, 512 hidden, ReLU.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.mlp = MLP(
            in_dim=x_dim + z_dim,
            out_dim=x_dim + z_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="relu",
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.mlp(torch.cat([x, z], dim=-1))
        dx = d[..., : self.x_dim]
        dz = d[..., self.x_dim :]
        return dx, dz


class Projector(nn.Module):
    """
    Projector P: X_query -> [X_proj, Z_proj]
    Paper-like default: 6 layers, 512 hidden, ReLU.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.mlp = MLP(
            in_dim=x_dim,
            out_dim=x_dim + z_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="relu",
        )

    def forward(self, x_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(x_query)
        x_proj = out[..., : self.x_dim]
        z_proj = out[..., self.x_dim :]
        return x_proj, z_proj
