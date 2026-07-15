"""MGMT/IDH heads and uncertainty-weighted multitask loss."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TaskHeadConfig:
    """Per-task head and optimizer hyperparameters."""

    task_name: str
    input_dim: int = 128
    hidden_dim: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    freeze_backbone_fraction: float = 0.0


class BinaryTaskHead(nn.Module):
    """Binary classification head returning logits."""

    def __init__(self, config: TaskHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a flat vector of binary logits."""

        return self.net(features).squeeze(-1)


class MultiTaskHeads(nn.Module):
    """MGMT and IDH heads with Kendall-style uncertainty loss weights."""

    def __init__(self, mgmt_config: TaskHeadConfig, idh_config: TaskHeadConfig) -> None:
        super().__init__()
        self.mgmt = BinaryTaskHead(mgmt_config)
        self.idh = BinaryTaskHead(idh_config)
        self.log_sigma_mgmt = nn.Parameter(torch.zeros(()))
        self.log_sigma_idh = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return MGMT and IDH logits."""

        return {"MGMT": self.mgmt(features), "IDH": self.idh(features)}

    def uncertainty_weighted_loss(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        pos_weights: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute class-weighted BCE losses with learned uncertainty weights."""

        mgmt_loss = F.binary_cross_entropy_with_logits(logits["MGMT"], targets["MGMT"], pos_weight=pos_weights["MGMT"])
        idh_loss = F.binary_cross_entropy_with_logits(logits["IDH"], targets["IDH"], pos_weight=pos_weights["IDH"])
        total = torch.exp(-self.log_sigma_mgmt) * mgmt_loss + self.log_sigma_mgmt
        total = total + torch.exp(-self.log_sigma_idh) * idh_loss + self.log_sigma_idh
        return total, {"mgmt_loss": mgmt_loss.detach(), "idh_loss": idh_loss.detach()}


def build_task_head(config: TaskHeadConfig) -> BinaryTaskHead:
    """Build one binary task head."""

    return BinaryTaskHead(config)


def build_multitask_heads(mgmt_config: TaskHeadConfig, idh_config: TaskHeadConfig) -> MultiTaskHeads:
    """Build the paired MGMT/IDH Stage 0 heads."""

    return MultiTaskHeads(mgmt_config, idh_config)
