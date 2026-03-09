"""
Classification heads for host-range prediction.

LinearHead  – single affine layer (fast baseline).
MLPHead     – two-layer MLP with LayerNorm and Dropout.
ViralClassifier – thin wrapper that bundles the chosen head with metadata.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ViralClassifier(nn.Module):
    """
    Binary classifier: vertebrate-infecting (1) vs. non-vertebrate-infecting (0).

    Args:
        embedding_dim: Dimensionality of Evo 2 embeddings (4096 for 7b).
        head_type: ``"linear"`` or ``"mlp"``.
        hidden_dim: Hidden units for MLP head.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embedding_dim: int = 4096,
        head_type: str = "mlp",
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_type = head_type

        if head_type == "linear":
            self.head = LinearHead(embedding_dim, dropout=dropout)
        elif head_type == "mlp":
            self.head = MLPHead(embedding_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown head_type '{head_type}'. Choose 'linear' or 'mlp'.")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Float tensor of shape ``(batch, embedding_dim)``.
        Returns:
            Logits of shape ``(batch,)``. Apply sigmoid for probabilities.
        """
        return self.head(embeddings)

    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities of vertebrate-infecting class."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(embeddings))
