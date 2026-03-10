"""
Classification heads for host-range prediction.

Supports binary (1 logit + sigmoid) and multiclass (N logits + softmax).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class MLPHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class ViralClassifier(nn.Module):
    """
    Args:
        embedding_dim: Evo 2 embedding size (4096 for 7b).
        num_classes: 2 for binary (outputs 1 logit), 3+ for multiclass (outputs N logits).
        head_type: "linear" or "mlp".
        hidden_dim: MLP hidden size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embedding_dim: int = 4096,
        num_classes: int = 2,
        head_type: str = "mlp",
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.is_binary = (num_classes == 2)

        output_dim = 1 if self.is_binary else num_classes

        if head_type == "linear":
            self.head = LinearHead(embedding_dim, output_dim=output_dim, dropout=dropout)
        elif head_type == "mlp":
            self.head = MLPHead(embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown head_type '{head_type}'.")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Binary:      (batch,) logits
            Multiclass:  (batch, num_classes) logits
        """
        return self.head(embeddings)

    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Sigmoid for binary, softmax for multiclass."""
        with torch.no_grad():
            logits = self.forward(embeddings)
            if self.is_binary:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=-1)
