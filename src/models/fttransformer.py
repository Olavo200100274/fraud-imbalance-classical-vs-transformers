"""
FT-Transformer for Tabular Fraud Detection
============================================
Manual implementation using PyTorch nn.TransformerEncoder.
No external dependency on rtdl — self-contained.

Architecture (Gorishniy et al., NeurIPS 2021):
  1. Numerical features  → per-feature linear embedding → d_token-dim tokens
  2. Categorical features → nn.Embedding per feature    → d_token-dim tokens
  3. Prepend learnable [CLS] token
  4. Positional bias (learnable)
  5. L Transformer encoder layers (multi-head self-attention + FFN)
  6. [CLS] output → LayerNorm → Linear → 1 logit

References:
  Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
  NeurIPS 2021.  arXiv:2106.11959
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════

class TabularDataset(Dataset):
    """Wraps numpy arrays of (X_num, X_cat, y) into a torch Dataset."""

    def __init__(self, X_num, X_cat=None, y=None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = (
            torch.tensor(X_cat, dtype=torch.long) if X_cat is not None else None
        )
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        item = {"x_num": self.X_num[idx]}
        if self.X_cat is not None:
            item["x_cat"] = self.X_cat[idx]
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════

class NumericalTokenizer(nn.Module):
    """Per-feature linear embedding for numerical features."""

    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (batch, n_features) → (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for binary classification on tabular data.

    Parameters
    ----------
    d_numerical : int
        Number of numerical features.
    cat_cardinalities : list[int]
        Number of unique categories for each categorical feature.
        Empty list if no categorical features.
    d_token : int
        Embedding dimension for each feature token.
    n_blocks : int
        Number of Transformer encoder layers.
    n_heads : int
        Number of attention heads.
    d_ffn : int
        Hidden dimension of the feed-forward network.
    attention_dropout : float
    ffn_dropout : float
    residual_dropout : float
    """

    def __init__(
        self,
        d_numerical,
        cat_cardinalities,
        d_token,
        n_blocks,
        n_heads,
        d_ffn,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        residual_dropout=0.0,
    ):
        super().__init__()
        self.d_numerical = d_numerical
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token

        # ── Numerical tokenizer ─────────────────────────────────────
        self.num_tokenizer = NumericalTokenizer(d_numerical, d_token)

        # ── Categorical embeddings ──────────────────────────────────
        self.cat_embeddings = nn.ModuleList()
        for card in cat_cardinalities:
            # Reserve the last embedding for unseen category codes.
            self.cat_embeddings.append(nn.Embedding(card + 1, d_token))

        # ── [CLS] token ─────────────────────────────────────────────
        n_tokens = d_numerical + len(cat_cardinalities) + 1  # +1 for CLS
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # ── Positional bias (learnable) ─────────────────────────────
        self.pos_embedding = nn.Parameter(torch.empty(1, n_tokens, d_token))
        nn.init.normal_(self.pos_embedding, std=0.02)

        # ── Transformer encoder ─────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (more stable training)
        )
        # Apply residual dropout via a wrapper
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_blocks
        )
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.attention_dropout_rate = attention_dropout

        # ── Output head ─────────────────────────────────────────────
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

        # Apply attention dropout to encoder layers
        for layer in self.transformer.layers:
            layer.self_attn.dropout = attention_dropout

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_num, x_cat=None):
        """
        Parameters
        ----------
        x_num : (batch, d_numerical) float tensor
        x_cat : (batch, n_cat) long tensor or None

        Returns
        -------
        logits : (batch,) float tensor
        """
        batch_size = x_num.size(0)

        # Tokenize numerical features → (batch, d_numerical, d_token)
        tokens = self.num_tokenizer(x_num)

        # Tokenize categorical features → (batch, n_cat, d_token)
        if x_cat is not None and len(self.cat_embeddings) > 0:
            cat_tokens = []
            for i, emb in enumerate(self.cat_embeddings):
                idx = x_cat[:, i]
                idx = torch.where((idx < 0) | (idx >= emb.num_embeddings - 1),
                                  emb.num_embeddings - 1, idx)
                cat_tokens.append(emb(idx))
            cat_tokens = torch.stack(cat_tokens, dim=1)
            tokens = torch.cat([tokens, cat_tokens], dim=1)

        # Prepend [CLS] → (batch, n_tokens, d_token)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Add positional bias
        tokens = tokens + self.pos_embedding

        # Transformer encoder
        tokens = self.transformer(tokens)
        tokens = self.residual_dropout(tokens)

        # [CLS] output → logit
        cls_out = tokens[:, 0]  # first token is CLS
        cls_out = self.head_norm(cls_out)
        logit = self.head(cls_out).squeeze(-1)

        return logit

    @torch.no_grad()
    def forward_with_attention(self, x_num, x_cat=None):
        """
        Run evaluation inference and return per-layer attention weights.

        For each Pre-LN layer, obtain head-averaged attention weights from
        its normalised incoming tokens, then propagate those tokens through
        the original layer forward. The auxiliary attention output is not
        used to reconstruct residual or feed-forward updates: doing so can
        introduce floating-point drift relative to the saved predictions.
        Any encoder final normalisation is applied as in forward(). This
        diagnostic switches the model to evaluation mode and uses no hooks.

        Returns
        -------
        logits : (batch,) float tensor
        attn_weights : list[Tensor]
            One (batch, n_tokens, n_tokens) tensor per Transformer layer.
            Each row sums to 1.0 (softmax over keys).
        """
        if any(not layer.norm_first for layer in self.transformer.layers):
            raise ValueError("Attention extraction requires the model's Pre-LN encoder layers.")
        self.eval()
        batch_size = x_num.size(0)

        # ── Tokenize (identical to forward()) ──
        tokens = self.num_tokenizer(x_num)

        if x_cat is not None and len(self.cat_embeddings) > 0:
            cat_tokens = []
            for i, emb in enumerate(self.cat_embeddings):
                idx = x_cat[:, i]
                idx = torch.where((idx < 0) | (idx >= emb.num_embeddings - 1),
                                  emb.num_embeddings - 1, idx)
                cat_tokens.append(emb(idx))
            cat_tokens = torch.stack(cat_tokens, dim=1)
            tokens = torch.cat([tokens, cat_tokens], dim=1)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding

        # ── Extract weights without replacing the original layer forward ──
        attn_weights = []
        for layer in self.transformer.layers:
            # Pre-LN: norm before attention
            normed = layer.norm1(tokens)
            _, weights = layer.self_attn(
                normed, normed, normed,
                need_weights=True,
                average_attn_weights=True,  # average over heads → (batch, seq, seq)
            )
            tokens = layer(tokens)
            attn_weights.append(weights)  # (batch, n_tokens, n_tokens)

        if self.transformer.norm is not None:
            tokens = self.transformer.norm(tokens)
        tokens = self.residual_dropout(tokens)

        # [CLS] output → logit
        cls_out = tokens[:, 0]
        cls_out = self.head_norm(cls_out)
        logit = self.head(cls_out).squeeze(-1)

        return logit, attn_weights


# ═══════════════════════════════════════════════════════════════════════════
#  Training utilities
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    """One training pass. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_cat = batch.get("x_cat")
        if x_cat is not None:
            x_cat = x_cat.to(device)
        y = batch["y"].to(device)

        logits = model(x_num, x_cat)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader.

    Returns
    -------
    y_true : np.ndarray
    y_scores : np.ndarray  (sigmoid probabilities)
    """
    model.eval()
    all_y = []
    all_scores = []

    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_cat = batch.get("x_cat")
        if x_cat is not None:
            x_cat = x_cat.to(device)

        logits = model(x_num, x_cat)
        scores = torch.sigmoid(logits).cpu().numpy()

        all_scores.append(scores)
        if "y" in batch:
            all_y.append(batch["y"].numpy())

    y_scores = np.concatenate(all_scores)
    y_true = np.concatenate(all_y) if all_y else None
    return y_true, y_scores


def build_model(hp, d_numerical, cat_cardinalities):
    """Build an FTTransformer from an Optuna hyperparameter dict."""
    d_token = hp["d_token"]
    d_ffn = int(d_token * hp["ffn_d_hidden_factor"])
    # Ensure d_ffn is even (helps with some GPU optimisations)
    d_ffn = d_ffn + (d_ffn % 2)

    return FTTransformer(
        d_numerical=d_numerical,
        cat_cardinalities=cat_cardinalities,
        d_token=d_token,
        n_blocks=hp["n_blocks"],
        n_heads=hp["attention_n_heads"],
        d_ffn=d_ffn,
        attention_dropout=hp["attention_dropout"],
        ffn_dropout=hp["ffn_dropout"],
        residual_dropout=hp["residual_dropout"],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Optuna hyperparameter search space
# ═══════════════════════════════════════════════════════════════════════════

def suggest_hyperparams(trial):
    """Suggest FT-Transformer hyperparameters for an Optuna trial."""
    d_token = trial.suggest_categorical("d_token", [64, 128, 192, 256])
    n_heads = trial.suggest_categorical("attention_n_heads", [4, 8])

    # Ensure d_token is divisible by n_heads
    if d_token % n_heads != 0:
        # Fallback: use 4 heads (always divides 64, 128, 192, 256)
        n_heads = 4

    return {
        "d_token": d_token,
        "n_blocks": trial.suggest_int("n_blocks", 2, 4),
        "attention_n_heads": n_heads,
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.3),
        "ffn_d_hidden_factor": trial.suggest_float("ffn_d_hidden_factor", 1.33, 2.67),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.4),
        "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.2),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
    }
