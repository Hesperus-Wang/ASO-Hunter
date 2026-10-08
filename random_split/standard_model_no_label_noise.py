# Ablation variant: w/o Label Noise Strategy
# Based on standard_model_no_adaptive_scheduler.py.
# Keeps MAML-style episodic meta-learning and uniform task sampling,
# but hard-disables training label noise injection.

import argparse
import contextlib
import copy
import json
import math
import os
import random
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Subset

try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call  # type: ignore

try:
    import wandb
except Exception:
    wandb = None

try:
    import yaml
except Exception:
    yaml = None

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None

try:
    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo
    from rinalmo.data.alphabet import Alphabet
    try:
        from datamodule_no_context_no_label_noise import ASODataModule  # type: ignore
    except Exception:
        try:
            from datamodule_no_context import ASODataModule  # type: ignore
        except Exception:
            from rinalmo.data.downstream.aso_meta.datamodule import ASODataModule
    from rinalmo.utils.scaler import StandardScaler
except Exception:
    # Fallbacks for local development / standalone use.
    from rinalmo.config import model_config  # type: ignore
    from rinalmo.model.model import RiNALMo  # type: ignore
    from rinalmo.data.alphabet import Alphabet  # type: ignore
    try:
        from datamodule_no_context_no_label_noise import ASODataModule  # type: ignore
    except Exception:
        try:
            from datamodule_no_context import ASODataModule  # type: ignore
        except Exception:
            from datamodule import ASODataModule  # type: ignore
    try:
        from rinalmo.utils.scaler import StandardScaler  # type: ignore
    except Exception:
        class StandardScaler:  # type: ignore
            def __init__(self):
                self.mean = None
                self.std = None

            def partial_fit(self, x: torch.Tensor):
                if x.ndim == 1:
                    x = x.unsqueeze(-1)
                self.mean = x.mean(dim=0)
                self.std = x.std(dim=0).clamp_min(1e-6)

            def transform(self, x: torch.Tensor) -> torch.Tensor:
                if self.mean is None or self.std is None:
                    raise RuntimeError("StandardScaler must be fitted before use")
                return (x - self.mean.to(x.device)) / self.std.to(x.device)

            def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
                if self.mean is None or self.std is None:
                    raise RuntimeError("StandardScaler must be fitted before use")
                return x * self.std.to(x.device) + self.mean.to(x.device)


# -------------------------------
# Model definition: RiNALMo + one-hot chemistry/backbone fusion
# -------------------------------

class GlobalPooling(nn.Module):
    """Simple and stable masked mean pooling."""

    def __init__(self, embed_dim: int, projection_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(embed_dim, projection_dim) if projection_dim < embed_dim else nn.Identity()
        self.output_dim = projection_dim if projection_dim < embed_dim else embed_dim

    def forward(self, representations: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        proj_repr = self.projection(representations)
        proj_repr = proj_repr.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        denom = (~pad_mask).sum(dim=1).clamp_min(1).unsqueeze(-1)
        return proj_repr.sum(dim=1) / denom


class MultiScaleConvBlock(nn.Module):
    """Token-level local pattern extractor with residual multi-scale convolution."""

    def __init__(self, embed_dim: int, dropout: float = 0.1, kernels: Tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=k // 2, groups=1, bias=False),
                nn.GELU(),
            )
            for k in kernels
        ])
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * len(kernels), embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        valid = (~pad_mask).unsqueeze(-1).to(dtype=x.dtype)
        x = x * valid
        x_norm = self.norm(x)
        x_t = x_norm.transpose(1, 2)
        branch_outs = [branch(x_t).transpose(1, 2) for branch in self.branches]
        mixed = self.proj(torch.cat(branch_outs, dim=-1))
        out = (x + mixed) * valid
        return out


class PositionAwareAttentionPooling(nn.Module):
    """Learned position-aware attention pooling over valid ASO tokens."""

    def __init__(self, embed_dim: int, projection_dim: int = 64, max_positions: int = 256, dropout: float = 0.1):
        super().__init__()
        self.position_embedding = nn.Embedding(max_positions, embed_dim)
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.projection = nn.Linear(embed_dim, projection_dim) if projection_dim < embed_dim else nn.Identity()
        self.output_dim = projection_dim if projection_dim < embed_dim else embed_dim
        self.max_positions = int(max_positions)

    def forward(self, representations: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = representations.shape
        pos = torch.arange(seq_len, device=representations.device).unsqueeze(0).expand(bsz, seq_len)
        pos = pos.clamp(max=self.max_positions - 1)
        x = representations + self.position_embedding(pos)
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(pad_mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * (~pad_mask).to(dtype=representations.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(representations * attn.unsqueeze(-1), dim=1)
        return self.projection(pooled)


class ModAwareRegionPooling(nn.Module):
    """Pool left wing / center gap / right wing using chemistry-token heuristics."""

    def __init__(self, embed_dim: int, projection_dim: int = 32, chem_dna_idx: int = 1, min_center_run: int = 3):
        super().__init__()
        self.projection = nn.Linear(embed_dim, projection_dim) if projection_dim < embed_dim else nn.Identity()
        self.region_dim = projection_dim if projection_dim < embed_dim else embed_dim
        self.output_dim = self.region_dim * 3
        self.chem_dna_idx = int(chem_dna_idx)
        self.min_center_run = int(min_center_run)

    @staticmethod
    def _longest_true_run(mask_1d: torch.Tensor) -> Tuple[int, int]:
        best_s, best_e, best_len = -1, -1, 0
        cur_s = -1
        for i in range(int(mask_1d.numel())):
            if bool(mask_1d[i]):
                if cur_s < 0:
                    cur_s = i
            else:
                if cur_s >= 0:
                    cur_len = i - cur_s
                    if cur_len > best_len:
                        best_s, best_e, best_len = cur_s, i, cur_len
                    cur_s = -1
        if cur_s >= 0:
            cur_len = int(mask_1d.numel()) - cur_s
            if cur_len > best_len:
                best_s, best_e, best_len = cur_s, int(mask_1d.numel()), cur_len
        return best_s, best_e

    def forward(self, representations: torch.Tensor, pad_mask: torch.Tensor, chem_tokens: torch.Tensor) -> torch.Tensor:
        proj = self.projection(representations)
        valid = ~pad_mask
        region_masks = [torch.zeros_like(valid) for _ in range(3)]
        for b in range(valid.size(0)):
            idx = torch.nonzero(valid[b], as_tuple=False).flatten()
            n = int(idx.numel())
            if n == 0:
                continue
            chem_valid = chem_tokens[b, idx]
            is_dna = chem_valid.eq(self.chem_dna_idx)
            start, end = self._longest_true_run(is_dna)
            use_fallback = (start < 0) or (end - start < self.min_center_run) or (start == 0) or (end == n)
            if use_fallback:
                for ridx in range(3):
                    s = (ridx * n) // 3
                    e = ((ridx + 1) * n) // 3
                    if e <= s:
                        e = min(n, s + 1)
                    chosen = idx[s:e]
                    region_masks[ridx][b, chosen] = True
            else:
                left_idx = idx[:start]
                center_idx = idx[start:end]
                right_idx = idx[end:]
                if left_idx.numel() == 0 or right_idx.numel() == 0:
                    for ridx in range(3):
                        s = (ridx * n) // 3
                        e = ((ridx + 1) * n) // 3
                        if e <= s:
                            e = min(n, s + 1)
                        chosen = idx[s:e]
                        region_masks[ridx][b, chosen] = True
                else:
                    region_masks[0][b, left_idx] = True
                    region_masks[1][b, center_idx] = True
                    region_masks[2][b, right_idx] = True

        outputs: List[torch.Tensor] = []
        for region_mask in region_masks:
            masked = proj.masked_fill(~region_mask.unsqueeze(-1), 0.0)
            denom = region_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            outputs.append(masked.sum(dim=1) / denom)
        return torch.cat(outputs, dim=-1)


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.0, min_delta: float = 0.0, max_pairs: int = 2048):
        super().__init__()
        self.margin = float(margin)
        self.min_delta = float(min_delta)
        self.max_pairs = int(max_pairs)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.float().view(-1)
        targets = targets.float().view(-1)
        if preds.numel() < 2:
            return preds.new_tensor(0.0)
        diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        valid = diff.abs() >= self.min_delta
        valid.fill_diagonal_(False)
        pair_idx = valid.nonzero(as_tuple=False)
        if pair_idx.numel() == 0:
            return preds.new_tensor(0.0)
        if pair_idx.size(0) > self.max_pairs:
            perm = torch.randperm(pair_idx.size(0), device=preds.device)[: self.max_pairs]
            pair_idx = pair_idx[perm]
        i = pair_idx[:, 0]
        j = pair_idx[:, 1]
        y_ij = torch.sign(targets[i] - targets[j])
        pred_diff = preds[i] - preds[j]
        losses = F.relu(self.margin - y_ij * pred_diff)
        return losses.mean()

class ASOHeadMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


class ASOAdaMBindBaseModel(nn.Module):
    """
    Ablation variant (no Gate fusion):
      1) frozen RiNALMo token features
      2) chemistry / backbone one-hot projections
      3) fixed ungated fusion (lm_proj + mod_to_lm)
      4) multi-scale local token mixer on fused / mod branches
      5) position-aware attention pooling + tri-region pooling
      6) concatenate pooled ASO features + method-scaled dosage
    """

    def __init__(
        self,
        lm_config: str,
        chem_vocab_size: int,
        chem_embed_dim: int,
        backbone_vocab_size: int,
        backbone_embed_dim: int,
        transfection_method_vocab_size: int,
        transfection_method_embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pooling_projection_dim: int = 64,
        lm_proj_dim: int = 128,
        chem_proj_dim: int = 32,
        backbone_proj_dim: int = 32,
        mod_proj_dim: int = 64,
        gate_hidden_dim: int = 128,
        region_projection_dim: int = 32,
        max_positions: int = 256,
        min_center_run: int = 3,
    ):
        super().__init__()
        self.lm = RiNALMo(model_config(lm_config))
        lm_embed_dim = self.lm.config["model"]["transformer"].embed_dim

        self.chem_embed_dim = int(chem_embed_dim)
        self.backbone_embed_dim = int(backbone_embed_dim)

        self.chem_vocab_size = int(chem_vocab_size)
        self.backbone_vocab_size = int(backbone_vocab_size)
        self.chem_pad_idx = 0
        self.backbone_pad_idx = 0
        self.pad_idx = self.lm.config["model"]["embedding"].padding_idx

        self.transfection_method_embedder = nn.Embedding(
            num_embeddings=transfection_method_vocab_size,
            embedding_dim=transfection_method_embed_dim,
        )

        self.lm_proj = nn.Sequential(
            nn.LayerNorm(lm_embed_dim),
            nn.Linear(lm_embed_dim, lm_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.chem_proj = nn.Sequential(
            nn.Linear(self.chem_vocab_size, chem_proj_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.backbone_proj = nn.Sequential(
            nn.Linear(self.backbone_vocab_size, backbone_proj_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mod_proj = nn.Sequential(
            nn.LayerNorm(chem_proj_dim + backbone_proj_dim),
            nn.Linear(chem_proj_dim + backbone_proj_dim, mod_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mod_proj_dim, mod_proj_dim),
            nn.GELU(),
        )
        self.mod_to_lm = nn.Sequential(
            nn.Linear(mod_proj_dim, lm_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # No-gate ablation: remove learnable gate and use fixed fusion instead.
        _ = gate_hidden_dim  # kept only for CLI/checkpoint compatibility

        self.fused_local_mixer = MultiScaleConvBlock(lm_proj_dim, dropout=dropout)
        self.mod_local_mixer = MultiScaleConvBlock(mod_proj_dim, dropout=dropout)
        self.fused_aso_pooler = PositionAwareAttentionPooling(
            lm_proj_dim,
            projection_dim=pooling_projection_dim,
            max_positions=max_positions,
            dropout=dropout,
        )
        self.mod_pooler = PositionAwareAttentionPooling(
            mod_proj_dim,
            projection_dim=pooling_projection_dim,
            max_positions=max_positions,
            dropout=dropout,
        )
        self.fused_region_pooler = ModAwareRegionPooling(lm_proj_dim, projection_dim=region_projection_dim, min_center_run=min_center_run)

        mlp_in_dim = (
            self.fused_aso_pooler.output_dim
            + self.mod_pooler.output_dim
            + self.fused_region_pooler.output_dim
            + transfection_method_embed_dim
        )
        self.pred_head = ASOHeadMLP(
            mlp_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def load_pretrained_lm_weights(self, pretrained_weights_path: str):
        self.lm.load_state_dict(torch.load(pretrained_weights_path, map_location="cpu"))

    def _align_aso_repr_to_mod_tokens(
        self,
        aso_repr_full: torch.Tensor,
        chem_tokens: torch.Tensor,
        backbone_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if chem_tokens.size(1) != backbone_tokens.size(1):
            raise ValueError(
                f"chem_tokens and backbone_tokens length mismatch: "
                f"{tuple(chem_tokens.shape)} vs {tuple(backbone_tokens.shape)}"
            )

        l_lm = aso_repr_full.size(1)
        l_mod = chem_tokens.size(1)
        if l_mod == l_lm:
            return aso_repr_full, chem_tokens, backbone_tokens
        if l_lm >= l_mod + 2:
            aso_repr = aso_repr_full[:, 1:1 + l_mod, :]
            return aso_repr, chem_tokens, backbone_tokens
        raise ValueError(
            f"Cannot align RiNALMo ASO repr to modification tensors: l_lm={l_lm}, l_mod={l_mod}."
        )

    def _build_valid_mod_mask(
        self,
        aso_tokens: torch.Tensor,
        chem_tokens: torch.Tensor,
        backbone_tokens: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = chem_tokens.shape
        if seq_len == aso_tokens.size(1):
            chem_counts = chem_tokens.ne(self.chem_pad_idx).sum(dim=1)
            bb_counts = backbone_tokens.ne(self.backbone_pad_idx).sum(dim=1)
            real_counts = torch.maximum(chem_counts, bb_counts)
            pos = torch.arange(seq_len, device=chem_tokens.device).unsqueeze(0)
            return (pos >= 1) & (pos < (1 + real_counts).unsqueeze(1))

        return chem_tokens.ne(self.chem_pad_idx) | backbone_tokens.ne(self.backbone_pad_idx)

    def forward(
        self,
        aso_tokens: torch.Tensor,
        chem_tokens: torch.Tensor,
        backbone_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        dosage: torch.Tensor,
        transfection_method_tokens: torch.Tensor,
    ) -> torch.Tensor:
        _ = context_tokens
        aso_repr_full = self.lm(aso_tokens)["representation"]

        aso_repr, chem_tokens_aligned, backbone_tokens_aligned = self._align_aso_repr_to_mod_tokens(
            aso_repr_full=aso_repr_full,
            chem_tokens=chem_tokens,
            backbone_tokens=backbone_tokens,
        )

        valid_mod_mask = self._build_valid_mod_mask(
            aso_tokens=aso_tokens[:, :chem_tokens_aligned.size(1)],
            chem_tokens=chem_tokens_aligned,
            backbone_tokens=backbone_tokens_aligned,
        )
        aso_pad_mask = ~valid_mod_mask
        valid_mod_mask_f = valid_mod_mask.unsqueeze(-1).to(dtype=aso_repr.dtype)

        chem_onehot = F.one_hot(
            chem_tokens_aligned.clamp(min=0),
            num_classes=self.chem_vocab_size,
        ).to(dtype=aso_repr.dtype)
        backbone_onehot = F.one_hot(
            backbone_tokens_aligned.clamp(min=0),
            num_classes=self.backbone_vocab_size,
        ).to(dtype=aso_repr.dtype)
        chem_onehot = chem_onehot * valid_mod_mask_f
        backbone_onehot = backbone_onehot * valid_mod_mask_f

        lm_proj = self.lm_proj(aso_repr)
        chem_proj = self.chem_proj(chem_onehot)
        backbone_proj = self.backbone_proj(backbone_onehot)

        mod_input = torch.cat([chem_proj, backbone_proj], dim=-1)
        mod_proj = self.mod_proj(mod_input)

        mod_to_lm = self.mod_to_lm(mod_proj)
        # No-gate ablation: fixed ungated fusion.
        fused = lm_proj + mod_to_lm
        fused = fused * valid_mod_mask_f
        mod_proj = mod_proj * valid_mod_mask_f

        fused = self.fused_local_mixer(fused, aso_pad_mask)
        mod_proj = self.mod_local_mixer(mod_proj, aso_pad_mask)

        pooled_fused = self.fused_aso_pooler(fused, aso_pad_mask)
        pooled_mod = self.mod_pooler(mod_proj, aso_pad_mask)
        pooled_regions = self.fused_region_pooler(fused, aso_pad_mask, chem_tokens_aligned)

        scaled_dosage = torch.log1p(dosage)
        transfection_method_embeds = self.transfection_method_embedder(transfection_method_tokens)
        method_scaled_dosage = transfection_method_embeds * scaled_dosage.unsqueeze(-1)

        features = torch.cat([pooled_fused, pooled_mod, pooled_regions, method_scaled_dosage], dim=-1)
        pred = self.pred_head(features)
        return pred


# -------------------------------
# No adaptive task scheduler is used in this ablation.

# Utility helpers
# -------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_cuda_backends(args: argparse.Namespace):
    """Work around cuDNN train-library mismatches by optionally disabling cuDNN."""
    disable_cudnn = bool(int(getattr(args, "disable_cudnn", 1)))
    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[backend] Disabled cuDNN; Conv1d/attention pooling will use PyTorch fallback kernels.")
    else:
        print("[backend] cuDNN remains enabled.")


def get_device(accelerator: str = "auto", devices: str = "auto") -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if isinstance(devices, str) and devices not in ("auto", ""):
            first = devices.split(",")[0].strip()
            if first.isdigit():
                return torch.device(f"cuda:{first}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def maybe_init_wandb(args: argparse.Namespace) -> Optional[Any]:
    if not args.wandb:
        return None
    if wandb is None:
        print("[WARN] wandb is not installed. Proceeding without Weights & Biases logging.")
        return None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_experiment_name,
        dir=args.output_dir,
        config=vars(args),
        reinit=True,
    )


def log_metrics(run: Optional[Any], metrics: Dict[str, float], step: int):
    pretty = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float, np.floating))])
    print(f"[step {step}] {pretty}")
    if run is not None:
        run.log(metrics, step=step)


def move_episode_to_device(episode: Dict[str, Dict[str, Any]], device: torch.device) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {"support": {}, "query": {}}
    for split in ("support", "query"):
        for key, value in episode[split].items():
            if torch.is_tensor(value):
                out[split][key] = value.to(device, non_blocking=True)
            else:
                out[split][key] = value
    return out


def collate_meta_items(items: Sequence[Tuple[Any, ...]], n_support: int) -> Dict[str, Dict[str, Any]]:
    support_items = list(items[:n_support])
    query_items = list(items[n_support:])

    def _stack(group: Sequence[Tuple[Any, ...]], idx: int) -> torch.Tensor:
        return torch.stack([x[idx] for x in group], dim=0)

    def _pack(group: Sequence[Tuple[Any, ...]]) -> Dict[str, Any]:
        return {
            "aso": _stack(group, 0),
            "chem": _stack(group, 1),
            "backbone": _stack(group, 2),
            "context": _stack(group, 3),
            "y": _stack(group, 4).float(),
            "dosage": _stack(group, 5).float(),
            "method": _stack(group, 6).long(),
            "custom_id": [x[7] for x in group],
            "task_id": [x[8] for x in group],
        }

    return {"support": _pack(support_items), "query": _pack(query_items)}


def build_subset_task_to_local_indices(subset: Subset) -> Dict[str, List[int]]:
    base_dataset = subset.dataset
    subset_base_indices: List[int] = list(subset.indices)
    base_to_subset = {base_idx: pos for pos, base_idx in enumerate(subset_base_indices)}
    subset_base_set = set(subset_base_indices)

    task_to_local: Dict[str, List[int]] = {}
    for task_id, base_idxs in base_dataset.task_to_indices.items():
        kept = [base_to_subset[i] for i in base_idxs if i in subset_base_set]
        if len(kept) >= 2:
            task_to_local[str(task_id)] = kept
    return task_to_local


class DeterministicTaskEpisodeBuilder:
    def __init__(
        self,
        subset: Subset,
        n_support: int,
        n_query: int,
        seed: int,
    ):
        self.subset = subset
        self.n_support = int(n_support)
        self.n_query = int(n_query)
        self.seed = int(seed)
        self.task_to_local_indices = build_subset_task_to_local_indices(subset)

    def build(self, task_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
        local_indices = self.task_to_local_indices.get(str(task_id), [])
        if len(local_indices) < 2:
            return None

        rng = random.Random(f"{self.seed}:{task_id}")
        all_indices = list(local_indices)
        rng.shuffle(all_indices)

        max_support = max(1, len(all_indices) - 1)
        support_size = min(self.n_support, max_support)
        support_indices = all_indices[:support_size]
        remaining = all_indices[support_size:]

        if len(remaining) == 0:
            # Ensure there is at least one query example.
            remaining = [support_indices.pop()]
            if len(support_indices) == 0:
                return None

        if self.n_query > 0 and len(remaining) > self.n_query:
            query_indices = remaining[: self.n_query]
        else:
            query_indices = remaining

        if len(query_indices) == 0:
            return None

        items = [self.subset[i] for i in support_indices + query_indices]
        return collate_meta_items(items, n_support=len(support_indices))

    def task_ids(self) -> List[str]:
        return sorted(self.task_to_local_indices.keys())


def select_named_trainable_parameters(model: nn.Module) -> OrderedDict:
    return OrderedDict((name, param) for name, param in model.named_parameters() if param.requires_grad)


def get_named_buffers(model: nn.Module) -> OrderedDict:
    return OrderedDict((name, buf) for name, buf in model.named_buffers())


def flatten_grad_cosines(
    support_grads: Sequence[Optional[torch.Tensor]],
    query_grads: Sequence[Optional[torch.Tensor]],
) -> torch.Tensor:
    cosines: List[torch.Tensor] = []
    for g1, g2 in zip(support_grads, query_grads):
        if g1 is None or g2 is None:
            cosines.append(torch.tensor(0.0, device=g1.device if g1 is not None else g2.device))
            continue
        v1 = g1.reshape(-1)
        v2 = g2.reshape(-1)
        denom = v1.norm() * v2.norm()
        if torch.isfinite(denom) and denom.item() > 0:
            cos = torch.dot(v1, v2) / denom.clamp_min(1e-12)
        else:
            cos = torch.tensor(0.0, device=v1.device)
        cosines.append(cos)
    return torch.stack(cosines)


class FenwickTree:
    def __init__(self, size: int):
        self.size = int(size)
        self.tree = np.zeros(self.size + 1, dtype=np.int64)

    def add(self, idx: int, value: int = 1):
        while idx <= self.size:
            self.tree[idx] += value
            idx += idx & -idx

    def prefix_sum(self, idx: int) -> int:
        total = 0
        while idx > 0:
            total += int(self.tree[idx])
            idx -= idx & -idx
        return total



def safe_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if mean_squared_error is not None:
        return float(mean_squared_error(y_true, y_pred))
    return float(np.mean((y_true - y_pred) ** 2))



def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    if r2_score is not None:
        return float(r2_score(y_true, y_pred))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)



def safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if mean_absolute_error is not None:
        return float(mean_absolute_error(y_true, y_pred))
    return float(np.mean(np.abs(y_true - y_pred)))



def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = safe_mse(y_true, y_pred)
    if np.isnan(mse):
        return float("nan")
    return float(np.sqrt(mse))



def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(corr):
        return 0.0
    return float(corr)



def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    y_true_std = float(np.std(y_true))
    y_pred_std = float(np.std(y_pred))
    if y_true_std < 1e-12 or y_pred_std < 1e-12:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)



def safe_concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(y_true, kind="mergesort")
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    unique_pred = np.unique(y_pred_sorted)
    pred_rank = {float(v): i + 1 for i, v in enumerate(unique_pred.tolist())}
    tree = FenwickTree(len(unique_pred))

    comparable = 0.0
    concordant = 0.0
    n_prev = 0
    start = 0
    n = y_true_sorted.size

    while start < n:
        end = start + 1
        while end < n and y_true_sorted[end] == y_true_sorted[start]:
            end += 1

        group_preds = y_pred_sorted[start:end]
        for pred in group_preds:
            rank = pred_rank[float(pred)]
            num_less = tree.prefix_sum(rank - 1)
            num_equal = tree.prefix_sum(rank) - num_less
            comparable += n_prev
            concordant += num_less + 0.5 * num_equal

        for pred in group_preds:
            rank = pred_rank[float(pred)]
            tree.add(rank, 1)
            n_prev += 1

        start = end

    if comparable <= 0:
        return 0.0
    return float(concordant / comparable)


# -------------------------------
# Meta-learning core
# -------------------------------

class MetaRunner:
    def __init__(
        self,
        model: ASOAdaMBindBaseModel,
        scaler: StandardScaler,
        device: torch.device,
        args: argparse.Namespace,
    ):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.args = args
        self.loss_fn = nn.MSELoss()
        self.huber_loss_fn = nn.HuberLoss(delta=args.huber_delta)
        self.rank_loss_fn = PairwiseRankingLoss(margin=args.rank_margin, min_delta=args.rank_min_delta, max_pairs=args.rank_max_pairs)

        precision = str(getattr(args, "precision", "16-mixed")).lower()
        self.use_amp = False
        self.amp_dtype = None
        if self.device.type == "cuda":
            if precision in ("16", "16-mixed", "fp16", "float16"):
                self.use_amp = True
                self.amp_dtype = torch.float16
            elif precision in ("bf16", "bf16-mixed", "bfloat16"):
                self.use_amp = True
                self.amp_dtype = torch.bfloat16

        self.outer_optimizer = Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        total_steps = max(1, args.max_epochs * args.steps_per_epoch)
        self.outer_scheduler = LinearLR(
            self.outer_optimizer,
            start_factor=1.0,
            end_factor=args.lr_end_factor,
            total_iters=total_steps,
        )

    def _scale_targets(self, targets: torch.Tensor) -> torch.Tensor:
        # Be robust to scaler implementations that keep mean/std on CPU.
        targets_cpu = targets.detach().cpu()
        if targets_cpu.ndim == 1:
            scaled = self.scaler.transform(targets_cpu.unsqueeze(-1)).squeeze(-1)
        else:
            scaled = self.scaler.transform(targets_cpu)
        return scaled.to(targets.device)

    def _unscale_predictions(self, preds_scaled: torch.Tensor) -> torch.Tensor:
        # Metrics use detached predictions, so CPU round-trip is safe here.
        preds_cpu = preds_scaled.detach().cpu()
        if preds_cpu.ndim == 1:
            preds = self.scaler.inverse_transform(preds_cpu.unsqueeze(-1)).squeeze(-1)
        else:
            preds = self.scaler.inverse_transform(preds_cpu)
        return preds.to(preds_scaled.device)

    def _compute_loss(self, preds_scaled: torch.Tensor, targets_scaled: torch.Tensor) -> torch.Tensor:
        # When the model forward runs under autocast, preds may be fp16/bf16 while
        # scaler outputs are float32. Computing the loss explicitly in float32 keeps
        # backward numerically stable and avoids half/float mismatch in autograd.
        preds = preds_scaled.float()
        targets = targets_scaled.float()
        if str(self.args.point_loss_type).lower() == "huber":
            point_loss = self.huber_loss_fn(preds, targets)
        else:
            point_loss = self.loss_fn(preds, targets)
        if self.args.rank_loss_weight > 0:
            rank_loss = self.rank_loss_fn(preds, targets)
            return point_loss + self.args.rank_loss_weight * rank_loss
        return point_loss

    def _maybe_noisy_targets(self, targets: torch.Tensor, training: bool) -> torch.Tensor:
        # w/o Label Noise ablation: always use the original labels.
        # `training` and CLI args `--noise/--noise_val` are intentionally ignored.
        _ = training
        return targets

    def _forward_with_state(self, batch: Dict[str, Any], params: OrderedDict, buffers: OrderedDict) -> torch.Tensor:
        state = OrderedDict()
        state.update(buffers)
        state.update(params)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.use_amp and self.amp_dtype is not None
            else contextlib.nullcontext()
        )
        with amp_ctx:
            return functional_call(
                self.model,
                state,
                (
                    batch["aso"],
                    batch["chem"],
                    batch["backbone"],
                    batch["context"],
                    batch["dosage"],
                    batch["method"],
                ),
            )

    def _adapt_single_task(
        self,
        episode: Dict[str, Dict[str, Any]],
        create_graph: bool,
        training: bool,
        base_params: Optional[OrderedDict] = None,
    ) -> Dict[str, Any]:
        named_params = select_named_trainable_parameters(self.model)
        if base_params is None:
            base_params = OrderedDict((k, v) for k, v in named_params.items())
        else:
            base_params = OrderedDict((k, v) for k, v in base_params.items())
        buffers = get_named_buffers(self.model)

        support = episode["support"]
        query = episode["query"]

        fast_params = OrderedDict((k, v) for k, v in base_params.items())
        last_support_loss = None
        last_support_grads: List[Optional[torch.Tensor]] = []

        for _ in range(self.args.update_step_train if training else self.args.update_step_test):
            support_targets_raw = self._maybe_noisy_targets(support["y"], training=training)
            support_targets_scaled = self._scale_targets(support_targets_raw)
            support_preds_scaled = self._forward_with_state(support, fast_params, buffers)
            last_support_loss = self._compute_loss(support_preds_scaled, support_targets_scaled)
            grads = torch.autograd.grad(
                last_support_loss,
                tuple(fast_params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )
            last_support_grads = list(grads)
            fast_params = OrderedDict(
                (
                    name,
                    param if grad is None else param - self.args.inner_lr * grad,
                )
                for (name, param), grad in zip(fast_params.items(), grads)
            )

        query_targets_raw = self._maybe_noisy_targets(query["y"], training=training)
        query_targets_scaled = self._scale_targets(query_targets_raw)
        query_preds_scaled = self._forward_with_state(query, fast_params, buffers)
        query_loss = self._compute_loss(query_preds_scaled, query_targets_scaled)
        query_grads = torch.autograd.grad(
            query_loss,
            tuple(fast_params.values()),
            create_graph=False,
            retain_graph=True,
            allow_unused=True,
        )
        query_preds_raw = self._unscale_predictions(query_preds_scaled.detach())

        return {
            "fast_params": fast_params,
            "support_loss": last_support_loss,
            "support_grads": last_support_grads,
            "query_loss": query_loss,
            "query_grads": list(query_grads),
            "query_preds_raw": query_preds_raw,
            "query_targets_raw": query["y"].detach(),
            "custom_ids": query["custom_id"],
            "task_id": query["task_id"][0] if len(query["task_id"]) > 0 else "UNKNOWN",
        }

    def _collect_buffer_task_infos(
        self,
        episodes: Sequence[Dict[str, Dict[str, Any]]],
        create_graph: bool,
        training: bool,
    ) -> List[Dict[str, Any]]:
        task_infos = []
        for ep in episodes:
            task_infos.append(self._adapt_single_task(ep, create_graph=create_graph, training=training))
        return task_infos

    def _virtual_outer_update(
        self,
        selected_task_infos: Sequence[Dict[str, Any]],
    ) -> OrderedDict:
        selected_query_loss = torch.stack([info["query_loss"] for info in selected_task_infos]).mean()
        named_params = select_named_trainable_parameters(self.model)
        grads = torch.autograd.grad(
            selected_query_loss,
            tuple(named_params.values()),
            create_graph=False,
            retain_graph=True,
            allow_unused=True,
        )
        virtual_params = OrderedDict(
            (
                name,
                param if grad is None else param - self.args.lr * grad,
            )
            for (name, param), grad in zip(named_params.items(), grads)
        )
        return virtual_params

    def _evaluate_episode_with_params(self, episode: Dict[str, Dict[str, Any]], base_params: OrderedDict) -> float:
        self.model.train()
        with torch.enable_grad():
            info = self._adapt_single_task(episode, create_graph=False, training=False, base_params=base_params)
        return float(info["query_loss"].detach().cpu())

    def meta_train_step(
        self,
        train_buffer_episodes: Sequence[Dict[str, Dict[str, Any]]],
        progress_bin: int = 0,
    ) -> Dict[str, float]:
        """
        No-adaptive-scheduler ablation.

        The full AdaMBind-style script scores a candidate task buffer using query loss,
        support/query gradient similarities, and training progress, then samples a
        meta-batch with a learned scheduler. This ablation removes that whole
        scheduler/reward path and samples tasks uniformly from the buffer.
        Meta-learning and inner-loop support adaptation,
        point/rank loss, and evaluation protocol are otherwise preserved.
        """
        _ = progress_bin
        self.model.train()

        buffer_size = len(train_buffer_episodes)
        if buffer_size == 0:
            raise ValueError("meta_train_step received an empty train_buffer_episodes list.")

        if buffer_size >= self.args.meta_batch_size:
            sampled_idx = torch.randperm(buffer_size, device=self.device)[: self.args.meta_batch_size].tolist()
        else:
            sampled_idx = torch.randint(0, buffer_size, (self.args.meta_batch_size,), device=self.device).tolist()

        selected_episodes = [train_buffer_episodes[i] for i in sampled_idx]
        selected_infos = self._collect_buffer_task_infos(selected_episodes, create_graph=False, training=True)
        selected_query_loss = torch.stack([info["query_loss"] for info in selected_infos]).mean()
        selected_task_ids = [info["task_id"] for info in selected_infos]

        self.outer_optimizer.zero_grad(set_to_none=True)
        selected_query_loss.backward()
        if self.args.gradient_clip_val is not None and self.args.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_val)
        self.outer_optimizer.step()
        self.outer_scheduler.step()

        return {
            "train/meta_query_loss": float(selected_query_loss.detach().cpu()),
            "train/mean_buffer_query_loss": float(selected_query_loss.detach().cpu()),
            "train/selected_task_count": float(len(selected_task_ids)),
            "train/sampler": 0.0,
        }

    @torch.no_grad()
    def meta_evaluate_builder(
        self,
        builder: DeterministicTaskEpisodeBuilder,
        split_name: str,
    ) -> Dict[str, float]:
        self.model.eval()
        all_preds: List[float] = []
        all_targets: List[float] = []
        n_tasks = 0

        for task_id in builder.task_ids():
            episode = builder.build(task_id)
            if episode is None:
                continue
            episode = move_episode_to_device(episode, self.device)
            with torch.enable_grad():
                info = self._adapt_single_task(episode, create_graph=False, training=False)
            preds = info["query_preds_raw"].detach().cpu().numpy().astype(float)
            targets = info["query_targets_raw"].detach().cpu().numpy().astype(float)
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            n_tasks += 1

        if len(all_targets) == 0:
            return {
                f"{split_name}/mse": float("nan"),
                f"{split_name}/ci": float("nan"),
                f"{split_name}/r2": float("nan"),
                f"{split_name}/spearman": float("nan"),
                f"{split_name}/pearson": float("nan"),
                f"{split_name}/mae_final": float("nan"),
                f"{split_name}/rmse": float("nan"),
                f"{split_name}/task_count": 0.0,
                f"{split_name}/sample_count": 0.0,
            }

        y_true = np.asarray(all_targets, dtype=float)
        y_pred = np.asarray(all_preds, dtype=float)

        return {
            f"{split_name}/mse": safe_mse(y_true, y_pred),
            f"{split_name}/ci": safe_concordance_index(y_true, y_pred),
            f"{split_name}/r2": safe_r2(y_true, y_pred),
            f"{split_name}/spearman": safe_spearman(y_true, y_pred),
            f"{split_name}/pearson": safe_pearson(y_true, y_pred),
            f"{split_name}/mae_final": safe_mae(y_true, y_pred),
            f"{split_name}/rmse": safe_rmse(y_true, y_pred),
            f"{split_name}/task_count": float(n_tasks),
            f"{split_name}/sample_count": float(y_true.size),
        }


# -------------------------------
# Training script
# -------------------------------

def draw_train_buffer(
    iterator: Iterable[Dict[str, Dict[str, Any]]],
    loader: Iterable[Dict[str, Dict[str, Any]]],
    buffer_size: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Dict[str, Any]]], Iterable[Dict[str, Dict[str, Any]]]]:
    episodes: List[Dict[str, Dict[str, Any]]] = []
    itr = iterator
    for _ in range(buffer_size):
        try:
            batch = next(itr)
        except StopIteration:
            itr = iter(loader)
            batch = next(itr)
        episodes.append(move_episode_to_device(batch, device))
    return episodes, itr


def draw_reward_episodes(
    iterator: Iterable[Dict[str, Dict[str, Any]]],
    loader: Iterable[Dict[str, Dict[str, Any]]],
    num_episodes: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Dict[str, Any]]], Iterable[Dict[str, Dict[str, Any]]]]:
    episodes: List[Dict[str, Dict[str, Any]]] = []
    itr = iterator
    for _ in range(max(0, num_episodes)):
        try:
            batch = next(itr)
        except StopIteration:
            itr = iter(loader)
            batch = next(itr)
        episodes.append(move_episode_to_device(batch, device))
    return episodes, itr


def fit_scaler_from_train_dataset(datamodule: ASODataModule) -> StandardScaler:
    scaler = StandardScaler()
    train_targets = [item[4] for item in datamodule.train_dataset]
    scaler.partial_fit(torch.tensor(train_targets, dtype=torch.float32).unsqueeze(-1))
    return scaler


class BasicFTSchedule:
    """
    Minimal best-effort support for gradual unfreezing in this standalone meta-learning loop.
    Supported YAML forms:
      - list of dicts: [{epoch: 0, patterns: ["pred_head", "aso_feature_combiner"]}, ...]
      - dict with key "schedule" or "stages" pointing to the same list

    Any other schema is ignored with a warning.
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        self.stages: List[Dict[str, Any]] = []
        self.applied_epochs: set = set()
        if path is None:
            return
        if yaml is None:
            print("[WARN] PyYAML is not installed. ft_schedule will be ignored.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f)
        except Exception as exc:
            print(f"[WARN] Failed to read ft_schedule={path}: {exc}. The schedule will be ignored.")
            return
        if isinstance(payload, dict):
            if isinstance(payload.get("schedule"), list):
                self.stages = payload["schedule"]
            elif isinstance(payload.get("stages"), list):
                self.stages = payload["stages"]
        elif isinstance(payload, list):
            self.stages = payload
        if not self.stages:
            print("[WARN] ft_schedule was provided, but no supported schedule entries were found. Ignoring it.")

    def maybe_apply(self, model: nn.Module, epoch: int) -> bool:
        if not self.stages:
            return False
        changed = False
        for stage in self.stages:
            stage_epoch = stage.get("epoch", stage.get("at_epoch", stage.get("start_epoch", None)))
            if stage_epoch is None or int(stage_epoch) > int(epoch) or int(stage_epoch) in self.applied_epochs:
                continue
            patterns = stage.get("patterns", stage.get("modules", stage.get("names", [])))
            if isinstance(patterns, str):
                patterns = [patterns]
            if not isinstance(patterns, list) or len(patterns) == 0:
                continue
            matched_any = False
            for name, param in model.named_parameters():
                if any(str(pat) in name for pat in patterns):
                    if not param.requires_grad:
                        param.requires_grad = True
                    matched_any = True
                    changed = True
            if matched_any:
                print(f"[ft_schedule] epoch={epoch}: unfroze patterns {patterns}")
                self.applied_epochs.add(int(stage_epoch))
        return changed


def freeze_backbone_if_schedule_present(model: nn.Module, ft_schedule: BasicFTSchedule):
    if not ft_schedule.stages:
        return
    for name, param in model.named_parameters():
        if name.startswith("lm"):
            param.requires_grad = False
    print("[ft_schedule] Initial state: froze `lm.*` parameters. Non-LM layers remain trainable.")


def freeze_lm_only(model: nn.Module):
    for name, param in model.named_parameters():
        if name.startswith("lm"):
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("[freeze] Froze `lm.*` completely. All RiNALMo parameters remain frozen; only non-LM layers are trainable.")


def freeze_all_except_head_and_combiner(model: nn.Module):
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_modules = (
        "lm_proj",
        "chem_proj",
        "backbone_proj",
        "mod_proj",
        "mod_to_lm",
        "gate_mlp",
        "fused_aso_pooler",
        "mod_pooler",
        "fused_local_mixer",
        "mod_local_mixer",
        "fused_region_pooler",
        "transfection_method_embedder",
        "pred_head",
    )
    for module_name in trainable_modules:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True
    print("[freeze] Froze RiNALMo backbone; training projection/fusion/pooling/transfection/head modules only.")


def summarize_trainable_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"[params] trainable={trainable:,} / total={total:,} ({100.0 * trainable / max(1, total):.2f}%), frozen={frozen:,}")

    module_stats = {}
    for name, param in model.named_parameters():
        root = name.split(".", 1)[0]
        stats = module_stats.setdefault(root, {"trainable": 0, "total": 0})
        stats["total"] += param.numel()
        if param.requires_grad:
            stats["trainable"] += param.numel()

    print("[params] module-wise trainable summary:")
    for root in sorted(module_stats.keys()):
        stats = module_stats[root]
        print(
            f"  - {root}: trainable={stats['trainable']:,} / total={stats['total']:,} "
            f"({100.0 * stats['trainable'] / max(1, stats['total']):.2f}%)"
        )


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, epoch: int, step: int):
    payload = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "step": step,
        "ablation": "no_adaptive_task_scheduler",
    }
    torch.save(payload, path)


def main(args: argparse.Namespace):
    seed = args.seed if args.seed is not None else 42
    seed_everything(seed)
    configure_cuda_backends(args)

    output_dir = Path(args.output_dir) if args.output_dir is not None else Path("./aso_adambind_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.accelerator, args.devices)
    print(f"Using device: {device}")
    precision = str(args.precision).lower()
    if device.type == "cuda":
        if precision in ("16", "16-mixed", "fp16", "float16"):
            print("[INFO] AMP enabled with float16 autocast for RiNALMo/FlashAttention compatibility.")
        elif precision in ("bf16", "bf16-mixed", "bfloat16"):
            print("[INFO] AMP enabled with bfloat16 autocast for RiNALMo/FlashAttention compatibility.")
        else:
            print(
                f"[WARN] precision={args.precision} disables autocast. RiNALMo FlashAttention expects fp16/bf16; "
                "use --precision 16-mixed or --precision bf16-mixed, or disable FlashAttention in RiNALMo."
            )
    else:
        print("[INFO] Running on CPU; autocast for FlashAttention is not used.")

    alphabet = Alphabet()
    datamodule = ASODataModule(
        data_path=args.data_path,
        alphabet=alphabet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=seed,
        meta=True,
        n_support=args.n_support,
        n_query=args.n_query,
        episodes_per_epoch=args.steps_per_epoch * args.buffer_size,
        meta_val=True,
        val_episodes=max(args.steps_per_epoch * max(1, args.val_reward_episodes), args.val_reward_episodes),
        meta_test=True,
        test_episodes=max(args.test_eval_episodes, 1),
        sample_with_replacement_if_needed=args.sample_with_replacement_if_needed,
    )
    datamodule.setup(stage="fit")

    scaler = fit_scaler_from_train_dataset(datamodule)
    train_dataset = datamodule.train_dataset.dataset
    model = ASOAdaMBindBaseModel(
        lm_config=args.lm_config,
        chem_vocab_size=len(train_dataset.chem_vocab),
        chem_embed_dim=args.chem_embed_dim,
        backbone_vocab_size=len(train_dataset.backbone_vocab),
        backbone_embed_dim=args.backbone_embed_dim,
        transfection_method_vocab_size=len(train_dataset.transfection_method_vocab),
        transfection_method_embed_dim=args.transfection_method_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling_projection_dim=args.pooling_projection_dim,
        lm_proj_dim=args.lm_proj_dim,
        chem_proj_dim=args.chem_proj_dim,
        backbone_proj_dim=args.backbone_proj_dim,
        mod_proj_dim=args.mod_proj_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        region_projection_dim=args.region_projection_dim,
        max_positions=args.max_positions,
        min_center_run=args.min_center_run,
    )
    if args.pretrained_rinalmo_weights:
        model.load_pretrained_lm_weights(args.pretrained_rinalmo_weights)
    if args.init_params:
        ckpt = torch.load(args.init_params, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    if args.train_head_combiner_only:
        freeze_all_except_head_and_combiner(model)
    else:
        # Enforce the legacy fully-frozen RiNALMo behavior for all runs.
        # This disables any accidental LM finetuning from omitted CLI flags or ft_schedule.
        freeze_lm_only(model)
        if args.ft_schedule:
            print("[freeze] `--ft_schedule` was provided, but LM unfreezing is disabled in this no-context fully-frozen variant.")
    summarize_trainable_parameters(model)

    model.to(device)
    print("[ablation] Adaptive task scheduler is disabled; label noise strategy is disabled; meta-batches are sampled uniformly from the train episode buffer.")

    if args.test_only:
        runner = MetaRunner(model=model, scaler=scaler, device=device, args=args)
        train_builder = DeterministicTaskEpisodeBuilder(datamodule.train_dataset, args.n_support, args.eval_n_query, seed)
        val_builder = DeterministicTaskEpisodeBuilder(datamodule.val_dataset, args.n_support, args.eval_n_query, seed + 1)
        test_builder = DeterministicTaskEpisodeBuilder(datamodule.test_dataset, args.n_support, args.eval_n_query, seed + 2)
        train_metrics = runner.meta_evaluate_builder(train_builder, "train")
        val_metrics = runner.meta_evaluate_builder(val_builder, "val")
        test_metrics = runner.meta_evaluate_builder(test_builder, "test")
        print(json.dumps({**train_metrics, **val_metrics, **test_metrics}, indent=2))
        return

    run = maybe_init_wandb(args)
    runner = MetaRunner(model=model, scaler=scaler, device=device, args=args)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    train_builder = DeterministicTaskEpisodeBuilder(datamodule.train_dataset, args.n_support, args.eval_n_query, seed)
    val_builder = DeterministicTaskEpisodeBuilder(datamodule.val_dataset, args.n_support, args.eval_n_query, seed + 1)
    test_builder = DeterministicTaskEpisodeBuilder(datamodule.test_dataset, args.n_support, args.eval_n_query, seed + 2)

    best_val_mae = float("inf")
    best_ckpt_path = output_dir / "aso_no_label_noise_best.pt"
    last_ckpt_path = output_dir / "aso_no_label_noise_last.pt"
    global_step = 0

    for epoch in range(args.max_epochs):
        # LM remains fully frozen for the whole run; no ft_schedule unfreezing is applied.
        epoch_meta_losses: List[float] = []
        for step_idx in range(args.steps_per_epoch):
            progress_bin = int(100.0 * (epoch * args.steps_per_epoch + step_idx) / max(1, args.max_epochs * args.steps_per_epoch - 1))
            train_buffer, train_iter = draw_train_buffer(train_iter, train_loader, args.buffer_size, device)
            train_logs = runner.meta_train_step(train_buffer, progress_bin=progress_bin)
            global_step += 1
            epoch_meta_losses.append(train_logs["train/meta_query_loss"])
            if global_step % args.log_every_n_steps == 0:
                train_logs["lr"] = runner.outer_optimizer.param_groups[0]["lr"]
                log_metrics(run, train_logs, global_step)

        val_metrics = runner.meta_evaluate_builder(val_builder, "val")
        train_epoch_metrics = {"train/epoch_meta_query_loss": float(np.mean(epoch_meta_losses))}
        merged_metrics = {**train_epoch_metrics, **val_metrics, "epoch": float(epoch)}
        log_metrics(run, merged_metrics, global_step)

        current_val_mae = float(val_metrics.get("val/mae_final", float("inf")))
        if np.isfinite(current_val_mae) and current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            save_checkpoint(best_ckpt_path, model, args, epoch=epoch, step=global_step)
            print(f"[checkpoint] Saved best checkpoint to {best_ckpt_path} (val/mae_final={best_val_mae:.4f})")

        if args.checkpoint_every_epoch:
            epoch_ckpt = output_dir / f"aso_no_adaptive_scheduler_epoch_{epoch:03d}.pt"
            save_checkpoint(epoch_ckpt, model, args, epoch=epoch, step=global_step)

        save_checkpoint(last_ckpt_path, model, args, epoch=epoch, step=global_step)

    if best_ckpt_path.exists():
        best_payload = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(best_payload["model_state"], strict=False)
        model.to(device)

    final_train_metrics = runner.meta_evaluate_builder(train_builder, "train")
    final_val_metrics = runner.meta_evaluate_builder(val_builder, "val")
    final_test_metrics = runner.meta_evaluate_builder(test_builder, "test")
    final_metrics = {**final_train_metrics, **final_val_metrics, **final_test_metrics}
    log_metrics(run, final_metrics, global_step)
    print(json.dumps(final_metrics, indent=2))

    if run is not None:
        run.finish()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASO inhibition prediction meta-learning ablation without AdaMBind-style adaptive task scheduling")
    parser.add_argument("data_path", type=str, help="Path to the CSV/CSV.GZ file")
    parser.add_argument("--init_params", type=str, default=None, help="Path to a checkpoint or model weights")
    parser.add_argument("--output_dir", type=str, default="./aso_no_label_noise_outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_every_epoch", action="store_true", default=False)
    parser.add_argument("--test_only", action="store_true", default=False)

    # Feature/model args
    parser.add_argument("--lm_config", type=str, default="giga")
    parser.add_argument("--pretrained_rinalmo_weights", type=str, default=None)
    parser.add_argument("--chem_embed_dim", type=int, default=16, help="Legacy arg kept for CLI compatibility; the new ASO branch is one-hot based.")
    parser.add_argument("--backbone_embed_dim", type=int, default=8, help="Legacy arg kept for CLI compatibility; the new ASO branch is one-hot based.")
    parser.add_argument("--transfection_method_embed_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling_projection_dim", type=int, default=64)
    parser.add_argument("--lm_proj_dim", type=int, default=128)
    parser.add_argument("--chem_proj_dim", type=int, default=32)
    parser.add_argument("--backbone_proj_dim", type=int, default=32)
    parser.add_argument("--mod_proj_dim", type=int, default=64)
    parser.add_argument("--gate_hidden_dim", type=int, default=128, help="Legacy arg kept for CLI compatibility; unused in the no-gate ablation variant.")
    parser.add_argument("--region_projection_dim", type=int, default=32)
    parser.add_argument("--max_positions", type=int, default=256)
    parser.add_argument("--min_center_run", type=int, default=3)

    # Split / dataloader args preserved from train_aso.py and datamodule.py
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32, help="Unused for meta-train episodes, kept for compatibility")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--sample_with_replacement_if_needed", type=int, default=1, help="1 to allow episodic sampling with replacement when a task is small")

    # Logging / runtime args preserved for compatibility
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_experiment_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=20)
    parser.add_argument("--ft_schedule", type=str, default=None, help="Accepted for CLI compatibility but ignored; this variant keeps RiNALMo fully frozen")
    parser.add_argument(
        "--freeze_lm_only",
        action="store_true",
        default=True,
        help="Kept for CLI compatibility. RiNALMo is fully frozen by default in this variant.",
    )
    parser.add_argument(
        "--train_head_combiner_only",
        action="store_true",
        default=False,
        help="Optional stricter mode: freeze everything except projection/fusion/pooling/head modules.",
    )
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument(
        "--disable_cudnn",
        type=int,
        default=1,
        help="1 to disable cuDNN and avoid Conv1d backward failures on mismatched cuDNN installs",
    )

    # Meta-learning args
    parser.add_argument("--lr", type=float, default=5e-4, help="Outer-loop/meta learning rate")
    parser.add_argument("--inner_lr", type=float, default=1e-3, help="Inner-loop adaptation learning rate")
    parser.add_argument("--scheduler_lr", type=float, default=1e-4, help="Ignored in this no-adaptive-scheduler ablation; kept for CLI compatibility")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--update_step_train", type=int, default=5)
    parser.add_argument("--update_step_test", type=int, default=5)
    parser.add_argument("--n_support", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=8)
    parser.add_argument("--eval_n_query", type=int, default=0, help="0 means use all remaining task samples as query during eval")
    parser.add_argument("--buffer_size", type=int, default=15)
    parser.add_argument("--meta_batch_size", type=int, default=8)
    parser.add_argument("--val_reward_episodes", type=int, default=4, help="Ignored in this no-adaptive-scheduler ablation; kept for CLI compatibility")
    parser.add_argument("--test_eval_episodes", type=int, default=200)
    parser.add_argument("--noise", type=int, default=0, help="Ignored in this w/o-label-noise ablation; label noise is always disabled")
    parser.add_argument("--noise_val", type=float, default=0.0, help="Ignored in this w/o-label-noise ablation; kept for CLI compatibility")
    parser.add_argument("--lr_end_factor", type=float, default=0.1)
    parser.add_argument("--point_loss_type", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--rank_loss_weight", type=float, default=0.2)
    parser.add_argument("--rank_margin", type=float, default=0.0)
    parser.add_argument("--rank_min_delta", type=float, default=0.25)
    parser.add_argument("--rank_max_pairs", type=int, default=2048)

    # Ignored scheduler args retained only so old commands do not break.
    parser.add_argument("--scheduler_loss_hidden_dim", type=int, default=32, help="Ignored; kept for CLI compatibility")
    parser.add_argument("--scheduler_grad_hidden_dim", type=int, default=32, help="Ignored; kept for CLI compatibility")
    parser.add_argument("--scheduler_task_hidden_dim", type=int, default=64, help="Ignored; kept for CLI compatibility")
    parser.add_argument("--scheduler_reward_momentum", type=float, default=0.9, help="Ignored; kept for CLI compatibility")
    parser.add_argument("--scheduler_grad_clip_val", type=float, default=1.0, help="Ignored; kept for CLI compatibility")
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    if int(getattr(args, "noise", 0)) != 0 or float(getattr(args, "noise_val", 0.0)) != 0.0:
        print("[ablation] This is the w/o Label Noise variant: --noise and --noise_val are ignored and set to 0.")
    args.noise = 0
    args.noise_val = 0.0
    main(args)
