"""Offline whitening statistics for the frozen V-JEPA features (analysis helper).

Computes the FIXED training-set statistics used by the feature-whitening path
(`cfg.train.whiten_features`, tmp/changes_bottleneck <2>, investigation_014):
the per-feature mean `mu` and the eigendecomposition `Sigma = U Lambda U^T` of
the covariance of encoder tokens pooled over batches and token positions,

```text
X in R^{(rows) x D_e},  mu = E[X],  Sigma = Cov(X)
```

`train.py` builds `models.FeatureWhitener` from this file's output and applies
`e_w = (e - mu) @ U (Lambda + eps I)^{-1/2} U^T` at the encoder/bottleneck seam.
The eigenvalue floor `eps` is applied at load time (config `whiten_eps`), so one
stats file supports an eps sweep.

Contracts:

- Statistics come from the TRAINING split with the training transforms, matching
  the feature distribution the model optimizes on. They are computed once and
  reused for train/val/inference — never per batch.
- Accumulation is single-pass (count / sum / sum of outer products) in fp64 on
  the encoding device, so bf16 encoder outputs cannot drift the estimate.
- Like `drift_probe.py` / `rank_probe.py`, this is an offline helper: training
  code never imports it, and it logs nothing to W&B.

Usage (RunPod, downloads the encoder on first run):

```bash
python whiten_stats.py --data ssv2 --max-batches 200
python train.py --data ssv2 --whiten-features \
    --whiten-stats-path logs/whiten/whiten_stats_ssv2_train_seed42.pt
```
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local docs-only environments.
    torch = None  # type: ignore[assignment]

from config import Config


def _require_torch() -> None:
    """Fail fast when the stats pass is invoked without PyTorch installed."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for whiten_stats.py. Install requirements.txt on RunPod."
        )


class RunningMoments:
    """Single-pass fp64 accumulator for mean and covariance of feature rows.

    Args:
        d: Feature dimension (D_e).
        device: Accumulation device (matmuls run where the features are).
    """

    def __init__(self, d: int, device):
        """Allocate zeroed fp64 count/sum/outer-product accumulators."""
        _require_torch()
        self.rows = 0
        self.sum = torch.zeros(d, dtype=torch.float64, device=device)
        self.outer = torch.zeros(d, d, dtype=torch.float64, device=device)

    def update(self, features) -> None:
        """Fold a batch of encoder tokens into the accumulators.

        Args:
            features: (B, N, D_e) or (rows, D_e) frozen encoder features.
        """
        x = features.reshape(-1, features.shape[-1]).to(torch.float64)
        self.rows += x.shape[0]
        self.sum += x.sum(dim=0)
        self.outer += x.t() @ x

    def finalize(self):
        """Return `(mean, cov)` as fp64 CPU tensors (cov is the biased MLE / rows).

        Raises:
            RuntimeError: Fewer than 2 rows were accumulated.
        """
        if self.rows < 2:
            raise RuntimeError(f"Need at least 2 feature rows for covariance; got {self.rows}.")
        mean = self.sum / self.rows
        cov = self.outer / self.rows - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.t())  # exact symmetry for eigh
        return mean.cpu(), cov.cpu()


def compute_whitening_stats(cfg: Config, split: str, max_batches: int, device) -> dict:
    """Encode training clips and return the whitening stats payload.

    Args:
        cfg: Global config (dataset selection, batch size, encoder repo).
        split: Dataset split to draw clips from ("train" per the spec).
        max_batches: Number of dataloader batches to fold in (context windows only).
        device: Encoding device.
    Returns:
        Payload dict with `mean` (D_e,), `eigvals` (D_e,), `eigvecs` (D_e, D_e)
        fp32 tensors plus provenance metadata.
    """
    _require_torch()
    from data import build_dataloader
    from models import FrozenEncoder

    encoder = FrozenEncoder(cfg.model).to(device)
    loader = build_dataloader(cfg, split)
    moments = RunningMoments(cfg.model.d_e, device)
    start = time.time()
    with torch.no_grad():
        for i, (context_clip, _target_clip) in enumerate(loader):
            if i >= max_batches:
                break
            detailed = encoder(context_clip.to(device, non_blocking=True))
            moments.update(detailed)
            if (i + 1) % 10 == 0:
                print(
                    f"batch {i + 1}/{max_batches}  rows={moments.rows}  "
                    f"elapsed={time.time() - start:.0f}s"
                )
    mean, cov = moments.finalize()
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return {
        "mean": mean.float(),
        "eigvals": eigvals.float(),
        "eigvecs": eigvecs.float(),
        "rows": moments.rows,
        "dataset": cfg.data.dataset,
        "split": split,
        "seed": cfg.seed,
        "encoder_repo": cfg.model.encoder_repo,
        "d_e": cfg.model.d_e,
    }


def parse_args() -> argparse.Namespace:
    """Parse the whitening-stats CLI."""
    parser = argparse.ArgumentParser(
        description="Compute fixed offline whitening stats for frozen V-JEPA features."
    )
    parser.add_argument(
        "--data", choices=["ssv2", "ssv2_tiny", "ego4d", "ego4d_tiny"], default="ssv2_tiny"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation"],
        help="Split the statistics are computed over (default: train, per the spec).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=200,
        help="Dataloader batches folded into the estimate (200 x 64 clips x 1024 "
        "tokens ~= 13M rows; plenty for a 1024x1024 covariance).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=None,
        help="Output .pt path (default: logs/whiten/whiten_stats_<data>_<split>_seed<seed>.pt).",
    )
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    """Compute and save the whitening statistics file."""
    args = parse_args()  # before _require_torch so --help works on torch-less machines
    _require_torch()
    cfg = Config()
    cfg.data.dataset = args.data
    cfg.seed = args.seed
    from train import set_seed

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = Path(
        args.out or f"logs/whiten/whiten_stats_{args.data}_{args.split}_seed{args.seed}.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = compute_whitening_stats(cfg, args.split, args.max_batches, device)
    torch.save(payload, out_path)
    eig = payload["eigvals"].double().clamp_min(0.0)
    top = eig.flip(0)[:5].tolist()
    print(
        f"Wrote {out_path}\n"
        f"  rows={payload['rows']}  d_e={payload['d_e']}\n"
        f"  top-5 eigenvalues: {[f'{v:.4g}' for v in top]}\n"
        f"  min eigenvalue: {eig.min().item():.4g} "
        f"(whiten_eps floors this at load time; default 1e-4)"
    )


if __name__ == "__main__":
    main()
