"""Fit and inspect immutable whitening statistics for any registered encoder."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError:  # pragma: no cover - docs-only environments
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]

from config import Config


def _require_torch() -> None:
    """Fail before doing data/model work when PyTorch is unavailable."""
    if torch is None:
        raise RuntimeError("PyTorch is required for whiten_stats.py.")


class RunningMoments:
    """Accumulate an exact single-pass fp64 mean and covariance over feature rows.

    Keeping only first and second moments avoids retaining the full `(clips*N_e,D_e)`
    matrix while making the row budget explicit and auditable.
    """

    def __init__(self, feature_dim: int, device: torch.device) -> None:
        """Allocate the fp64 accumulator on the selected compute device.

        Args:
            feature_dim: Detailed feature width `D_e`.
            device: Device used for moment accumulation.
        Returns:
            None.
        """
        _require_torch()
        self.rows = 0
        self.sum = torch.zeros(feature_dim, dtype=torch.float64, device=device)
        self.outer = torch.zeros(feature_dim, feature_dim, dtype=torch.float64, device=device)

    def update(self, features: Tensor) -> None:
        """Fold `(B,N_e,D_e)` or `(rows,D_e)` features into the accumulator.

        Flattening only leading dimensions preserves every token row and makes the
        recorded row count independent of dataloader batch boundaries.

        Args:
            features: Detailed features with final dimension `D_e`.
        Returns:
            None.
        """
        rows = features.reshape(-1, features.shape[-1]).to(torch.float64)
        self.rows += rows.shape[0]
        self.sum += rows.sum(dim=0)
        self.outer += rows.t() @ rows

    def finalize(self) -> tuple[Tensor, Tensor]:
        """Return the fp64 CPU mean and symmetric biased covariance.

        The biased maximum-likelihood covariance matches the artifact's recorded
        eigensolver contract.

        Returns:
            `(mean, covariance)` with shapes `(D_e,)` and `(D_e,D_e)`.
        """
        if self.rows < 2:
            raise RuntimeError(f"Need at least 2 feature rows; got {self.rows}.")
        mean = self.sum / self.rows
        covariance = self.outer / self.rows - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.t())
        return mean.cpu(), covariance.cpu()


def compute_whitening_stats(
    cfg: Config,
    split: str,
    max_clips: int,
    device: torch.device,
    *,
    batch_size: int | None = None,
    _encoder: Any | None = None,
    _loader: Iterable[Any] | None = None,
    _dataset_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode a deterministic context-only prefix and return a bound envelope.

    Private injections keep the full artifact contract offline-testable; production
    always resolves the same factory and raw dataloader used by training.

    Args:
        cfg: Finalized runtime configuration selecting encoder, data, and seed.
        split: Dataset split; only `train` is accepted.
        max_clips: Exact deterministic clip budget.
        device: Encoder and accumulation device.
        batch_size: Optional dataloader batch override.
        _encoder: Offline injected frozen-encoder fixture.
        _loader: Offline injected context-only batch iterable.
        _dataset_identity: Offline injected dataset identity.
    Returns:
        Versioned whitening envelope bound to encoder, dataset, and sampling identity.
    """
    _require_torch()
    if split != "train":
        raise ValueError("Whitening statistics may only be fitted on the training split.")
    if max_clips <= 0:
        raise ValueError("max_clips must be positive.")
    from data import build_dataloader
    from encoders import build_frozen_encoder
    from provenance import (
        WHITENING_EIGENSOLVER,
        build_dataset_identity,
        build_whitening_envelope,
    )

    dataset_identity = _dataset_identity or build_dataset_identity(
        cfg, require_complete=cfg.data.dataset == "ego4d"
    )
    encoder = _encoder or build_frozen_encoder(cfg.encoder).to(device)
    loader = _loader or build_dataloader(
        cfg,
        split,
        batch_size=batch_size,
        needs_target=False,
        epoch=0,
        drop_last=False,
    )
    moments = RunningMoments(encoder.spec.feature_dim, device)
    clip_count = 0
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            context = batch.context if hasattr(batch, "context") else batch[0]
            remaining = max_clips - clip_count
            if remaining <= 0:
                break
            context = context[:remaining].to(device, non_blocking=True)
            detailed = encoder(context)
            moments.update(detailed)
            clip_count += context.shape[0]
            if clip_count % 640 == 0:
                print(
                    f"clips={clip_count}/{max_clips} rows={moments.rows} "
                    f"elapsed={time.time() - started:.0f}s"
                )
            if clip_count == max_clips:
                break
    if clip_count != max_clips:
        raise RuntimeError(
            f"Requested {max_clips} clips, but the deterministic loader yielded {clip_count}."
        )
    mean, covariance = moments.finalize()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if not all(torch.isfinite(value).all() for value in (mean, eigenvalues, eigenvectors)):
        raise RuntimeError("Whitening eigensystem contains non-finite values.")
    return build_whitening_envelope(
        mean=mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        encoder_spec=encoder.spec,
        dataset_identity=dataset_identity,
        split=split,
        transform_seed=cfg.seed,
        clip_count=clip_count,
        token_row_count=moments.rows,
        eigensolver=WHITENING_EIGENSOLVER,
    )


def inspect_stats(path: str | Path) -> dict[str, Any]:
    """Validate an artifact and print credential-free, JSON-safe metadata.

    Inspection uses the same strict loader as training, so a readable summary never
    blesses a malformed payload.

    Args:
        path: Whitening artifact to inspect.
    Returns:
        JSON-compatible validation report.
    """
    from provenance import load_whitening_envelope, sha256_file

    envelope = load_whitening_envelope(path)
    metadata = envelope["metadata"]
    tensors = envelope["tensors"]
    report = {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "payload_fingerprint": envelope["payload_fingerprint"],
        "feature_fingerprint": metadata["feature_fingerprint"],
        "encoder": metadata["encoder_spec"]["family"],
        "repo_id": metadata["encoder_spec"]["repo_id"],
        "resolved_revision": metadata["encoder_spec"]["resolved_revision"],
        "dataset": metadata["dataset_identity"].get("dataset"),
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "split": metadata["split"],
        "clip_count": metadata["clip_count"],
        "token_row_count": metadata["token_row_count"],
        "tokens_per_clip": metadata["encoder_spec"]["layout"]["temporal"]
        * metadata["encoder_spec"]["layout"]["height"]
        * metadata["encoder_spec"]["layout"]["width"],
        "feature_dim": int(tensors["mean"].shape[0]),
        "precision": metadata["precision"],
        "finite": all(torch.isfinite(value).all().item() for value in tensors.values()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _log_artifact(path: Path, *, project: str, entity: str | None) -> None:
    """Log one deliberately requested, modest-size stats artifact to W&B."""
    import wandb

    run = wandb.init(project=project, entity=entity, job_type="whitening-stats")
    artifact = wandb.Artifact(path.stem, type="whitening-stats")
    artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()


def parse_args() -> argparse.Namespace:
    """Parse fit/inspect options without requiring a model download."""
    parser = argparse.ArgumentParser(description="Fit or inspect encoder-bound whitening stats.")
    parser.add_argument("--inspect", metavar="PATH")
    parser.add_argument(
        "--data", choices=("ssv2", "ssv2_tiny", "ego4d", "ego4d_tiny"), default="ssv2_tiny"
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument(
        "--encoder",
        choices=("vjepa2_vitl16", "dinov3_vitb16", "siglip2_vitb16"),
        default="vjepa2_vitl16",
    )
    parser.add_argument("--encoder-revision", "--revision", dest="encoder_revision")
    parser.add_argument(
        "--encoder-precision", "--precision", dest="encoder_precision", choices=("fp32", "bf16")
    )
    parser.add_argument(
        "--encoder-frame-microbatch", "--frame-microbatch", dest="frame_microbatch", type=int
    )
    parser.add_argument("--encoder-attention-implementation", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=12_800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "--out", dest="output")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default=None)
    parser.add_argument("--wandb-artifact", action="store_true")
    parser.add_argument("--wandb-project", default="hjepa-vwm")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main() -> None:
    """Fit one immutable envelope or inspect an existing one."""
    args = parse_args()
    _require_torch()
    if args.inspect:
        inspect_stats(args.inspect)
        return
    cfg = Config()
    cfg.data.dataset = args.data
    cfg.seed = args.seed
    cfg.encoder.alias = args.encoder
    cfg.encoder.revision = args.encoder_revision
    if args.encoder_precision:
        cfg.encoder.precision = args.encoder_precision
    if args.frame_microbatch is not None:
        cfg.encoder.frame_microbatch = args.frame_microbatch
    if args.encoder_attention_implementation:
        cfg.encoder.attention_implementation = args.encoder_attention_implementation
    if args.hf_cache_dir:
        cfg.hf_cache_dir = args.hf_cache_dir
    from provenance import atomic_torch_save
    from train import set_seed

    set_seed(cfg.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(
        args.output or f"logs/whiten/{args.encoder}_{args.data}_{args.split}_seed{args.seed}.pt"
    )
    envelope = compute_whitening_stats(
        cfg, args.split, args.max_clips, device, batch_size=args.batch_size
    )
    atomic_torch_save(envelope, output)
    inspect_stats(output)
    if args.wandb_artifact:
        _log_artifact(output, project=args.wandb_project, entity=args.wandb_entity)


if __name__ == "__main__":
    main()
