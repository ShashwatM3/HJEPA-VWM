# fixed-position-decoder - Run C: full residual recipe with fixed-position D

**Investigation:** [investigation_011](../DESCRIPTION.md)
**Position:** Run C, after Run A (`new_recon_loss`) and Run B (`original_recon_loss + no-pred`)
**Status:** READY TO LAUNCH
**Code requirement:** commit `0a9ff46` or later
**W&B group:** `inv011_fixed_position_decoder`

## Question

Run A in this investigation changed the reconstruction objective to cosine distance while keeping the
full residual/SIGReg recipe. It produced healthier reconstruction numbers but still tied the
zero-residual/copy baseline. This run keeps that same training recipe and changes the decoder
architecture: `D` no longer owns learned per-output-token queries.

There is no CLI flag for this feature. The fixed-position decoder is now the default `models.Decoder`
implementation, so the run must start from commit `0a9ff46` or later and must not resume an old
learned-query decoder checkpoint.

## Config Delta

Compared with Run A (`new_recon_loss`):

- same full residual recipe;
- same cosine reconstruction objective;
- same decoder width/depth knobs: `decoder_dim=512`, `decoder_blocks=4`;
- changed only by code: learned output queries replaced by fixed tubelet position codes.

The intended decoder rule is:

```text
position tells D where to write;
c tells D what to write.
```

## Command Shape

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 \
  --steps 15000 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 5.0 \
  --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.05 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --predict-residual \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv011_fixed_position_decoder \
  --log-every 50 \
  --diag-every 500
```

See [GUIDE.md](GUIDE.md) for smoke checks, launch steps, and tripwires.

## Main Comparison

Compare primarily against:

- Run A `new_recon_loss` (`1u69hpfm`): same recipe, learned-query decoder, cosine recon.
- investigation_010 `soft-universe-37` (`2vbo6pbm`): previous best residual/SIGReg run.

## Main Readouts

- `coarse_vs_copy_ratio`: does the fixed-position decoder help prediction beat copy?
- `L_recon_present`: does reconstruction become more honest, even if numerically harder?
- `L_recon_cplus` and `L_recon_chat`: does the true-future vs predicted-future gap widen?
- `c_effective_rank`, `c_plus_effective_rank`, `c_cross_video_cosine`, `c_std_mean`: representation
  health must remain intact.
- `grad_skipped`, `grad_norm`, `agc_D_*`: decoder change must not destabilize training.

## Interpretation

- **Win:** copy ratio drops below Run A while rank/cosine/stability stay healthy, and
  `L_recon_chat - L_recon_cplus` becomes meaningfully larger when `c_hat` is poor.
- **Useful neutral:** copy ratio remains near 1, but reconstruction readouts become more diagnostic;
  this confirms the old decoder was masking prediction quality without solving dynamics.
- **Fail:** reconstruction returns to the same floor with tiny `chat-cplus` gap, or representation
  health degrades without exposing new predictive signal.
