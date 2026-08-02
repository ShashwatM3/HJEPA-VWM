# V-JEPA2 latent-shape sweep

## Status

OPEN — no queue is active. Duplicate center
[`kiti1gpc`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/kiti1gpc) was stopped and excluded.
The first unique arm, `16×512`
([`ihiuptdp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ihiuptdp)), crashed at step
7,100; the seven other non-center shapes are unlaunched.

## Question

Under the raw-feature covariance-plus-variance recipe established by Run 66, does V-JEPA2 benefit
more from external slot multiplicity, per-slot width, or total nominal channel size?

## Baseline and delta

Run 66 ([W&B `guiduvjp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/guiduvjp))
validated the center recipe at `N_c=32`, `D_c=256`, `M=512`. It was intentionally stopped after
step 10,950 while stable; late medians showed correct-versus-shuffled gap about `0.098`, mean std
`0.872`, pair cosine `0.441`, and effective rank about `115`.

W&B confirms Run 66 has the exact proposed center model, training, data, encoder, seed, dataset
fingerprint, and trainable initialization hash. It supplies the `32×256` control without another
paid run. The eight neighboring arms keep its recipe fixed and change only `N_c` and `D_c`.

## Encoder contract

`vjepa2_vitl16@b3c1679b7c34d3255ef3547f27c7b226aefab26f` returns four temporally aware
16×16 tubelet grids, flattened to `(B,1024,1024)`. Inference is frozen bf16 with frame microbatch 8.

Design: [`PLAN.md`](PLAN.md). Launch: [`GUIDE.md`](GUIDE.md).
