# Observations — V-JEPA2 latent-shape sweep

The queue uses clean training commit `771cbba077d9f846bdf7a7dd48e12dbf29d54b49`. Its first process,
W&B run [`kiti1gpc`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/kiti1gpc), was stopped at
step 1,000 after a W&B audit proved it exactly duplicated Run 66
[`guiduvjp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/guiduvjp). The model, training,
data, encoder, seed, dataset fingerprint, and trainable initialization hash match. `kiti1gpc` is
excluded, `guiduvjp` supplies the center evidence, and the replacement queue contains only the
eight untried shapes. An earlier separate preflight controller was also stopped before it created
any science run and is excluded from sweep evidence.

The duplicate reached 100% A100 utilization and step 1,000 without skipped gradients or an
instability warning before it was stopped. Those records are operational evidence only and are not
used in the sweep verdict.

The corrected controller has SHA-256
`e34017a1b5d0b6d3f555e6345a5aa2853748b4b95f7f50d09a40ff2eb3fa7c9c` and contains exactly the
eight W&B-audited untried shapes. It is active in tmux `inv017_vjepa2_sweep`; PID 461520 is the
first unique arm, `16×512`.

Registered prior: Run 66 makes the fresh `32×256` center likely to retain healthy geometry and
honest within-source code dependence. The earlier `M=512` versus `M=1024` null result makes a large
capacity-only reconstruction gain unlikely. The equal-capacity diagonal is expected to be more
informative than raw parameter count.

The remaining eight W&B IDs, late-window measurements, and the lane verdict will be appended as
the sequential queue advances.

## 2026-07-26 — queue is terminal, not running

W&B `ihiuptdp` is the unique `N_c=16`, `D_c=512` arm. It is `crashed` at step 7,100, and no later
V-JEPA2 sweep ID exists. The queue did not advance.

All available stability tripwires remain zero through the last row. Late six-diagnostic medians
are correct/rolled reconstruction `0.38514/0.42271`, exact-chunk gap `0.03759`, std `0.71049`,
within-source cosine `0.55934`, effective rank `64.81`, and centered slot rank `14.64/15`.
This is **INVALID AS A COMPLETE SWEEP ARM; STABLE, DECODABLE PARTIAL TRAJECTORY**. It cannot
select the `16×512` cell or support a full-schedule comparison with center `guiduvjp`.

The seven other non-center shapes remain unlaunched. A run-local record is in
[`ihiuptdp_vjepa2_n16_d512/`](ihiuptdp_vjepa2_n16_d512/).
