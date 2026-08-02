# Next steps — run 070

1. Keep this checkpoint as the completed whitened DINO geometry reference; do not relaunch the
   100-step smoke or overwrite W&B `qqozribu`.
2. Compare its geometry with the raw/unwhitened DINO center in
   [`../run_071_unwhitened_dinov3_m512_cov_var/`](../run_071_unwhitened_dinov3_m512_cov_var/),
   but do not rank their raw reconstruction losses because the targets differ.
3. Use source-diverse diagnostics before promoting the recorded-batch specificity claim to a
   global one.
4. Treat DINO full-prediction certification as separate, unrun work.
