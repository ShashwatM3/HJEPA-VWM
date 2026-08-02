# Next steps — investigation_021

1. Push a clean launch commit that contains implementation commit
   `0c1d343ce19258d5d575ea17a1654a46b32abf85` and this investigation record.
2. Complete `AGENT_FILES/SETUPS/NEW_POD.md` through step 5 (`wandb login`) on a one-GPU pod.
3. Execute [`fixed_residual_coordinates_fc_only/GUIDE.md`](fixed_residual_coordinates_fc_only/GUIDE.md)
   end to end without changing the scientific recipe.
4. Record the new W&B ID/URL and clean launch commit in the run `DESCRIPTION.md` and
   `OBSERVATIONS.md` only after process, GPU, W&B, provenance, and step-0 proof all pass.
5. Monitor the immutable-state tripwires and the Reading-Cycle-A prediction metrics through the
   full 15,000 steps. Do not stop merely because early ratios are poor; this is the registered
   full-scale test.
6. After completion, pull unsampled W&B history, write `METRIC_READOUT.md` and `ANALYSIS.md`, update
   both triads, and reconcile `KANBAN/PHASE_1/README.md` with the immutable W&B ID.
7. If fixed coordinates still lose to both baselines, open a predictor/objective investigation.
   Do not return to encoder, bottleneck-geometry, SIGReg, or latent-shape sweeps first.
