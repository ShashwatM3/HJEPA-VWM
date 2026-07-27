# Plan — Run 67, SigLIP2-B standard-ViT companion

1. Keep the remote on the same clean published commit used by Run 66 and verify the pinned SigLIP
   adapter revision `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`.
2. Preserve every Run 66 scientific/runtime argument except the encoder alias and immutable
   revision. Do not pass whitening flags or reuse any V-JEPA whitening payload.
3. Pass the real SigLIP CUDA adapter smoke, exact Stage 0, and batch-64 resource preflight before
   paid work. The resource report must have finite loss/gradients, positive throughput, and
   non-null CUDA peak memory.
4. Use a separate tmux session, log, checkpoint root, provenance JSON, and W&B display name.
5. On the one A100, overlap paid training with Run 66 only if both preflight peaks plus a safety
   margin fit in 80 GB and both processes remain stable. Otherwise queue this arm sequentially;
   concurrent execution is not part of the scientific intervention.
6. Monitor through step 15,000, verify W&B finished state and final artifacts, then read the two
   encoder arms interleaved with the present-only Reading Cycle B.

## Abort conditions

- encoder alias/revision or feature layout differs from the pinned SigLIP contract;
- batch, bottleneck, decoder, objective, geometry weights, or schedule differs from Run 66;
- whitening/prediction activates, or `L_flow`/`L_recon_pred` becomes nonzero;
- W&B required initialization, provenance, CUDA resource, or output isolation fails;
- nonfinite loss, repeated skipped updates, OOM, or persistent instability occurs.
