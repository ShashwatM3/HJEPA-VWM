# Oral-exam quiz bank

Answer aloud without notes. State the truth layer—implemented, shipped default, experiment recipe,
or planned—whenever a question could mix them. The answer key is separate.

## A — Whole system

1. **A1.** Describe current Phase 1 in one minute from video file to optimizer transition.
2. **A2.** Name the five modules in the runtime tuple, in order.
3. **A3.** Which modules are trainable, EMA-mutated, frozen, and buffer-only?
4. **A4.** What are the three representation levels and their default shapes?
5. **A5.** Why is there no target encoder?
6. **A6.** What is the current decoder's output, and what is it explicitly not?
7. **A7.** List six major planned components absent from current code.
8. **A8.** What is the central scientific distinction between representation health and prediction?
9. **A9.** What four truth labels must every architectural claim use?
10. **A10.** Give the quiz-quality one-sentence architecture answer.

## B — Data and time

1. **B1.** Write the context and target index equations for arbitrary `T,s,k`.
2. **B2.** List both index arrays at shipped defaults.
3. **B3.** How many exact frames overlap at `k=4,12,14,16`?
4. **B4.** What is the span-disjoint threshold and why does the guide say 16+?
5. **B5.** Derive `max_start`.
6. **B6.** How is the per-item RNG seed constructed?
7. **B7.** State the full transform sequence, including operations that are absent.
8. **B8.** Why are context and target transformed together?
9. **B9.** What happens for short videos?
10. **B10.** At 12-FPS EGO, what real time do stride two and `k=4` represent?

## C — Encoders

1. **C1.** State `N_e,D_e,T_e,H_e,W_e` for all three encoders.
2. **C2.** Why does V-JEPA have four temporal planes?
3. **C3.** What exact tokens are removed from DINOv3 per frame?
4. **C4.** How does a frame encoder turn `(B,T,...)` into downstream video tokens?
5. **C5.** State the three normalization rules.
6. **C6.** State the three repositories and immutable revisions.
7. **C7.** State the three parameter counts.
8. **C8.** What fields conceptually enter the encoder fingerprint?
9. **C9.** What does BF16 encoder mode require, and what does FP32 mode change?
10. **C10.** How much BF16 detailed-feature storage does one video and a batch of 64 require?

## D — Bottleneck

1. **D1.** Trace every bottleneck shape from `e` to `c`.
2. **D2.** What exactly do the two ConvNeXt blocks mix?
3. **D3.** Enumerate one ConvNeXt block and derive `8M²+57M`.
4. **D4.** What are the learned position/query shapes and initializations?
5. **D5.** Enumerate one latent block.
6. **D6.** How does sharpened cosine attention compute and cap logits?
7. **D7.** Which bridges are zero initialized and why?
8. **D8.** What is the exact bottleneck output at initialization conceptually?
9. **D9.** Derive the full bottleneck parameter formula.
10. **D10.** State bottleneck counts for V-JEPA and frame encoders at `M=256,512,1024`.

## E — EMA and flow

1. **E1.** Write the EMA update and exact state-transition order.
2. **E2.** Write the momentum schedule and state values at 0, 15k, and 105k.
3. **E3.** Write ordinary rectified-flow source, interpolation, velocity, and loss.
4. **E4.** List every learned `F_c` input identity tensor and its shape/init.
5. **E5.** Explain per-example condition dropout.
6. **E6.** Derive the timestep embedding.
7. **E7.** Enumerate one AdaLN-Zero block and its initialization behavior.
8. **E8.** Derive the `F_c` parameter formula and default count.
9. **E9.** Write residual-prediction target/noise/endpoint equations.
10. **E10.** Why is the one-step endpoint not an inference sampler?

## F — Decoder, mean, whitening

1. **F1.** Trace all decoder shapes and sublayers.
2. **F2.** Derive the 85/85/86 positional split.
3. **F3.** Why can fixed position guide where without supplying what?
4. **F4.** Derive the decoder parameter formula and both default counts.
5. **F5.** Write cosine reconstruction and legacy relative MSE.
6. **F6.** Compare gradient routes for present and predicted reconstruction.
7. **F7.** What is reconstruction blindness, and which metrics expose it?
8. **F8.** State the mean tracker's update timing, equation, shape, dtype, and storage.
9. **F9.** Write ZCA whiten/unwhiten equations and buffer sizes.
10. **F10.** How are offline whitening statistics fitted and identity-bound?

## G — Losses and optimization

1. **G1.** Write the total full-prediction objective with all gates/ramps.
2. **G2.** State variance-floor axes, estimator, and limitation.
3. **G3.** State covariance axes, denominator, and limitation.
4. **G4.** State slot-diversity centering, normalization, and denominator.
5. **G5.** Explain SIGReg's row population, projections, statistic, precision, and RNG.
6. **G6.** Give the complete loss-to-module gradient matrix from memory.
7. **G7.** Which parameters receive weight decay and which are excluded?
8. **G8.** Write AGC's formula and module factors.
9. **G9.** Distinguish `grad_norm` from `grad_global_norm_postclip`.
10. **G10.** Exactly what happens when a step exceeds the skip threshold?

## H — Schedule and training mechanics

1. **H1.** Write the LR multiplier piecewise.
2. **H2.** Give LR scales at steps 0,1499,1500,7500,14999,15000.
3. **H3.** Explain why `--steps=2000` and `--steps=20000` are surprising.
4. **H4.** Write the linear reconstruction/SIG ramp and values at 0,1000,2000.
5. **H5.** List the exact training-step order from seeding through EMA.
6. **H6.** Which stochastic draws use the main step seed and which do not?
7. **H7.** What does present-only mode skip and require?
8. **H8.** State log, diagnostic, checkpoint cadences and 15k totals.
9. **H9.** Explain fixed diagnostic batch and RNG isolation.
10. **H10.** Calculate example/frame presentation counts for the default run.

## I — Diagnostics

1. **I1.** Define all three variance-health metrics exactly.
2. **I2.** Define cross-video cosine and its EGO caveat.
3. **I3.** Derive effective rank and its maximum.
4. **I4.** Compare raw and centered slot ranks.
5. **I5.** Define normalized per-head attention entropy.
6. **I6.** Derive copy velocity/loss in ordinary and residual modes.
7. **I7.** Define batch-mean baseline and both ratio gates.
8. **I8.** Name all five reconstruction readouts and interpret video gap.
9. **I9.** Walk the eight-question full-prediction reading cycle.
10. **I10.** Assign a verdict to rank 61/std1/cos0.16/copy ratio1.06 with stable gradients.

## J — Checkpoint, provenance, resume

1. **J1.** Name every checkpoint payload category.
2. **J2.** What does `phase1_step2500.pt` mean?
3. **J3.** Derive sampler epoch/offset and compute them at SSv2-tiny step 7500.
4. **J4.** What must validate before any live state mutation?
5. **J5.** Which RNG states are saved, and why still seed per step?
6. **J6.** What exactly enters dataset identity?
7. **J7.** What is common run identity, and why separate encoder identity?
8. **J8.** What may dataset transfer change, and what remains guarded?
9. **J9.** How do whitening and mean state survive resume?
10. **J10.** Why is loading weights alone not exact resume?

## K — Datasets, probes, MLOps

1. **K1.** State full/tiny SSv2 counts and 15k epoch arithmetic.
2. **K2.** Walk EGO selection filters, diversity, split, and batching.
3. **K3.** State the exact EGO chunk encoding recipe.
4. **K4.** What makes the chunker idempotent and failure-safe?
5. **K5.** What are exact EGO-tiny contracts versus full-EGO estimates?
6. **K6.** Compare rank probe, drift probe, whitening stats, run history, and log parser.
7. **K7.** What identities make a shared feature cache safe?
8. **K8.** State the five fresh-pod bootstrap stages.
9. **K9.** What eight things prove a remote launch is alive/correct?
10. **K10.** What proves completion, and what does W&B not store by default?

## L — History, future, traps

1. **L1.** Why did the system add AGC and the skip guard?
2. **L2.** Why did orthogonal queries/sharpened cosine replace tiny random queries?
3. **L3.** What did the rank-13 arc teach?
4. **L4.** What did runs 052 and 053 jointly prove?
5. **L5.** What did SIGReg teach about geometry versus prediction?
6. **L6.** Why was `M=512` selected over 1024?
7. **L7.** What did the `N_c=32/64/128` sweep show?
8. **L8.** Describe planned `F_e` teacher forcing and gradient boundary.
9. **L9.** What must multi-horizon and inference add?
10. **L10.** Name ten stale-prose/truth-layer traps in the repository.
