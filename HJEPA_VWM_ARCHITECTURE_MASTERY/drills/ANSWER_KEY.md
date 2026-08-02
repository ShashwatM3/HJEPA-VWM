# Oral-exam answer key

Use only after committing to an answer. These are compressed reference answers; chapters provide
derivations.

## A

- **A1.** Deterministic paired windows → shared transforms → same frozen `E` → online `B(e_t)=c_t`,
  EMA `B(e⁺)=c⁺` → flow interpolation/noise/time → conditioned six-block `F_c` velocity MSE plus
  representation/recon terms → backward, AGC, global clip, optional skip, AdamW, EMA update.
- **A2.** `(encoder,bottleneck,target_bottleneck,coarse_flow,decoder)`.
- **A3.** Trainable: `B,F_c,D`; EMA-mutated/frozen-gradient: `B_EMA`; frozen: `E`; buffers:
  mean tracker/whitener/fixed positions.
- **A4.** Raw `(B,8,3,256,256)`; detailed V-JEPA `(B,1024,1024)` or frame
  `(B,2048,768)`; abstract `(B,32,256)`.
- **A5.** `E` is frozen and shared, so a copied EMA encoder would be identical; the moving target is
  at `B`.
- **A6.** Detailed encoder features `(B,N_e,D_e)`, not pixels/RGB/VAE latents.
- **A7.** `F_e`, pixel/VAE generator, horizon embedding, multi-horizon loader, numerical sampler,
  rollout integration (also no trainable/EMA encoder).
- **A8.** Broad/video-specific `c` can still be static or unpredictably dynamic; prediction must beat
  copy and batch mean independently.
- **A9.** Implemented, shipped default, experiment recipe, planned.
- **A10.** See Chapter 00's one-sentence answer; it must name frozen `E`, online/EMA `B`, coarse flow,
  feature `D`, and strict experiment state.

## B

- **B1.** `I_c(i)=a+is`; `I_t(i)=a+k+is`, `i=0..T-1`.
- **B2.** Context `0,2,…,14`; target `4,6,…,18`, plus start.
- **B3.** 6,2,1,0.
- **B4.** Span-disjoint `k>14`; on the normal even ladder, first is 16.
- **B5.** `max_start=frames-1-k-(T-1)s`.
- **B6.** SHA-256 of `seed:epoch:sample_id:transform_version`, first eight bytes unsigned big-endian.
- **B7.** Decode needed union, uint8→float/255, resize short side 256 bilinear, train random/val center
  crop, train brightness/contrast/saturation independently 0.6–1.4, clamp. No flip/rotation/hue or
  encoder norm here.
- **B8.** Same crop/color draw preserves paired temporal semantics.
- **B9.** Start zero; indices clamp to last frame, repeating terminal content.
- **B10.** Stride two = 1/6 s; `k=4` = 1/3 s.

## C

- **C1.** V-JEPA `4×16×16`, `1024×1024`; Sig/DINO `8×16×16`, `2048×768`.
- **C2.** Temporal tubelet support/stride two.
- **C3.** One CLS and four register tokens; retain 256 patches/frame.
- **C4.** Flatten `B*T`, microbatch frames, retain 256 patches, restore `(B,T,256,D)`, flatten
  time/patch.
- **C5.** V-JEPA/DINO ImageNet; SigLIP mean/std 0.5.
- **C6.** V-JEPA `facebook/vjepa2-vitl-fpc64-256` @ `b3c...26f`; Sig
  `google/siglip2-base-patch16-256` @ `3f9...c1ab`; DINO
  `facebook/dinov3-vitb16-pretrain-lvd1689m` @ `593...96bc`.
- **C7.** 325,971,328; 85,843,200; 85,660,416.
- **C8.** Repo/revision, normalization, geometry/order, width, parameter count, precision, attention
  implementation, microbatch, cache/runtime fields.
- **C9.** BF16 requires CUDA; FP32 disables outer encoder autocast.
- **C10.** V-JEPA 2 MiB/video,128 MiB/batch; frame 3 MiB/video,192 MiB/batch.

## D

- **D1.** `(B,N_e,D_e)→in_proj(B,N_e,M)→(B*T_e,M,H_e,W_e)→two ConvNeXt→(B,N_e,M)+pos→KV;
  queries(B,N_c,M)→three latent blocks→optional M→D_c→LN→(B,N_c,D_c)`.
- **D2.** Spatial neighbors within each temporal plane; no temporal convolution.
- **D3.** DW7×7+bias `50M`, LN `2M`, two linears `8M²+5M`; total `8M²+57M`.
- **D4.** Pos `(1,N_e,M)`, trunc-normal std .5; queries `(N_c,M)`, orthogonal unit rows.
- **D5.** Pre-norm sharpened cosine cross-attn residual; pre-norm self-attn residual; pre-norm
  `M→4M→M` residual.
- **D6.** Normalize q/k, cosine×`min(exp(log_scale),100)`; initial temperature .07.
- **D7.** Cross output, self-attn output, final MLP projection; stable identity/query initialization
  and later learned opening.
- **D8.** Input-independent learned query template after optional projection/final normalization.
- **D9.** Formula in parameter appendix: input + `2(8M²+57M)` + KV + position + queries +
  `3(16M²+19M+1)` + optional projection + final norm.
- **D10.** V-JEPA 4,837,123 / 18,324,739 / 70,727,427; frame 5,033,731 / 18,717,955 /
  71,513,859.

## E

- **E1.** `θema←mθema+(1-m)θ`; backward→AGC→global clip/check→optimizer→EMA.
- **E2.** Cosine .996→.9999 over105k; at 0=.996,15k=.996193111,105k=.9999.
- **E3.** `ε~N`, `τ~U`, `z=(1-τ)ε+τc⁺`, `u=c⁺-ε`, mean squared velocity error.
- **E4.** Null `(N_c,D_c)` zero; slot pos `(N_c,D_c)` normal .02; z/condition type
  `(1,1,D_c)` zero.
- **E5.** Each training example replaces entire condition with learned null at p=.1; diagnostics force
  false.
- **E6.** Sin/cos half channels at `exp(-log10000*i/half)`, then `D→4D→D` SiLU MLP.
- **E7.** Nonaffine LN + modulated MHA gated residual; nonaffine LN + modulated MLP gated residual;
  `D→6D` modulation zero, so block identity.
- **E8.** `2ND+2D +(8D²+5D)+6(18D²+15D)+2D = 7,643,904` at 32/256.
- **E9.** `Δ=Bema(e⁺)-Bema(e_t)`, `σ=max(std(Δ),1e-6)`, `ε=σN`,
  `z=(1-τ)ε+τΔ`, `u=Δ-ε`, `Δhat=z+(1-τ)û`, `chat=c_online+Δhat`.
- **E10.** It is one algebraic estimate at sampled `τ`; no integration scheme/steps/guidance/rollout.

## F

- **F1.** `c→D_c→D_d` memory; fixed `(N_e,D_d)` query reads memory once; two position-query
  cross-attn+MLP residual blocks; LN and `D_d→D_e`.
- **F2.** `floor(256/3)=85`; remaining171 split floor85 and86.
- **F3.** Position enters queries/weights, never as residual value content; values come from `c`.
- **F4.** KV + initial MHA + `2(12D²+13D)` + final LN + output. 2,172,160 V-JEPA;
  2,106,368 frame.
- **F5.** Mean `1-cos(normalize ê,normalize detached e)`; or MSE/biased target variance.
- **F6.** Present→`B,D`; predicted→`B,F_c,D`.
- **F7.** Decoder predicts generic template regardless of video; expose with shuffled `c`, video gap,
  and chat-vs-cplus against copy ratio.
- **F8.** First update copies batch positional mean before loss; later `.99 mean+.01 batch`; FP32
  `(N_e,D_e)`; 4 MiB V-JEPA/6 MiB frame.
- **F9.** `(e-μ)U(λ+eps)^-1/2Uᵀ`, inverse plus μ; ~8.004 MiB at1024,4.503 at768.
- **F10.** deterministic training context prefix,12,800 clips, FP64 sum/outer, biased covariance,
  `eigh`; envelope binds encoder/data/seed/count/solver/payload.

## G

- **G1.** Flow + weighted var + gated cov/slot + ramped SIG + ramped present/pred recon.
- **G2.** Flatten each video to `N_cD_c`; population std across batch; hinge at1. Cannot prevent
  correlations.
- **G3.** Pool batch×slots, center features, sample covariance `/N-1`, squared off-diagonal sum `/D`.
  Cannot prove slots/video specificity.
- **G4.** Per-video slot centering, L2 normalize, squared cosine off-diagonals divided by
  `B*N_c*(N_c-1)`.
- **G5.** Pool batch×slots, cap512, center,128 random unit directions, BHEP statistic beta1, FP32/no
  autocast, dedicated generator.
- **G6.** Flow→B/Fc; var/cov/slot/SIG→B; present recon→B/D; predicted→B/Fc/D; never E/B_EMA.
- **G7.** Only genuine 2-D+ Linear/Conv matrices; exclude ndim<2, named geometry, marked zero-init
  modules.
- **G8.** Bound `λ(||w||+.001)`, rescale if grad exceeds; B .2,Fc .1,D .2.
- **G9.** `grad_norm` after AGC before global rescale; postclip metric after 0.5 clipping.
- **G10.** Zero grads, no AdamW, no EMA; metric says skipped.

## H

- **H1.** Before1500 `(step+1)/1500`; then half-cosine using `(step-1500)/13500`, clamped.
- **H2.** .0006667,1,1,.586824,~1.3539e-8,0.
- **H3.** 2k is only a prefix of 15k schedule; after15k a 20k run has zero LR.
- **H4.** `min(1,max(0,step/2000))`: 0,.5,1.
- **H5.** Seed→transfer→zero grads→autocast→forward/whiten→flow→mean update→all regularizers→assemble
  →backward→AGC→global clip/check→Adam→EMA.
- **H6.** Flow noise/time/dropout use step stream; SIG dedicated same-formula generator; data uses
  item hash; diagnostics fork fixed seed.
- **H7.** Skips target transfer/encode, B_EMA target, flow; requires positive present recon; disables
  temporal residual/pred recon.
- **H8.** 50/500/2500 →300/30/6.
- **H9.** Same >1 up-to16 validation examples; fixed forked seed and restored RNG; condition dropout
  forced off.
- **H10.** 960k examples, 7.68M context frames and another7.68M target frames in full mode.

## I

- **I1.** Flatten per video; population std across batch; report mean/median and fraction below
  .1×median.
- **I2.** Mean off-diagonal cosine of flattened videos; EGO batch may be same-source chunks.
- **I3.** Pool B×slots over feature width, sample covariance, entropy of normalized eigenvalues,
  exponentiate; max256.
- **I4.** Per-video slot Gram entropy rank; centered version removes shared mean and is cleaner.
- **I5.** `-Σa loga/logN_e` per head/slot; report mean and min.
- **I6.** Ordinary copy velocity `c_t-ε`; residual `-ε`; both loss `mean((c_t-c⁺)²)=meanΔ²`.
- **I7.** Use mean target across fixed batch; model/mean ratio≤.50; model/copy≤.70.
- **I8.** Present,cplus,chat,shuffled,video gap=shuffled-present; positive gap shows latent identity
  matters.
- **I9.** Stability, collapse, rank, static-c, copy, batch mean, recon honesty, verdict.
- **I10.** Healthy representation, no predictor.

## J

- **J1.** Progress; B/BEMA/Fc/D; optimizer; config; encoder; dataset; init hash; W&B ID; provenance;
  sampler; all RNG; optional mean/whitener.
- **J2.** Updates0–2499 complete;2500 next.
- **J3.** `epoch=step//floor(N/B)`, offset=`(step%floor(N/B))*B`; at7500/62: epoch120,
  offset3840.
- **J4.** Schema, encoder/data/provenance/sampler, every module/buffer/optimizer key/shape/hash.
- **J5.** Python,NumPy,Torch CPU/all CUDA; exact non-step consumers and complete state, while per-step
  reseeding isolates continuation from diagnostics.
- **J6.** Sorted relative paths, resolved sizes, decoded frame counts and aggregates, manifest
  hashes, transform version, input geometry, EGO completeness/source leakage.
- **J7.** Hash of all encoder-independent controls; separate encoder permits controlled substrate
  arms while guarding common factors.
- **J8.** Only data fingerprint/order/config; everything non-data remains equal.
- **J9.** State dicts plus hashes embedded in checkpoint; whitener checkpoint state overrides
  external stats on resume.
- **J10.** It omits EMA/optimizer/buffers/sampler/RNG/identity/W&B lineage.

## K

- **K1.** Full168,913/24,777,2639 batches,~5.684 epochs; tiny4002/348,62 batches,241+58.
- **K2.** Filter tier/group/stereo/FPS/duration; seeded primary-scenario cap with relaxed top-up;
  source-hour train/val; four LPT hour-balanced source batches.
- **K3.** Nonoverlap4s,12fps,short256,H.264 libx264 veryfast CRF27,yuv420p,GOP12,no audio,
  faststart, one ffmpeg thread,≤4 workers, skip redactions.
- **K4.** Skip existing nonempty, process present raw UIDs, `.part` atomic rename, delete partial/fail
  batch on any encode failure, cumulative rescan manifest.
- **K5.** Tiny exactly4000/350≤10/source; full roughly165–175k/15–20k but manifest exact.
- **K6.** Frozen rank; temporal drift; whitening fit; unsampled W&B; console-log parsing.
- **K7.** Encoder/data/manifest/offset/dtype/shape/finiteness/payload identities.
- **K8.** System packages; persistent dirs/env; exact repo; pinned Python deps; W&B auth.
- **K9.** tmux, PID/command, GPU, advancing log, W&B, provenance, destinations, tripwires.
- **K10.** Exit0, final step/W&B, checkpoint/hash, identities/provenance, all arms, verdict. W&B does
  not store checkpoint bytes by default.

## L

- **L1.** Late gradient explosions after healthy-looking training; stabilize moderate spikes and skip
  only catastrophic state transitions.
- **L2.** Tiny queries produced tiny logits/uniform same-slot reads; orthogonal geometry and
  temperature create distinct, competitive starts.
- **L3.** Std alone permits correlated low-rank codes; geometry/content/prediction need separate
  interventions.
- **L4.** 052 proved low recon can be template collapse;053 proved residual target restores honesty
  but does not restore geometry.
- **L5.** Isotropic rank can rise while EMA/predictor lag or content is static; rank≠forecastability.
- **L6.** 1024 cost 3.86× B for sub-threshold recon/gap gains;512 chosen.
- **L7.** Larger slot counts gave tiny recon gain and worse recorded geometry; no 256 arm.
- **L8.** Train fine flow first with stopped true cplus, then stopped predicted chat; fine loss must
  not update Fc in isolated stage.
- **L9.** Per-example horizon identity/embedding and per-horizon metrics; ODE solver/steps/seeds/
  guidance/rollout boundary.
- **L10.** Examples: original shapes, target encoder, decoder generator, one epoch, `f_c_dim`,
  105k LR comment, unwired pred-recon comment, tuple docstring, exact EGO count, checkpoint default,
  stale W&B snapshot, EGO “cross-video.”
