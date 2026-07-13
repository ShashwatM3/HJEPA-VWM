# Guide: From V-JEPA-only to Encoder-Pluggable Experiments

> This guide starts from the repository's actual state on 2026-07-13: the data selector
> supports SSv2/EGO4D, but the model, preprocessing, whitening, checkpoints, and probes are
> V-JEPA-specific. Follow the stages in order. Do not run a later command merely because
> it is visible below.

The design authority is
[`ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md`](ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md).
The encoder research is in [`DINOv3_ViT-B-16.md`](DINOv3_ViT-B-16.md) and
[`SigLIP_2_ViT-B-16.md`](SigLIP_2_ViT-B-16.md).

## Stage 0 — Know the finish line

You are done only when all of these are true:

- `--encoder vjepa2_vitl16|dinov3_vitb16|siglip2_vitb16` works through one interface;
- the dataloader emits raw `[0,1]` clips and each adapter normalizes exactly once;
- B, D, whitening, mean tracking, checkpoints, resume, rank, and drift use one resolved
  feature specification rather than V-JEPA constants;
- present-only and current full Phase 1 both pass on `ssv2_tiny` and `ego4d_tiny`;
- DINO and SigLIP use the same data order and byte-identical trainable initialization;
- wrong stats/checkpoints/caches fail before training;
- W&B cannot silently disable itself for a paid run;
- a clean V-JEPA regression and the two real frame-adapter preflights pass;
- the two-arm command differs only in the permitted encoder/run/artifact fields.

No Python code has been changed merely by writing this guide.

## Stage 1 — `[YOU, NOW]` Finish and freeze EGO4D verification

Do this before asking an agent to modify data preprocessing. The EGO4D corpus and tiny
subset are complete; preserve them and finish the current smoke verification.

1. Reattach to the current smoke session:

   ```bash
   tmux attach -t ego4d_smoke
   ```

2. Do not rerun Stage 4 or Stage 5. Their final checks already established:

   ```text
   Stage 4 manifest/filesystem/leakage checks OK
   raw video_540ss mp4 files left: 0
   Stage 5 checks OK
   ```

3. Finish [`../ego4d/GUIDE.md`](../ego4d/GUIDE.md) Stage 6 before the encoder refactor
   changes the dataloader's normalization boundary. If Paste 3 raises the historical
   `B_EMA did not update` assertion, pull commit `0af0dc7` and rerun Paste 3. That failure
   does not alter the dataset or create a checkpoint. Continue only after:

   ```text
   Stage 6 switchability smoke passed
   ```

4. Stop before the EGO4D guide's Stage 7 real run. Do not generate EGO whitening stats
   yet. The encoder experiment first changes only the encoder on SSv2.
5. Record the final immutable identities:

   ```bash
   cd /workspace/hierarchal-jepa-flow-world-model
   sha256sum /workspace/ego4d_raw/manifests/selection_manifest.json
   sha256sum /workspace/ego4d_raw/video_540ss_manifest.csv
   sha256sum /workspace/data/ego4d/chunk_manifest.json
   sha256sum /workspace/data/ego4d_tiny/manifest.json
   find /workspace/data/ego4d/train -maxdepth 1 -name '*.mp4' | wc -l
   find /workspace/data/ego4d/validation -maxdepth 1 -name '*.mp4' | wc -l
   ```

Save the terminal output in your experiment notes. Do not manually edit any manifest.

## Stage 2 — `[YOU, ONCE]` Give the machines legitimate access

### 2A. Hugging Face

1. Log in at [Hugging Face](https://huggingface.co/).
2. Open the
   [DINOv3-B model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
3. Read and accept the DINOv3 license/access terms. The access request must be made in the
   browser and belongs to your individual account.
4. Open [token settings](https://huggingface.co/settings/tokens), create a fine-grained
   read token named `hjepa-dinov3-read`, scope it only to the DINO repository, and save it
   in a password manager.
5. In RunPod: **Secrets -> Create Secret**. Name it `hf_dinov3_read`, use the token as the
   value, and save.
6. For future pod templates map:

   ```text
   HF_TOKEN={{ RUNPOD_SECRET_hf_dinov3_read }}
   ```

7. Do not restart the pod that is still building EGO4D. After Stage 1 is safe, authenticate
   an already-running pod interactively:

   ```bash
   hf auth login
   hf auth whoami
   ```

Paste the token only into the hidden prompt. Never commit it or put it in a launch command.

### 2B. W&B

1. In W&B: profile icon -> **User Settings** -> create a personal API key if needed.
2. In RunPod Secrets, create `wandb_hjepa` and map future-template variable:

   ```text
   WANDB_API_KEY={{ RUNPOD_SECRET_wandb_hjepa }}
   ```

3. On the active pod verify:

   ```bash
   wandb status
   wandb login --verify
   ```

4. Expected destination:

   ```text
   entity:  smahalanobis-uc-davis
   project: hjepa-vwm
   ```

### 2C. Hardware—not yet

Do not rent the two experiment GPUs until Stage 8 passes. When ready, provision either one
two-GPU pod or two equivalent one-GPU pods. Keep the code, dataset volume, model cache,
CUDA stack, and GPU type identical.

## Stage 3 — `[CODING AGENT]` Build the encoder seam and real adapters

Send the following as one prompt. Do not paraphrase away its constraints.

### Prompt 1 — encoder contracts, configuration, adapters, and dependency pin

```text
Read AGENT_FILES/AGENTS.md and every mandatory linked document first. Then read
AGENT_FILES/KNOWLEDGE/encoders/ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md,
DINOv3_ViT-B-16.md, SigLIP_2_ViT-B-16.md, and the entire current implementation/tests.

Implement only the encoder foundation stage, test-first, without creating a package tree:

1. Create one flat root encoders.py deep module. Its small public interface is
   FeatureLayout, EncoderSpec, FrozenEncoder, and build_frozen_encoder. Keep V-JEPA2,
   DINOv3, and SigLIP2 backend classes private. FrozenEncoder accepts raw float clips in
   [0,1], shape B,T,3,H,W, and returns only dense B,N,D tokens. It owns normalization,
   encoder inference precision, frame microbatching, sticky eval/freeze, cache/revision
   loading, resolved commit capture, shape/range/finiteness checks, and the immutable
   feature fingerprint.
2. Add EncoderConfig and stable aliases vjepa2_vitl16, dinov3_vitb16, siglip2_vitb16.
   Each alias must default to the exact tested 40-character Hub commit SHA, never main;
   documented commands that omit --encoder-revision use that immutable registry default.
   Keep square 8x256x256 as the supported v1 contract and fail loudly otherwise. Make
   requested immutable revision, cache directory, precision, frame microbatch, and
   attention implementation explicit and identity-bearing.
3. V-JEPA must emit layout 4x16x16, D=1024. DINO must flatten all eight frames, remove
   exactly one CLS plus four register tokens, and emit time-major 8x16x16, D=768. SigLIP
   must load the vision tower only, use its correct normalization, validate there are 256
   patch tokens per frame, and emit the same 8x16x16, D=768 layout. Never load SigLIP's
   text tower. Do not temporally pool either frame encoder.
4. Add offline fake-backend unit tests for normalization-once, token stripping/order,
   microbatch equivalence, feature fingerprints, sticky eval/freeze, and error paths. Add
   an authenticated real-adapter smoke CLI to encoders.py which reports resolved revision,
   parameter count, output shape/dtype/range/finiteness, and peak encoder memory without
   exposing credentials.
5. Upgrade and pin one exact tested Transformers 4.x version with DINOv3 support, at least
   4.56 and below 5. Prove all three adapters work on that exact version. Use hf_cache_dir.
6. Do not yet rewrite the data/model/training pipeline in this stage. Preserve the old
   models.FrozenEncoder path temporarily only as a compatibility bridge that the next
   stage will replace; do not duplicate new logic into models.py.
7. Run focused tests, pytest -q, Ruff, Black check, and the existing model/diagnostic smoke
   tests. Report every changed file, test evidence, exact pinned version, and anything that
   requires my authenticated RunPod rather than claiming unrun real-adapter success.

Do not change abstract latent dimensions, loss formulas, schedules, or gradient routing.
Do not ask me questions; make the plan's documented decisions.
```

### Stage 3 verification

The coding agent performs local/offline tests. On an authenticated GPU pod, run each block
separately after pulling the implementation commit:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python -m pip install -r requirements.txt
python encoders.py --smoke --encoder vjepa2_vitl16 --batch-size 1
```

```bash
python encoders.py --smoke --encoder dinov3_vitb16 --batch-size 1
```

```bash
python encoders.py --smoke --encoder siglip2_vitb16 --batch-size 1
```

Expected shapes are `(1,1024,1024)`, `(1,2048,768)`, and `(1,2048,768)`. All report zero
trainable encoder parameters, finite values, an immutable resolved revision, and the
documented parameter scale. Stop on any 401/403, special-token count mismatch, or fallback
to a full SigLIP text model.

## Stage 4 — `[CODING AGENT]` Move the seam through data, B, D, and training

Only start after Stage 3's three adapter contracts are real.

### Prompt 2 — canonical data and generic detailed-feature geometry

```text
Continue from the completed encoder-foundation commit. Re-read AGENT_FILES/AGENTS.md and
AGENT_FILES/KNOWLEDGE/encoders/ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md, then
trace data.py -> encoders.py -> models.py -> train.py -> losses.py -> diagnostics.py at
line level before editing.

Implement the next vertical slice, test-first:

1. Make data.py emit canonical raw float clips in [0,1]. Remove encoder normalization from
   data.py and all private imports of it. Preserve the same context/target window indices,
   shared resize/crop/jitter in full mode, decord random access, VP9 num_threads=1, sorted
   .webm/.mp4 indexing, and EGO4D/SSv2 selection. Version the transform recipe. Explicitly
   validate the supported square 256 crop.
2. Make present-recon-only data genuinely context-only: do not compute or decode target
   indices/frames. Give train.py a clear typed batch-mode boundary. Full mode must still
   apply identical random transforms to the combined context+target stack. Add tests that
   count decoded indices and encoder calls.
3. Resolve the FrozenEncoder and its EncoderSpec before constructing downstream modules.
   Remove ModelConfig's encoder-derived d_e/n_ctx/tubelet/grid values as independent truths;
   retain only a deliberate legacy-checkpoint migration if needed. Offline tests construct
   B and D from synthetic specs without a download.
4. Rewrite Bottleneck geometry to use layout.temporal, height, width, and time_y_x order.
   Rewrite fixed decoder position codes and FeatureMeanTracker/FeatureWhitener sizing from
   the same spec. Preserve c shape N_c=32,D_c=256, the latent stack, fixed non-trainable
   position buffer, no learned content queries, all stop-gradient boundaries, all loss
   formulas, and optimizer membership.
5. Route present and full train/diagnostic forwards through the generic FrozenEncoder.
   The wrapper, not the outer train autocast, owns encoder precision. Change Stage 0 raw
   synthetic clips from torch.randn to torch.rand. Preserve the exact dtype-rounded
   `_assert_ema_transition` check from commit 0af0dc7 and test that every adapter's Stage 0
   invokes it on successful and skipped steps. Fix all V-JEPA/tubelet-only docstrings where
   the code is now generic.
6. Add --encoder, --encoder-revision, --encoder-precision,
   --encoder-frame-microbatch, --batch-size, and --lr-decoder CLI options with strict
   validation. Add --resource-preflight that executes the exact selected recipe's forward,
   backward, optimizer, diagnostic, and peak-memory path without W&B or a research
   checkpoint and writes a JSON report.
7. Update every existing test rather than weakening it. Add both synthetic geometries and
   both batch modes to the matrix. Run pytest -q, Ruff, Black check, model smoke,
   diagnostics smoke, and dataloader smokes. Report the entire forward/backward shape trace
   and prove encoder parameters remain frozen.

Do not add gradient accumulation. Do not alter loss weights, warmups, LR/EMA formulas,
AGC, or accepted run-057 defaults. Do not modify EGO4D manifests or regenerate its data.
Do not ask me questions; use the decisions in the plan.
```

### Stage 4 verification

On the pod, pull the commit and run:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
pytest -q
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

Then verify raw data on both tiny datasets. The values must now remain in `[0,1]`:

```bash
python - <<'EOF'
from config import Config
from data import build_dataloader

for name in ("ssv2_tiny", "ego4d_tiny"):
    cfg = Config()
    cfg.data.dataset = name
    batch = next(iter(build_dataloader(cfg, "train", batch_size=2, needs_target=False)))
    clips = batch.context if hasattr(batch, "context") else batch
    print(name, tuple(clips.shape), float(clips.min()), float(clips.max()))
    assert tuple(clips.shape) == (2, 8, 3, 256, 256)
    assert 0.0 <= float(clips.min()) <= float(clips.max()) <= 1.0
print("raw dual-dataset contract OK")
EOF
```

If the exact batch object differs, the coding agent must update this verification block in
the same commit; the semantic assertions are non-negotiable.

Run a one-step resource preflight for each new encoder with a deliberately small batch:

```bash
python train.py --resource-preflight --data ssv2_tiny --encoder dinov3_vitb16 \
  --batch-size 2 --encoder-frame-microbatch 4 --present-recon-only \
  --lambda-recon 0.05 --lambda-var 0.5 --lambda-cov 0.01 --lambda-sigreg 0 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32
```

```bash
python train.py --resource-preflight --data ssv2_tiny --encoder siglip2_vitb16 \
  --batch-size 2 --encoder-frame-microbatch 4 --present-recon-only \
  --lambda-recon 0.05 --lambda-var 0.5 --lambda-cov 0.01 --lambda-sigreg 0 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32
```

## Stage 5 — `[CODING AGENT]` Make runs reproducible, strict, and resumable

### Prompt 3 — RNG streams, identity, checkpoints, W&B, and parity preflight

```text
Continue from the generic data/model/training commit. Read the full corrected encoder plan
again and audit every source of randomness, construction order, checkpoint field, W&B call,
and data identity before editing.

Implement the reproducibility/provenance slice, test-first:

1. Add deterministic dataset identity. SSv2 full binds labels.json plus the sorted resolved
   clip inventory; ssv2_tiny binds its manifest; EGO4D full binds selection manifest,
   final chunk manifest, split inventory, and completeness; EGO4D tiny also binds its tiny
   manifest and parent identity. Resolve the retained EGO selection manifest through an
   explicit configurable provenance path, defaulting to
   /workspace/ego4d_raw/manifests/selection_manifest.json, and bind the retained
   /workspace/ego4d_raw/video_540ss_manifest.csv authoritative-tier hash as well; never
   search for or guess either path. Refuse final EGO stats/probes if the corpus is incomplete.
2. Isolate named data, model-init, training, and diagnostic RNG streams. Build B/B_EMA/F/D
   inside a forked model-init RNG after encoder loading. Use explicit DataLoader generator,
   worker seeding, resumable sampler epoch/offset, training flow/noise/dropout generator,
   and diagnostic generator. Keep SIGReg's existing per-step generator. Diagnostics must
   not alter future training values.
3. Hash ordered trainable state before optimization. DINO and SigLIP must produce identical
   trainable_init_hash values because their resolved shapes match. Hash the fixed initial
   sample order and validation-batch identity too. Add a preflight JSON/provenance output
   and `python train.py --compare-provenance LEFT.json RIGHT.json` mode that fails unless
   differences are restricted to documented arm-specific fields.
4. Version the checkpoint schema. Store next_step (not ambiguous global_step), completed
   updates, resolved encoder spec/fingerprint, dataset/preprocess identity, stats/mean
   identity, optimizer, every RNG/sampler state, W&B run ID, trainable-init hash, runtime
   versions, git commit, and dirty flag. Keep frozen encoder weights excluded.
5. Validate all compatibility metadata before loading a single model state. Make optimizer
   reset, dataset transfer, or legacy migration explicit flags. Remove silent optimizer
   reset. Inspect a checkpoint before building a whitener so resume can reconstruct the
   embedded whitener even when the external stats file is missing. Resume the same W&B run.
6. Add atomic checkpoint writes. Prove save after update N resumes at N+1. Add an
   interrupted-vs-uninterrupted test that compares sample IDs, RNG draws, parameters,
   optimizer, EMA, mean tracker, and metrics after continuation.
7. W&B config must be updated with resolved runtime provenance after encoder construction.
   Add --require-wandb for real runs; it aborts on initialization/logging failure. Ensure
   diagnostics are logged even when diag_every is not a multiple of log_every, or reject
   that configuration clearly. Use the actual key coarse_vs_batch_mean_ratio.
8. Add --preflight-only/--provenance-out if needed so no-step paired configurations can be
   materialized and compared before launch. Add exact run/group/name/entity/project CLI or
   environment handling. Never leak auth values into config, logs, or checkpoints.
9. Run all tests and explicitly report the initialization-hash parity test, diagnostic RNG
   isolation test, checkpoint mismatch tests, missing-stats resume test, optimizer-reset
   test, and exact resume-equivalence test.

Do not launch a real experiment. Do not loosen deterministic assertions to make tests pass.
Do not change EGO files, loss science, or abstract dimensions. Do not ask me questions.
```

### Stage 5 verification

Run the complete suite twice to catch state leakage:

```bash
pytest -q
pytest -q
```

Run the coding agent's focused resume/parity tests by their committed names, then inspect
the help surface:

```bash
python train.py --help
```

Stop unless the help exposes encoder selection, batch/frame microbatch, resource/preflight,
strict W&B, explicit optimizer reset/transfer policy, and decoder LR.

## Stage 6 — `[CODING AGENT]` Migrate whitening, probes, and artifacts

### Prompt 4 — stats, rank, drift, cache identity, and W&B artifacts

```text
Continue from the strict deterministic training/checkpoint commit. Re-read the encoder
plan, whiten_stats.py, rank_probe.py, every line of drift_probe.py, their tests, and the
EGO4D manifest contract before editing.

Implement the offline-tooling slice, test-first:

1. Migrate whiten_stats.py to build the same FrozenEncoder factory and context-only raw
   dataloader used by training. Encoder precision/preprocessing must be identical across
   training, stats, rank, and drift. Add --encoder, revision, precision, frame-microbatch,
   batch-size, --max-clips, output path, and --inspect. Sampling is deterministic by clip
   identity and records both clip count and token-row count.
2. Write whitening stats atomically with a versioned envelope: mean/eigenvalues/eigenvectors,
   resolved EncoderSpec and feature fingerprint, dataset/split fingerprint, preprocessing
   version, input geometry, precision, transform/data seed, row/clip counts, eigensolver
   settings, creation version, and payload fingerprint. Training validates the whole
   envelope before state construction. DINO and SigLIP stats must reject each other despite
   identical shapes.
3. Migrate rank_probe.py to the factory/spec. Use generic detailed-feature names, include
   raw rank and rank/feature-dimension, dataset/encoder identities in JSON and plot names,
   and retain the existing tested rank formulas.
4. Migrate drift_probe.py to raw clips and the factory. Remove its import of private data
   normalization. Replace the bare V-JEPA cache with a versioned atomic envelope binding
   encoder/revision/token-selection/preprocess/precision, dataset/manifest, probe manifest,
   offsets, storage dtype, and feature fingerprint. Explicit cache paths must still reject
   a mismatch. Default filenames and labels become encoder-generic.
5. Rebuild checkpoint B/whitener for drift from saved EncoderSpec, not current ModelConfig.
   Add a narrow legacy V-JEPA checkpoint reader with an explicit warning. Make cache dtype
   explicit; fp32 is required when exact latent comparisons are requested.
6. Add optional W&B artifact logging for whitening stats, provenance JSON, probe reports,
   and final checkpoints. At minimum, paid runs must log stats/provenance and reference the
   final checkpoint checksum; do not silently upload giant artifacts against policy.
7. Add mismatch, atomicity, metadata, cache, shape, and both-geometry tests. Run every
   existing probe/stat test plus the full repository suite, Ruff, and Black check. Use a
   temporary fake encoder/dataset offline; clearly identify any real GPU test still needed.

Do not compute EGO4D full stats if the dataset completeness fingerprint is not final. Do
not reuse old V-JEPA stats merely because D or the filename matches. Do not ask questions.
```

### Stage 6 verification

The following must print help containing encoder, precision, dataset, fingerprint-aware
output, and deterministic clip-count options:

```bash
python whiten_stats.py --help
python rank_probe.py --help
python drift_probe.py --help
```

Run the committed focused tests, then:

```bash
pytest -q
```

## Stage 7 — `[CODING AGENT]` Re-walk everything and update repository truth

This is not a rubber-stamp pass. It is the requested restart from ingestion to output.

### Prompt 5 — final whole-pipeline audit and repair

```text
Treat the four encoder-pluggability commits as untrusted until verified. Read
AGENT_FILES/AGENTS.md, CODE_DESIGN.md, PROTOCOL.md, the corrected encoder plan, the EGO4D
guide/contract, every root Python file, every test, and all changed diffs.

Now re-walk the complete implemented pipeline at the smallest practical steps:
dataset-root resolution; manifest/file identity; sample ordering; frame-window selection;
decord indices; resize/crop/jitter; raw batch collation; device transfer; adapter
normalization; backend calls/special-token removal/frame reassembly; EncoderSpec; fixed
whitening; mean tracking; B; B_EMA; present/full forwards; residual/full flow construction;
tau/noise/dropout; D; every loss and warmup; optimizer groups/LR; AGC/clip/skip; EMA;
diagnostics; W&B; checkpoint save/load/resume; whitening stats; rank; drift/cache; final
metrics/artifacts. At every step inspect direct and indirect upstream/downstream assumptions.
Fix every discovered gap, then restart the walk from dataset-root resolution. Repeat until
one resolved feature contract owns every encoder-dependent fact and no adapter-specific
logic leaks outside encoders.py.

Required final proof:
- full offline suite, lint, formatting, model/diagnostic smoke;
- authenticated real-adapter shape/freeze/revision preflight for all three encoders;
- ssv2_tiny and ego4d_tiny through all three;
- present-only and full-prediction 100-step paths;
- whitening/stats mismatch rejection;
- full-prediction forward/backward, checkpoint reload, and drift-probe round-trip with
  whitening active for every adapter;
- V-JEPA legacy numerical non-regression;
- DINO/SigLIP trainable-init and data-order hash parity;
- checkpoint exact-resume and probe round-trip;
- strict W&B initialization path;
- resource-preflight JSON for both new encoders.

Update AGENT_FILES/AGENTS.md, AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md,
GUIDES/CODEBASE_STRUCTURE.md, GUIDES/MLOPS.md, requirements, docstrings, and encoder README
so they describe shipped behavior, not the old V-JEPA-only pipeline. Preserve the flat
layout. Clearly state that current full implementation ends at Phase 1 and future pixel
stages remain unimplemented.

Do not launch 15,000 steps. Do not modify dataset content/manifests. Do not claim a gate
passed without command output. Report the complete trace, all fixes made during re-walks,
remaining external/manual gates, exact commands, and final git diff/status.
```

## Stage 8 — `[YOU + CODING AGENT]` Put the verified commit on the pod

The coding agent should give you one verified commit SHA. Do not use an uncommitted working
tree as the experiment source.

Inside the pod:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git status --short
git pull --ff-only
git rev-parse HEAD
python -m pip install -r requirements.txt
pytest -q
```

Stop if `git status --short` shows unexplained changes before pulling, if the pulled SHA is
not the verified SHA, or if tests fail.

Authenticate/verify services, then pre-download sequentially:

```bash
hf auth whoami
wandb login --verify
python encoders.py --smoke --encoder dinov3_vitb16 --batch-size 1
python encoders.py --smoke --encoder siglip2_vitb16 --batch-size 1
python encoders.py --smoke --encoder vjepa2_vitl16 --batch-size 1
```

Do not launch the pair yet.

## Stage 9 — `[CODING AGENT ON POD]` Fit immutable SSv2 whitening artifacts

This is code execution, so the coding agent can do it once your Hugging Face/W&B access is
available. The two fits must select the same 12,800 context clips by identity regardless of
batching. The output paths are intentionally impossible to confuse.

First create the destination:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
mkdir -p /workspace/stats/encoder_pair_ssv2
```

Fit DINOv3 statistics in tmux:

```bash
tmux has-session -t encoder_stats 2>/dev/null && tmux attach -t encoder_stats || tmux new -s encoder_stats
```

Inside tmux:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python whiten_stats.py \
  --data ssv2 \
  --split train \
  --encoder dinov3_vitb16 \
  --encoder-precision bf16 \
  --encoder-frame-microbatch 32 \
  --batch-size 8 \
  --max-clips 12800 \
  --seed 42 \
  --output /workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt
```

Wait for it to finish, then fit SigLIP with the identical clip-selection settings:

```bash
python whiten_stats.py \
  --data ssv2 \
  --split train \
  --encoder siglip2_vitb16 \
  --encoder-precision bf16 \
  --encoder-frame-microbatch 32 \
  --batch-size 8 \
  --max-clips 12800 \
  --seed 42 \
  --output /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt
```

Inspect both envelopes:

```bash
python whiten_stats.py --inspect /workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt
python whiten_stats.py --inspect /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt
sha256sum /workspace/stats/encoder_pair_ssv2/*.pt
```

Required:

- both say dataset `ssv2`, split `train`, seed 42, and 12,800 clips;
- dataset/preprocess/clip-selection identities match;
- feature fingerprints and payload hashes differ;
- both report `N=2048`, `D=768`, bf16 encoder inference, finite eigensystems;
- passing DINO stats to a SigLIP preflight and vice versa is covered by a failing test;
- both artifacts are logged or registered in W&B with their checksums.

Never reuse the historical V-JEPA stats file for either arm.

## Stage 10 — `[CODING AGENT ON POD]` Choose one common resource envelope

Start at the historical physical batch 64. The frame microbatch controls only the frozen
image encoder; it does not reduce B/D's 2,048-token graph. Run the complete selected recipe,
not a toy shape-only forward.

```bash
export COMMON_BATCH=64
export FRAME_MB=32
mkdir -p /workspace/preflight/encoder_pair
```

DINO preflight:

```bash
python train.py --resource-preflight \
  --data ssv2 --encoder dinov3_vitb16 --encoder-precision bf16 \
  --encoder-frame-microbatch "$FRAME_MB" --batch-size "$COMMON_BATCH" --seed 42 \
  --horizon-k 12 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features \
  --whiten-stats-path /workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --provenance-out /workspace/preflight/encoder_pair/dino_resource.json
```

SigLIP preflight:

```bash
python train.py --resource-preflight \
  --data ssv2 --encoder siglip2_vitb16 --encoder-precision bf16 \
  --encoder-frame-microbatch "$FRAME_MB" --batch-size "$COMMON_BATCH" --seed 42 \
  --horizon-k 12 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features \
  --whiten-stats-path /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --provenance-out /workspace/preflight/encoder_pair/siglip_resource.json
```

If either OOMs, set `COMMON_BATCH` to 48 and rerun **both**; then 32, then 16 if needed.
You may lower `FRAME_MB` symmetrically first if the failure is inside the frozen encoder.
Do not use temporal pooling, asymmetric batches, or naive gradient accumulation.

`FRAME_MB` and the attention implementation are feature-identity fields. If Stage 10 lowers
`FRAME_MB` from the value used to fit Stage 9 statistics, first run an unwhitened resource
preflight at the candidate value, then regenerate **both** statistics artifacts with that
same value before rerunning these whitened preflights. A fingerprint mismatch must fail;
never override it merely because the tensor shape matches.

The coding agent must inspect both JSON files and report encoder-only/total peak memory,
throughput, selected batch/microbatch, and whether either arm has suspiciously different
resource behavior. Keep the largest common envelope with safe headroom.

## Stage 11 — `[CODING AGENT ON POD]` Prove parity and run short integrations

Materialize the exact no-step configurations. These commands use the common values chosen
in Stage 10.

```bash
python train.py --preflight-only \
  --data ssv2 --encoder dinov3_vitb16 --encoder-precision bf16 \
  --encoder-frame-microbatch "$FRAME_MB" --batch-size "$COMMON_BATCH" --seed 42 \
  --horizon-k 12 --steps 15000 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 --whiten-features \
  --whiten-stats-path /workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group encoder_substrate_ssv2_v1 --wandb-name dino3b_present_ssv2 \
  --checkpoint-dir /workspace/ckpt/encoder_pair/dino3b \
  --provenance-out /workspace/preflight/encoder_pair/dino_exact.json
```

```bash
python train.py --preflight-only \
  --data ssv2 --encoder siglip2_vitb16 --encoder-precision bf16 \
  --encoder-frame-microbatch "$FRAME_MB" --batch-size "$COMMON_BATCH" --seed 42 \
  --horizon-k 12 --steps 15000 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 --whiten-features \
  --whiten-stats-path /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group encoder_substrate_ssv2_v1 --wandb-name siglip2b_present_ssv2 \
  --checkpoint-dir /workspace/ckpt/encoder_pair/siglip2b \
  --provenance-out /workspace/preflight/encoder_pair/siglip_exact.json
```

Compare them:

```bash
python train.py --compare-provenance \
  /workspace/preflight/encoder_pair/dino_exact.json \
  /workspace/preflight/encoder_pair/siglip_exact.json
```

The comparison must pass and explicitly show identical trainable-init, dataset, sample
order, validation batch, code commit, dependency, batch, seed-stream, schedule, loss, and
downstream architecture hashes. Only encoder/revision/normalization/feature fingerprint,
whitener, run name/path, and eventual GPU assignment may differ.

Before paying for 15k, run 100 steps from scratch on full SSv2 with matching full-SSv2
stats for each arm. Use distinct smoke names/directories and `--require-wandb`. Then run the
agent-authored full-prediction 100-step matrix on both tiny datasets for all three encoders.
Do not proceed if any adapter-specific edit is needed to switch present-only to full mode.

The two exact present-only smoke commands are:

```bash
python train.py --data ssv2 --steps 100 --encoder dinov3_vitb16 \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --batch-size "$COMMON_BATCH" --seed 42 --horizon-k 12 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 20 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 --whiten-features \
  --whiten-stats-path /workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group encoder_substrate_ssv2_v1_smoke --wandb-name dino3b_present_smoke \
  --checkpoint-dir /workspace/ckpt/encoder_pair_smoke/dino3b \
  --require-wandb --log-every 10 --diag-every 50
```

```bash
python train.py --data ssv2 --steps 100 --encoder siglip2_vitb16 \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --batch-size "$COMMON_BATCH" --seed 42 --horizon-k 12 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 20 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 --whiten-features \
  --whiten-stats-path /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group encoder_substrate_ssv2_v1_smoke --wandb-name siglip2b_present_smoke \
  --checkpoint-dir /workspace/ckpt/encoder_pair_smoke/siglip2b \
  --require-wandb --log-every 10 --diag-every 50
```

The shorter smoke changes `recon_warmup_steps` to 20 only so the reconstruction gradient is
actually exercised; it is not scientific data and is never compared to the 15k recipe.

## Stage 12 — `[YOU + CODING AGENT]` Launch the paid pair

### 12A. Final human hardware check

1. Confirm two equivalent GPUs are visible:

   ```bash
   nvidia-smi -L
   ```

2. Confirm no unrelated process is occupying them:

   ```bash
   nvidia-smi
   ```

3. Confirm code/data/artifacts and authentication:

   ```bash
   cd /workspace/hierarchal-jepa-flow-world-model
   git status --short
   git rev-parse HEAD
   hf auth whoami
   wandb login --verify
   test -f /workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt
   test -f /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt
   ```

`git status --short` must be empty. If you intentionally use two pods, repeat these checks
on both and compare the exact commit, dependency lock, dataset fingerprint, and stats
checksums.

### 12B. Enter a persistent session

```bash
tmux has-session -t encoder_pair 2>/dev/null && tmux attach -t encoder_pair || tmux new -s encoder_pair
```

Inside tmux set the values proven in Stage 10 and create distinct outputs:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export COMMON_BATCH=64
export FRAME_MB=32
export PYTHONHASHSEED=42
mkdir -p logs/encoder_pair
mkdir -p /workspace/ckpt/encoder_pair/dino3b
mkdir -p /workspace/ckpt/encoder_pair/siglip2b
```

If Stage 10 chose different common values, replace 64/32 above with those exact values.

### 12C. Launch DINO on GPU 0

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 --steps 15000 --encoder dinov3_vitb16 \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --batch-size "$COMMON_BATCH" --seed 42 --horizon-k 12 \
  --present-recon-only --lambda-recon 0.05 --lambda-recon-pred 0 \
  --recon-loss-mode cosine --recon-warmup-steps 2000 \
  --lambda-var 0.5 --lambda-cov 0.01 --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features \
  --whiten-stats-path /workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group encoder_substrate_ssv2_v1 --wandb-name dino3b_present_ssv2 \
  --checkpoint-dir /workspace/ckpt/encoder_pair/dino3b \
  --require-wandb --log-every 50 --diag-every 500 \
  > logs/encoder_pair/dino3b.log 2>&1 &
export DINO_PID=$!
echo "DINO PID: $DINO_PID"
```

### 12D. Launch SigLIP on GPU 1

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --data ssv2 --steps 15000 --encoder siglip2_vitb16 \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --batch-size "$COMMON_BATCH" --seed 42 --horizon-k 12 \
  --present-recon-only --lambda-recon 0.05 --lambda-recon-pred 0 \
  --recon-loss-mode cosine --recon-warmup-steps 2000 \
  --lambda-var 0.5 --lambda-cov 0.01 --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features \
  --whiten-stats-path /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group encoder_substrate_ssv2_v1 --wandb-name siglip2b_present_ssv2 \
  --checkpoint-dir /workspace/ckpt/encoder_pair/siglip2b \
  --require-wandb --log-every 50 --diag-every 500 \
  > logs/encoder_pair/siglip2b.log 2>&1 &
export SIGLIP_PID=$!
echo "SigLIP PID: $SIGLIP_PID"
```

Do not add `--resume`; both runs start from scratch.

### 12E. Verify both are alive before detaching

```bash
sleep 10
ps -fp "$DINO_PID" "$SIGLIP_PID"
tail -n 20 logs/encoder_pair/dino3b.log
tail -n 20 logs/encoder_pair/siglip2b.log
nvidia-smi
```

Both logs must show the intended encoder, feature/stats/dataset fingerprints, identical
trainable-init and sample-order hashes, `present_recon_only=1`, `prediction_active=0`,
`whiten_active=1`, `recon_target_residual=0`, and no traceback. Detach with **Ctrl-B**, then
**D**.

## Stage 13 — `[YOU OR AGENT]` Monitor without perturbing the runs

Reconnect and inspect logs:

```bash
tmux attach -t encoder_pair
tail -f logs/encoder_pair/dino3b.log
```

Use another terminal for SigLIP or GPU status:

```bash
tail -f logs/encoder_pair/siglip2b.log
watch -n 5 nvidia-smi
```

In W&B open
[the HJEPA-VWM project](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm), filter group
`encoder_substrate_ssv2_v1`, and confirm exactly the two intended run names. You do not need
to create panels manually for the coding/research agent to inspect the runs through W&B.

Hard-stop either run for NaN/Inf, sustained skipped steps, wrong stats/dataset/encoder
identity, wrong mode flags, W&B loss, or asymmetric config/resource changes. A different
throughput is expected and is a measurement, not automatically a validity failure.

At completion verify:

```bash
test -f /workspace/ckpt/encoder_pair/dino3b/phase1_step15000.pt
test -f /workspace/ckpt/encoder_pair/siglip2b/phase1_step15000.pt
sha256sum /workspace/ckpt/encoder_pair/dino3b/phase1_step15000.pt
sha256sum /workspace/ckpt/encoder_pair/siglip2b/phase1_step15000.pt
```

Then ask the research agent to read both W&B histories with the repository's present-only
reading cycle. Do not choose a winner from terminal loss alone.

If you use two one-GPU pods rather than one two-GPU pod, run Stage 12C on pod A and Stage
12D on pod B but change SigLIP's `CUDA_VISIBLE_DEVICES=1` to `CUDA_VISIBLE_DEVICES=0`.
Everything else—including commit, stats checksums, `COMMON_BATCH`, `FRAME_MB`, group, and
seed—must match.

## Stage 14 — Prove and then research full prediction

The implementation has already passed full-mode integration in Stage 7. This explicit
EGO4D-tiny pair demonstrates that switching from present-only to the entire implemented
Phase 1 does not require an encoder-specific code edit. It uses the last canonical full
prediction recipe as a wiring test, with warmups shortened only because the run is 100
steps. It is not a scientific result.

DINO full-mode smoke:

```bash
python train.py --data ego4d_tiny --steps 100 --encoder dinov3_vitb16 \
  --encoder-precision bf16 --encoder-frame-microbatch 16 --batch-size 16 \
  --seed 42 --horizon-k 12 --predict-residual \
  --lambda-var 0.5 --lambda-cov 0 --lambda-slot 0 \
  --lambda-sigreg 5 --sigreg-warmup-steps 20 \
  --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-warmup-steps 20 \
  --recon-loss-mode cosine --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group encoder_full_ego4d_tiny_smoke --wandb-name dino3b_full_ego_smoke \
  --checkpoint-dir /workspace/ckpt/encoder_full_smoke/dino3b \
  --require-wandb --log-every 10 --diag-every 50
```

SigLIP full-mode smoke:

```bash
python train.py --data ego4d_tiny --steps 100 --encoder siglip2_vitb16 \
  --encoder-precision bf16 --encoder-frame-microbatch 16 --batch-size 16 \
  --seed 42 --horizon-k 12 --predict-residual \
  --lambda-var 0.5 --lambda-cov 0 --lambda-slot 0 \
  --lambda-sigreg 5 --sigreg-warmup-steps 20 \
  --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-warmup-steps 20 \
  --recon-loss-mode cosine --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group encoder_full_ego4d_tiny_smoke --wandb-name siglip2b_full_ego_smoke \
  --checkpoint-dir /workspace/ckpt/encoder_full_smoke/siglip2b \
  --require-wandb --log-every 10 --diag-every 50
```

Required full-mode evidence:

- `present_recon_only=0`, `prediction_active=1`;
- context and target are each encoded exactly once;
- finite `L_flow`, `L_recon_pred`, `L_recon_cplus`, and `L_recon_chat`;
- finite `coarse_vs_copy_ratio` and `coarse_vs_batch_mean_ratio`;
- F_c and D both receive/update gradients; E and B_EMA do not;
- both checkpoints save, strictly reload, resume at the next step, and work in drift probe;
- no difference in code path except resolved adapter/spec.

For a paid EGO4D full experiment:

1. Wait for the SSv2 substrate-pair analysis.
2. Create a new KANBAN investigation that states the temporal hypothesis and exact recipe.
3. Use the final EGO4D fingerprint recorded in Stage 1. If whitening is selected, fit
   separate full-EGO training stats for every encoder; never use SSv2 stats.
4. Include a refactored V-JEPA2 control. If only two GPUs are available, run DINO/SigLIP
   in parallel and V-JEPA sequentially under the same committed recipe/resource policy.
5. A full predictor only succeeds if stable late points reach both:

   ```text
   coarse_vs_copy_ratio <= 0.70
   coarse_vs_batch_mean_ratio <= 0.50
   ```

6. Also require healthy representation geometry and reconstruction honesty. EGO and SSv2
   raw reconstruction values are not directly comparable.

Do not blindly resume the old EGO4D guide's Stage 7 command after the refactor; that command
predates encoder-aware fingerprints, normalization, strict revisions, and the outcome of
the new substrate experiment.

## Troubleshooting and hard stops

| Symptom | Meaning | Action |
|---|---|---|
| DINO returns 401/403 | Browser access or token scope is missing | Recheck the model page while logged in, token scope, then `hf auth login`; never paste the token into logs. |
| DINO returns 261/2053/etc. unexpected tokens | CLS/register stripping or input grid is wrong | Stop. Inspect the exact resolved model config and adapter test; do not slice until shape “looks right.” |
| SigLIP process loads ~0.4B parameters | Text tower was loaded | Stop and fix the vision-only adapter. |
| DINO/SigLIP init hashes differ | Backend loading contaminated trainable RNG or downstream configs differ | Stop. Do not call the pair controlled. Fix construction/RNG and rerun preflight. |
| Data/sample hashes differ | Loader RNG/order or dataset identity differs | Stop both arms and fix parity. |
| Whitening fingerprint mismatch | Wrong dataset, encoder, revision, transform, or precision | Regenerate the correct stats; do not override validation. |
| Full EGO stats refuse to run | Corpus is incomplete or manifests/files differ | Return to EGO4D Stage 4D. Never bless a partial cumulative build. |
| Batch 64 OOMs in one arm | No common resource envelope | Lower frame microbatch if encoder-only; otherwise lower physical batch for both. |
| Someone suggests naive accumulation | Batch-statistic objectives would change | Reject it unless exact global-stat accumulation is implemented and gradient-equivalence tested. |
| W&B exception prints a warning but training continues | Strict run mode is broken | Stop; paid runs require `--require-wandb`. |
| Resume repeats the saved step | `next_step` semantics are wrong | Stop and fix checkpoint tests before a long run. |
| Resume needs a deleted external stats file | Embedded-whitener bootstrap is wrong | Stop and fix strict self-contained resume. |
| `diag_every` metrics never appear | Diagnostic logging is still coupled to `log_every` | Fix/impose the validated cadence before launch. |
| Raw DINO recon loss is lower | Not sufficient evidence of a better encoder | Read within-run gain, shuffled gap/share, geometry, stability, compute, and full prediction. |
| EGO4D tiny contains more than requested after changing parameters | Stale symlinks from an old tiny selection | Cleanly rebuild tiny from the final parent dataset; do not edit its manifest. |

## What you personally never need to do

You do not need to hand-edit Python, JSON manifests, W&B run config, checkpoint metadata,
feature-cache metadata, model config, or requirements. You only:

1. finish/verify the concurrent EGO4D build;
2. accept the DINO license and supply secrets securely;
3. provide equivalent GPUs after all implementation gates pass;
4. paste the verified launch blocks with the preflight-selected common batch values;
5. stop on a hard gate and hand the evidence back to the coding/research agent.

All implementation, tests, model downloads after authentication, stats fitting, parity
comparison, smoke runs, probes, W&B reading, and code/document maintenance can be handled
by the coding agent.
