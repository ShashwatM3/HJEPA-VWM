# Guide: From V-JEPA-only to Encoder-Pluggable Experiments

> Updated 2026-07-14: Prompt 1C's strict encoder seam, fake two-layout contracts, pinned
> V-JEPA2 adapter, and exact Transformers lock are complete. The data hot path,
> preprocessing, whitening, checkpoints, and probes are still V-JEPA-specific. The guide
> is dependency-segmented: finish the remaining common track once, run the
> SigLIP 2 ViT and DINOv3 tracks independently, and meet at one join gate before a paid
> comparison. Stage numbers identify commands; they are not a requirement to wait for DINO.

The design authority is
[`ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md`](ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md).
The encoder research is in [`DINOv3_ViT-B-16.md`](DINOv3_ViT-B-16.md) and
[`SigLIP_2_ViT-B-16.md`](SigLIP_2_ViT-B-16.md).

## Dependency map — use this while DINO access is pending

In this guide, **the ViT lane** means the selected standard encoder,
**SigLIP 2 ViT-B/16**. V-JEPA2 is a separate regression-control lane.

```text
COMMON TRACK — no DINO or SigLIP files required
  EGO4D verification + W&B + encoder interface/two-layout fake contracts
  data/model/training refactor + determinism/checkpoints + stats/probes
  V-JEPA2 regression + repository-wide pre-join audit
                         |
             +-----------+-----------+
             |                       |
   SIGLIP 2 ViT LANE          DINOV3 LANE
   available immediately      offline adapter available now
   private adapter            private adapter
   real smoke                 approval + auth, then real smoke
   SSv2/EGO4D smokes          SSv2/EGO4D smokes
   SSv2 stats                 SSv2/EGO4D smokes
   resource profile           SSv2 stats + resource profile
             |                       |
             +-----------+-----------+
                         |
                 JOIN/PARITY GATE
       same final commit, dependency lock, dataset, seeds,
       init/data hashes, common batch, strict W&B, config diff
                         |
          15k runs concurrently OR sequentially
```

### What can be completed before DINO approval

Do all of the following now:

1. Stage 1 and Stage 2B. Stage 2A's browser request is already submitted; only its
   post-approval authentication waits.
2. Common Prompts 1C, 2, 3, 4, and 5A in Stages 3-7. Both frame-encoder geometries are
   represented by strict injected fake backends/fixtures; the common code does not import,
   download, or branch on DINO or SigLIP.
3. The real V-JEPA2 regression, which proves the refactor preserves the current encoder.
4. The independent SigLIP Prompt 1S and its real tests on both tiny datasets.
5. The SigLIP-only portions of Stages 9, 10, 11, and 14: fit its SSv2 stats, resource-profile
   it, and run short present-only/full-mode smokes.
6. DINO Prompt 1D-O, which implements and exhaustively fake-tests the private adapter but
   explicitly leaves real revision/weight validation pending.
7. Prepare the final code, dependency lock, dataset fingerprints, W&B project, directories,
   and experiment notes.

Do **not** launch the 15,000-step SigLIP run yet. A DINO-specific fix could still change
the shared commit or dependency lock. Short SigLIP smokes are disposable; the paid pair
must start only after the join gate.

### What genuinely waits for DINO approval

Only these items wait:

1. authenticate/download the gated DINO files and resolve/pin their exact Hub commit;
2. run the real DINO adapter, SSv2/EGO4D smokes, whitening fit, and resource profile;
3. rerun any shared SigLIP preflight invalidated by a DINO-driven shared-code change;
4. execute the join/parity gate and start the 15,000-step runs.

The two final runs do not communicate. **Concurrent execution is optional.** Sequential
execution is scientifically valid when both use the same immutable code commit, dependency
lock, dataset/stats fingerprints, seed streams, common physical batch/frame microbatch,
hardware class, and resolved config. Parallel execution only saves wall-clock time.

### Safe ordering and safe concurrency

The safest single-agent order is:

```text
1C -> 2 -> 3 -> 4 -> 5A -> (1S and 1D-O) -> 1D-R -> 5B -> paid pair
```

Prompt 1S and Prompt 1D-O may be reversed. They may also run concurrently **after Prompt
1C is committed**, but only in separate git worktrees/branches based on that same common
commit. Never point two coding agents at the same checkout: both adapters live in
`encoders.py` and their edits would race. The coding agents—not you—should create the
worktrees, merge both lane commits into the audited common branch, resolve any overlap,
and execute Prompt 5B. If separate worktrees are unavailable, run the lanes sequentially.

The common Prompts 2-5A can also proceed in one worktree while an adapter is implemented
in another, because their shared public interface is frozen by Prompt 1C. Before any real
pipeline smoke, merge/rebase the adapter onto the completed common pipeline and rerun its
entire lane verification. Concurrency saves time; it never waives the post-merge tests.

| Track | Exact work | Prerequisite | Can run now? |
|---|---|---|---|
| Common | Stage 1, Stage 2B, Prompt 1C, Prompts 2-4, Prompt 5A, V-JEPA regression, common Stage 8 sync | None beyond the existing repo/data | Yes |
| SigLIP | Prompt 1S, real dual-dataset/mode tests, 9S, 10S, 11S, 14S | Prompt 1C; merge common pipeline before real pipeline tests | Yes |
| DINO offline | Prompt 1D-O and exact fake-backend tests | Prompt 1C | Yes |
| DINO real | Finish 2A auth, Prompt 1D-R, real dual-dataset/mode tests, 9D, 10D, 11D, 14D | DINO offline plus Hugging Face approval | After approval |
| Join | Integrate both lanes, Prompt 5B, common resource envelope, 11J, both final 100-step smokes | Common + SigLIP + DINO lanes complete | No |
| Research pair | Stage 12-13, concurrently or sequentially | Join passes on one immutable commit | No |

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

Current checkpoint: the Prompt 1C foundation is implemented, but none of the remaining
finish-line bullets should be inferred from that. Continue at Prompt 2 after completing the
human EGO4D verification prerequisite.

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

### 2A. DINO lane: Hugging Face approval and authentication

The access request has been submitted. That is sufficient for now: continue the common and
SigLIP lanes. Return to steps 4-7 only after the model page says access is granted.

1. Log in at [Hugging Face](https://huggingface.co/).
2. Open the
   [DINOv3-B model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
3. Read and accept the DINOv3 license/access terms. The access request must be made in the
   browser and belongs to your individual account. If it says the request is pending, stop
   this subsection without blocking any common/SigLIP work.
4. Open [token settings](https://huggingface.co/settings/tokens), create a fine-grained
   read token named `hjepa-dinov3-read`, scope it only to the DINO repository, and save it
   in a password manager.
5. In RunPod: **Secrets -> Create Secret**. Name it `hf_dinov3_read`, use the token as the
   value, and save.
6. For future pod templates map:

   ```text
   HF_TOKEN={{ RUNPOD_SECRET_hf_dinov3_read }}
   ```

7. Do not restart a pod merely to inject the secret. Authenticate an already-running pod
   interactively after access is granted:

   ```bash
   hf auth login
   hf auth whoami
   ```

Paste the token only into the hidden prompt. Never commit it or put it in a launch command.

### 2B. Common track: W&B

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

### 2C. Join gate: hardware—not yet

Do not rent experiment GPUs until the common and adapter lanes pass. For concurrent runs,
provision one two-GPU pod or two equivalent one-GPU pods. For sequential runs, one GPU is
enough. In either case keep code, dataset volume, model cache, CUDA stack, GPU type, and
resource envelope identical.

## Stage 3 — `[COMMON CODE; NO DINO APPROVAL]` Build the encoder seam

**Completed 2026-07-14.** The repository pins `transformers==4.57.6` and V-JEPA2 Hub
revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`. The real adapter produced
`(1,1024,1024)`, 325,971,328 frozen parameters, finite fp32 features, and exact numerical
parity with the temporary legacy path. The prompt remains below as the auditable contract;
do not rerun it unless this foundation changes.

Send the following as one prompt. Do not paraphrase away its constraints.

### Prompt 1C — generic contracts, V-JEPA regression adapter, and offline fixtures

```text
Read AGENT_FILES/AGENTS.md and every mandatory linked document first. Then read
AGENT_FILES/KNOWLEDGE/encoders/ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md,
DINOv3_ViT-B-16.md, SigLIP_2_ViT-B-16.md, and the entire current implementation/tests.

Implement only the encoder-independent foundation stage, test-first, without creating a
package tree:

1. Create one flat root encoders.py deep module. Its small public interface is
   FeatureLayout, EncoderSpec, FrozenEncoder, and build_frozen_encoder. Keep every backend
   class private. FrozenEncoder accepts raw float clips in
   [0,1], shape B,T,3,H,W, and returns only dense B,N,D tokens. It owns normalization,
   encoder inference precision, frame microbatching, sticky eval/freeze, cache/revision
   loading, resolved commit capture, shape/range/finiteness checks, and the immutable
   feature fingerprint.
2. Add EncoderConfig and a private adapter registry. Refactor the existing V-JEPA2 backend
   behind it and pin vjepa2_vitl16 to its exact tested 40-character Hub commit SHA. Reserve
   stable aliases dinov3_vitb16 and siglip2_vitb16, but make either unresolved alias fail
   clearly until its lane supplies a tested private adapter and immutable default revision;
   never fall back to main. Keep square 8x256x256 as the supported v1 contract and fail
   loudly otherwise. Make requested immutable revision, cache directory, precision, frame
   microbatch, and attention implementation explicit and identity-bearing.
3. V-JEPA must preserve layout 4x16x16,D=1024. Add injected offline fixtures for both a
   tubelet layout and a generic time-major frame layout 8x16x16,D=768. The common pipeline
   may depend only on EncoderSpec/FeatureLayout, never on a DINO/SigLIP class, processor,
   special-token rule, or model ID. Do not implement either real frame adapter here.
4. Add offline fake-backend unit tests for normalization-once, token selection/order,
   microbatch equivalence, both layouts, feature fingerprints, sticky eval/freeze, registry
   failures, and error paths. Add
   an authenticated real-adapter smoke CLI to encoders.py which reports resolved revision,
   parameter count, output shape/dtype/range/finiteness, and peak encoder memory without
   exposing credentials.
5. Pin one exact Transformers 4.x version whose installed source supports the two planned
   adapter architectures, at least 4.56 and below 5, without downloading their weights.
   Prove the real V-JEPA regression on that version now. The SigLIP and DINO lanes must
   validate this same pin against their real weights; a later dependency change invalidates
   every earlier real-adapter result. Use hf_cache_dir.
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

The coding agent performs both-layout local/offline tests. DINO and SigLIP are not needed.
On a GPU pod, run the real regression adapter after pulling the implementation commit:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python -m pip install -r requirements.txt
python encoders.py --smoke --encoder vjepa2_vitl16 --batch-size 1
```

Expected shape is `(1,1024,1024)`. It must report zero trainable encoder parameters, finite
values, an immutable resolved revision, and the documented parameter scale. Real SigLIP
and DINO verification belongs only to their independent lanes below.

## Stage 4 — `[CODING AGENT]` Move the seam through data, B, D, and training

Start after Prompt 1C's common interface/fake contracts and real V-JEPA regression pass.
Nothing in this stage may need real DINO or SigLIP weights.

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

Run one real V-JEPA resource preflight with a deliberately small batch:

```bash
python train.py --resource-preflight --data ssv2_tiny --encoder vjepa2_vitl16 \
  --batch-size 2 --encoder-frame-microbatch 4 --present-recon-only \
  --lambda-recon 0.05 --lambda-var 0.5 --lambda-cov 0.01 --lambda-sigreg 0 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32
```

The committed test suite must run the same forward/backward path with injected tubelet and
frame-layout fixtures. Real SigLIP/DINO resource commands belong only to their lanes.

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
3. Hash ordered trainable state before optimization. Two independently named, shape-matched
   frame-layout fixtures must produce identical trainable_init_hash values; the final join
   repeats this proof with real DINO and SigLIP. Hash the fixed initial sample order and
   validation-batch identity too. Add a preflight JSON/provenance output and a
   `--compare-provenance LEFT.json RIGHT.json` mode that fails unless differences are
   restricted to documented arm-specific fields.
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
   envelope before state construction. Stats from two fake frame encoders with identical
   shapes but different feature fingerprints must reject each other; the final join repeats
   the test with real DINO and SigLIP.
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

## Stage 7 — `[COMMON PRE-JOIN AUDIT; NO DINO APPROVAL]` Re-walk everything

This is not a rubber-stamp pass. It is the requested restart from ingestion to output.

### Prompt 5A — pre-join whole-pipeline audit and repair

```text
Treat the common encoder-pluggability commits as untrusted until verified. Read
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
- authenticated real-adapter shape/freeze/revision preflight for V-JEPA;
- ssv2_tiny and ego4d_tiny through V-JEPA plus strict synthetic tubelet/frame geometries;
- present-only and full-prediction V-JEPA paths plus offline two-layout forward/backward;
- whitening/stats mismatch rejection;
- full-prediction forward/backward, checkpoint reload, and drift-probe round-trip with
  whitening active for V-JEPA and both synthetic geometries;
- V-JEPA legacy numerical non-regression;
- two independent shape-matched frame fixtures with trainable-init/data-order hash parity;
- checkpoint exact-resume and probe round-trip;
- strict W&B initialization path;
- real resource-preflight JSON for V-JEPA plus fixture resource-path coverage. Record both
  real frame adapters as independent lane work; do not let either block shared completion.

Update AGENT_FILES/AGENTS.md, AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md,
GUIDES/CODEBASE_STRUCTURE.md, GUIDES/MLOPS.md, requirements, docstrings, and encoder README
so they describe shipped behavior, not the old V-JEPA-only pipeline. Preserve the flat
layout. Clearly state that current full implementation ends at Phase 1 and future pixel
stages remain unimplemented.

Do not launch 15,000 steps. Do not modify dataset content/manifests. Do not claim either
real frame adapter passed in this common audit. Report the complete trace, all fixes made
during re-walks, the two pending adapter-lane evidence bundles, exact commands, and final
git diff/status.
```

## Stage 8 — `[COMMON OR ADAPTER LANE]` Put a verified commit on the pod

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

Verify W&B and always run the V-JEPA regression:

```bash
wandb login --verify
python encoders.py --smoke --encoder vjepa2_vitl16 --batch-size 1
```

If the pulled commit includes the completed SigLIP lane, run its smoke too. Run real DINO
only when its lane is complete and access is granted. Do not launch the pair yet.

## Stage 9 — `[SPLIT ADAPTER LANES]` Fit immutable SSv2 whitening artifacts

The two fits are independent and may happen days apart. Fit SigLIP now; fit DINO after
approval. Both must select the same 12,800 context clips by identity regardless of batching.
The output paths are intentionally impossible to confuse.

Before running a fit, complete that adapter's Prompt 1S or Prompt 1D-R in the lane section
immediately below Stage 9 and pass its real smoke/pipeline verification. Stage numbers do
not override this dependency.

First create the destination:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
mkdir -p /workspace/stats/encoder_pair_ssv2
```

Enter the stats tmux session:

```bash
tmux has-session -t encoder_stats 2>/dev/null && tmux attach -t encoder_stats || tmux new -s encoder_stats
```

### 9S. SigLIP 2 ViT lane — run now

Inside tmux:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
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

Inspect it immediately. It is usable for SigLIP short smokes without DINO:

```bash
python whiten_stats.py --inspect /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt
sha256sum /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt
```

### 9D. DINOv3 lane — run only after approval

After `hf auth whoami` and the real DINO adapter smoke pass, fit DINO with the identical
clip-selection settings:

```bash
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

When both lanes are complete, inspect the pair:

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

## SigLIP 2 ViT lane — available immediately

This lane has no dependency on DINO access or DINO code. Start it after Prompt 1C freezes
the public encoder interface. If it runs concurrently with Prompts 2-5A, use a separate
worktree; merge it onto the completed common pipeline before running real pipeline smokes.

### Prompt 1S — implement and validate the real SigLIP adapter

```text
Continue from the committed Prompt 1C encoder interface. Read AGENT_FILES/AGENTS.md, the
corrected encoder plan, this dependency-segmented guide, SigLIP_2_ViT-B-16.md, encoders.py,
and every encoder test. Do not require DINO access or DINO code.

Complete only the SigLIP 2 ViT-B/16 lane:

1. Implement the private siglip2_vitb16 adapter without changing the public
   FeatureLayout/EncoderSpec/FrozenEncoder interface. Load only the vision tower—never the
   text tower—and pin its exact tested 40-character Hub commit as the registry default.
2. Keep preprocessing inside the wrapper and apply the model's exact image normalization
   once. Preserve all eight frames, validate exactly 256 patch tokens per frame, flatten in
   time-y-x order, and emit B,2048,768 without CLS/text tokens or temporal pooling.
3. Prove sticky eval/freeze, zero trainable encoder parameters, expected vision-parameter
   scale, fp32/bf16 behavior, frame-microbatch equivalence, finiteness, immutable revision
   capture, feature fingerprint stability, and clear range/shape/revision errors.
4. After merging/rebasing onto the completed common pipeline, run SigLIP on ssv2_tiny and
   ego4d_tiny in present-only and full prediction. Exercise whitening mismatch rejection,
   checkpoint save/strict exact resume, rank, drift, cache identity, resource-preflight,
   diagnostic RNG isolation, and V-JEPA-vs-SigLIP generic-path regression. No SigLIP
   conditional may exist outside encoders.py.
5. Run the full offline suite plus authenticated SigLIP tests. Report commands/output,
   exact dependency/revision, feature fingerprint, parameter/peak-memory evidence, changed
   files, and final git status. If any shared interface/dependency/preprocessing change was
   unavoidable, list every common/V-JEPA artifact or test invalidated by it.

Do not implement DINO. Do not fit final SSv2 stats or launch 15,000 steps in this prompt.
Do not alter losses, schedules, abstract geometry, manifests, or gradient routing. Do not
ask me questions; use the documented contracts.
```

After Prompt 1S is integrated, complete this lane in order:

1. run the real SigLIP smoke and prove the text tower is not loaded:

   ```bash
   python encoders.py --smoke --encoder siglip2_vitb16 --batch-size 1
   ```

2. run real SigLIP batches on `ssv2_tiny` and `ego4d_tiny` in present-only and full mode;
3. run whitening/checkpoint/rank/drift round-trips and the V-JEPA-vs-SigLIP generic-path
   regression matrix;
4. fit and inspect the SigLIP SSv2 stats in Stage 9S;
5. run Stage 10S to find SigLIP's safe resource ceiling;
6. materialize the SigLIP exact config and run its 100-step present-only smoke in Stage 11;
7. run the SigLIP full-prediction EGO4D-tiny smoke in Stage 14.

These outputs are useful engineering evidence, but do not launch the 15,000-step SigLIP
research run before the join gate.

## DINOv3 lane — offline adapter now, real evidence after approval

The common fake-backend tests already exercise the generic time-major frame layout,
normalization-once, microbatching, shape, freeze, and error contracts. The DINO lane owns
its exact 1+4 token rule. Its implementation and strict fakes do not need gated weights;
only immutable revision resolution and real evidence wait.

### Prompt 1D-O — implement the DINO adapter offline while access is pending

```text
Continue from the committed Prompt 1C encoder interface. Read AGENT_FILES/AGENTS.md, the
corrected encoder plan, this dependency-segmented guide, DINOv3_ViT-B-16.md, encoders.py,
the pinned Transformers implementation/config classes, and every encoder test. Do not
download gated weights and do not require SigLIP code.

Implement only the offline DINOv3 ViT-B/16 lane:

1. Add the private dinov3_vitb16 adapter behind the frozen public interface. Keep the alias
   unable to perform a research load without an explicit immutable revision until real
   validation supplies the tested default; never use main or invent a SHA.
2. Put DINO-only model construction, normalization, output extraction, one-CLS/four-register
   removal, per-frame 256-patch validation, and time-y-x reassembly inside encoders.py.
   Preserve all eight frames and emit B,2048,768; never temporally pool.
3. Use an injected strict fake matching the installed Transformers DINO output/config
   structure. Prove exact token identity/order—not just output shape—plus fp32/bf16,
   frame-microbatch equivalence, normalization-once, sticky eval/freeze, feature-fingerprint
   fields, and failures for wrong special-token count/grid/range/revision.
4. Do not change the public interface for a backend convenience. Run the full offline suite,
   Ruff, Black check, model/diagnostic smokes, and report changed files/test output. Label
   model download, parameter scale, resolved SHA, real numerical microbatch comparison,
   and real pipeline runs as pending external evidence.

Do not claim the real encoder works. Do not implement SigLIP. Do not fit stats or launch a
run. Do not change losses, schedules, abstract geometry, manifests, or gradient routing.
Do not ask me questions; use the documented contracts.
```

### Prompt 1D-R — resolve and validate the real gated DINO adapter

```text
Continue from the integrated Prompt 1D-O adapter and completed common pipeline. The SigLIP
lane may or may not already be merged; do not depend on it.
Read AGENT_FILES/AGENTS.md, the corrected encoder plan, this dependency-segmented guide,
encoders.py, every encoder test, and the recorded pre-join audit. DINO access is now granted.

Complete only the real DINO validation lane:

1. Authenticate through the existing HF token mechanism without printing or serializing
   the token. Resolve the exact DINOv3 ViT-B/16 Hub commit and pin that 40-character SHA as
   the dinov3_vitb16 registry default. Never use main for a research run.
2. Load the real model into hf_cache_dir on the already pinned Transformers version. Prove
   zero trainable parameters, sticky eval, expected ~85.7M scale, bf16/fp32 behavior, exact
   removal of one CLS plus four registers, exactly 256 patch tokens per frame, time-major
   8x16x16x768 output, microbatch equivalence, finiteness, and resolved-revision capture.
3. Run the real DINO path on ssv2_tiny and ego4d_tiny in present-only and full prediction,
   including whitening mismatch rejection, checkpoint save/strict resume, rank, drift,
   cache identity, and diagnostic RNG isolation. Do not add any DINO conditional outside
   encoders.py.
4. If real DINO exposes a backend quirk, contain it in the private adapter. If a shared
   interface/preprocessing/dependency change is unavoidable, explicitly invalidate and
   list every common, V-JEPA, and already-integrated SigLIP artifact/preflight that must be
   rerun. Never silently preserve an old fingerprint.
5. Run the full offline suite plus authenticated DINO tests and report commands/output,
   exact revision, feature fingerprint, peak memory, changed files, and final git status.

Do not fit final stats or launch 15,000 steps in this prompt. Do not claim parity merely
from matching tensor shapes. Do not ask me questions; use the documented contracts.
```

After that prompt passes, run:

```bash
hf auth whoami
python encoders.py --smoke --encoder dinov3_vitb16 --batch-size 1
```

Then complete Stage 9D, Stage 10D, the DINO exact-config/100-step blocks in Stage 11, and
the DINO EGO4D-tiny full-mode block in Stage 14.

## Stage 10 — `[SPLIT ADAPTER LANES]` Measure resources independently

Start at the historical physical batch 64. The frame microbatch controls only the frozen
image encoder; it does not reduce B/D's 2,048-token graph. Run the complete selected recipe,
not a toy shape-only forward. Run SigLIP now. Run DINO after approval. The final common
envelope is the lower safe envelope and is chosen only when both reports exist.

```bash
export COMMON_BATCH=64
export FRAME_MB=32
mkdir -p /workspace/preflight/encoder_pair
```

### 10D. DINOv3 resource preflight — after approval

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

### 10S. SigLIP 2 ViT resource preflight — run now

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

Within an adapter lane, if batch 64 OOMs, try 48, then 32, then 16 and record its largest
safe value. After both lanes finish, choose the lower safe batch/frame-microbatch and rerun
**both** at that common value. Do not use temporal pooling, asymmetric final batches, or
naive gradient accumulation.

`FRAME_MB` and the attention implementation are feature-identity fields. If Stage 10 lowers
`FRAME_MB` from the value used to fit Stage 9 statistics, first run an unwhitened resource
preflight at the candidate value, then regenerate **both** statistics artifacts with that
same value before rerunning these whitened preflights. A fingerprint mismatch must fail;
never override it merely because the tensor shape matches.

The coding agent can inspect each JSON independently. The join gate compares both reports,
records encoder-only/total peak memory and throughput, and keeps the largest common envelope
with safe headroom.

## Final join coding audit — only after both adapter lanes pass

### Prompt 5B — real three-adapter join and invalidation audit

```text
The common pre-join audit, real SigLIP/V-JEPA lane, and real DINO lane are complete. Read
AGENT_FILES/AGENTS.md, the encoder plan/guide, all implementation/tests, every lane report,
artifact metadata, resource JSON, and the full diff since Prompt 5A.

Perform the final join audit without launching 15,000 steps:

1. Integrate the completed Prompt 1S, Prompt 1D-O, and Prompt 1D-R lane commits into the
   audited common branch. If they were built in parallel worktrees, merge/cherry-pick them yourself,
   resolve overlapping private-registry/adapter tests without dropping either lane, and
   rerun the full suite. The human must not have to hand-merge encoders.py.
2. Determine the one final clean commit and exact dependency lock. If either lane changed any
   shared encoder interface, preprocessing version, precision, dependency, data/model code,
   or fingerprint field, invalidate and rerun every affected SigLIP/V-JEPA test, stats file,
   resource preflight, provenance file, checkpoint/probe round-trip, and short smoke. Reuse
   an artifact only when strict metadata validation proves it remains identical.
3. Re-walk the complete pipeline from dataset identity to final artifacts with all three
   real adapters. Prove no DINO/SigLIP/V-JEPA conditional exists outside encoders.py and one
   resolved EncoderSpec owns all encoder-dependent facts.
4. Run all three on ssv2_tiny and ego4d_tiny in present-only and full prediction; run
   whitening-active forward/backward, checkpoint strict reload/exact resume, rank, drift,
   cache, diagnostics, and strict W&B paths.
5. Rerun DINO and SigLIP resource preflights at the largest safe common physical batch and
   frame microbatch. Regenerate stats first if the identity-bearing frame microbatch changes.
6. Materialize both final 15k provenance JSON files. Prove identical trainable-init, dataset,
   sample-order, validation-batch, code/dependency, seed-stream, schedule/loss, batch, and
   downstream hashes. Permit only documented encoder/stat/run/path/GPU differences.
7. Run both final 100-step present-only smokes from scratch. Report W&B URLs, checksums,
   test output, allowed config diff, remaining human launch steps, and clean git status.

Do not waive a mismatch. Do not modify datasets/manifests. Do not launch paid runs. Do not
ask me questions; resolve all discoverable inconsistencies and stop only on an external
failure that cannot be fixed in code.
```

## Stage 11 — `[JOIN GATE; BOTH REAL ADAPTERS REQUIRED]` Prove parity

Each adapter's no-step configuration and short smoke may be materialized in its own lane.
Before both resource reports exist, that output is **provisional** and uses the lane's own
safe values. The comparison and paid-run authorization wait for both. After Stage 10D and
10S, export the lower common `COMMON_BATCH`/`FRAME_MB` and rerun **both** exact-config
commands; only those regenerated JSON files are eligible for 11J.

### 11D. DINO exact configuration — after approval

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

### 11S. SigLIP exact configuration — provisional now, final after 10D

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

### 11J. Join comparison — only after Prompt 5B

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

## Stage 12 — `[JOINED PAIR; YOU + CODING AGENT]` Launch the paid runs

### 12A. Final human hardware check

1. Choose either valid schedule:

   - **Concurrent:** two equivalent GPUs, one arm per GPU.
   - **Sequential:** one GPU, DINO and SigLIP one after the other. Do not change the pod,
     GPU type, code, environment, artifacts, or recipe between arms.

2. Confirm the available GPU or GPUs:

   ```bash
   nvidia-smi -L
   ```

3. Confirm no unrelated process is occupying any experiment GPU:

   ```bash
   nvidia-smi
   ```

4. Confirm code/data/artifacts and authentication:

   ```bash
   cd /workspace/hierarchal-jepa-flow-world-model
   git status --short
   git rev-parse HEAD
   hf auth whoami
   wandb login --verify
   test -f /workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt
   test -f /workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt
   ```

`git status --short` must be empty. The coding agent must pre-download both encoders
sequentially before this stage so concurrent jobs cannot race in the Hub cache. If you use
two pods, repeat these checks on both and compare the exact commit, dependency lock, CUDA
stack, GPU type, dataset fingerprint, and stats checksums. A sequential pair on one
unchanged GPU satisfies hardware parity by construction.

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

### 12D. Launch SigLIP concurrently or sequentially

For a **concurrent** pair, run this block immediately and use device 1:

```bash
export SIGLIP_DEVICE=1
```

For a **sequential** pair, do not run this block until DINO has completed successfully,
its step-15000 checkpoint exists, and its W&B history is intact. Then use the same GPU:

```bash
export SIGLIP_DEVICE=0
```

In either schedule, launch the otherwise identical SigLIP arm:

```bash
CUDA_VISIBLE_DEVICES="$SIGLIP_DEVICE" python train.py \
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

### 12E. Verify the active run or runs before detaching

For concurrent execution:

```bash
sleep 10
ps -fp "$DINO_PID" "$SIGLIP_PID"
tail -n 20 logs/encoder_pair/dino3b.log
tail -n 20 logs/encoder_pair/siglip2b.log
nvidia-smi
```

For sequential execution, check DINO immediately after Stage 12C:

```bash
sleep 10
ps -fp "$DINO_PID"
tail -n 20 logs/encoder_pair/dino3b.log
nvidia-smi
```

After DINO finishes and Stage 12D starts SigLIP, repeat for SigLIP:

```bash
sleep 10
ps -fp "$SIGLIP_PID"
tail -n 20 logs/encoder_pair/siglip2b.log
nvidia-smi
```

Every active log must show the intended encoder, feature/stats/dataset fingerprints,
the preflight-approved trainable-init and sample-order hashes,
`present_recon_only=1`, `prediction_active=0`, `whiten_active=1`,
`recon_target_residual=0`, and no traceback. For the sequential schedule, compare the two
logs/provenance files before declaring the pair complete; the hashes must match the Stage
11 join result. Detach with **Ctrl-B**, then **D**.

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
`encoder_substrate_ssv2_v1`, and confirm the intended run name for every arm launched so
far. A concurrent schedule should show both promptly; a sequential schedule shows DINO
first and both after SigLIP starts. You do not need to create panels manually for the
coding/research agent to inspect the runs through W&B.

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

If you use two one-GPU pods rather than one two-GPU pod, run Stage 12C on pod A and set
`SIGLIP_DEVICE=0` before Stage 12D on pod B. Everything else—including commit, dependency
lock, stats checksums, `COMMON_BATCH`, `FRAME_MB`, group, and seed—must match.

## Stage 14 — `[SPLIT ADAPTER LANES]` Prove and then research full prediction

The implementation has already passed full-mode integration in Stage 7. This explicit
EGO4D-tiny pair demonstrates that switching from present-only to the entire implemented
Phase 1 does not require an encoder-specific code edit. It uses the last canonical full
prediction recipe as a wiring test, with warmups shortened only because the run is 100
steps. It is not a scientific result.

### 14D. DINO full-mode smoke — after approval

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

### 14S. SigLIP full-mode smoke — may run now

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

Required full-mode evidence for each lane independently:

- `present_recon_only=0`, `prediction_active=1`;
- context and target are each encoded exactly once;
- finite `L_flow`, `L_recon_pred`, `L_recon_cplus`, and `L_recon_chat`;
- finite `coarse_vs_copy_ratio` and `coarse_vs_batch_mean_ratio`;
- F_c and D both receive/update gradients; E and B_EMA do not;
- that lane's checkpoint saves, strictly reloads, resumes at the next step, and works in
  drift probe;
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

1. finish the already-built EGO4D Stage 6 smoke and record its immutable fingerprints;
2. accept the DINO license after approval and supply secrets securely;
3. provide one stable GPU for sequential runs or two equivalent GPUs for concurrent runs
   after all implementation gates pass;
4. paste the verified launch blocks with the preflight-selected common batch values;
5. stop on a hard gate and hand the evidence back to the coding/research agent.

All implementation, tests, model downloads after authentication, stats fitting, parity
comparison, smoke runs, probes, W&B reading, and code/document maintenance can be handled
by the coding agent.
