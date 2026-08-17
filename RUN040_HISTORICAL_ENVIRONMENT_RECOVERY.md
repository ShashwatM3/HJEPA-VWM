# Run 040 historical evaluation environment recovery

Date: 2026-08-05

## Executive decision

**CONDITIONAL GO.** The historical source tree and all four learned modules can be recovered exactly and loaded strictly without changing the current checkout. An evaluation can be run later in an isolated worktree, but it must be labelled **reconstructed evaluation**, not a replay of the original fixed batch: W&B did not save validation sample identifiers or a fixed batch, and the exact Hugging Face revision was not recorded. No forward pass, training, checkout, download, upload, or S3 mutation was performed during this audit.

## 1. Authoritative Run 040 provenance

W&B run `smahalanobis-uc-davis/hjepa-vwm/io74f32b` is the authoritative record.

- Name: `Investigation 11 · Fixed-position decoder · Full prediction`
- State: `finished`; group: `inv011_fixed_position_decoder`
- Created: `2026-06-30T18:48:31Z`
- Git commit: `87ee2e1a9a8d1d084a6d7ca128374bfe108a8cc5`
- Remote: `https://github.com/ShashwatM3/HJEPA-VWM.git`
- Program: `/workspace/hierarchal-jepa-flow-world-model/train.py`; code path `train.py`; executable `/usr/bin/python`
- Host: `d981893f8672`; Python: CPython 3.11.10
- Seed: `42`; no additional deterministic-algorithm flag was recorded
- Data: `ssv2`, `data_root=/workspace/data`, workers 8, pinned memory
- Encoder: `facebook/vjepa2-vitl-fpc64-256`, frozen; patch 16, tubelet 2, input 256x256, `d_e=1024`
- Encoder revision: **not recorded** in W&B config, metadata, command, or checkpoint
- Temporal geometry: context 8, stride 2, prediction horizon 12
- Model: `n_c=32`, `d_c=256`, CoarseFlow dim 256, 8 heads, 6 blocks; fixed-position decoder dim 512, 8 heads, 4 blocks
- Objective: predictive mode (`predict_residual=true`, `present_reconstruction=false`), flow/coarse loss enabled, `lr_coarse_flow=1e-4`; bottleneck and CoarseFlow received finite gradients
- Reconstruction: cosine loss, `lambda_recon=0.05`, `lambda_recon_pred=0.05`, warm-up 2000
- Schedule: batch 64, 15,000 steps, checkpoint every 2,500, diagnostics every 500
- Command arguments: `--data ssv2 --steps 15000 --horizon-k 12 --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lambda-var 0.5 --lambda-sigreg 5.0 --sigreg-warmup-steps 2000 --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-loss-mode cosine --recon-warmup-steps 2000 --predict-residual --decoder-dim 512 --decoder-blocks 4 --n-c 32 --checkpoint-dir /workspace/ckpt/inv011_fixed_position_decoder --log-every 50 --diag-every 500`

W&B retained only `wandb_manifest.json`, `config.yaml`, `output.log`, `requirements.txt`, `wandb-metadata.json`, and `wandb-summary.json`. `codeSaved` is null. There is no uploaded source artifact, validation table, sample-ID table, or fixed diagnostic batch.

### Observed plateau evidence

The run remained numerically stable: no reported NaNs or skipped steps and finite gradient norms around 2–3. Representations did not collapse by the standard-deviation/dead-dimension checks (`c_std_mean≈1.005`, dead fraction 0, cross-video cosine ≈0.165), but effective rank was low (about 50.8, below the 60 target).

The decisive failure is predictor advantage. The diagnostic copy/model ratio fell from 22.94 at step 0 to the vicinity of 1 and stayed there late in training: 0.994 at 9,000; 0.987 at 10,000; 1.005 at 12,000; 1.031 at 13,000; 1.028 at 14,000; and 0.971 at 14,500. At 14,500 the model loss was 1.462 versus copy loss 1.506; the batch-mean ratio was 1.067. This fails both the desired copy advantage (≤0.7) and batch-mean advantage (≤0.5). The present/c-plus/c-hat reconstruction values were approximately 0.346/0.344/0.353, showing only weak separation. The observed conclusion is a coarse/copy plateau, not numerical collapse.

## 2. Exact source and checkpoint compatibility

Commit `87ee2e1a9a8d1d084a6d7ca128374bfe108a8cc5` exists locally as a commit object. Its parent is `0a9ff46f76b5a206e91dc2cc365f86bb32ea490f`; subject: `Add fixed-position decoder run guide`. It is contained by multiple local/remote-tracking branches, although no currently advertised remote head or tag points exactly to it. No fetch or checkout was needed.

For read-only verification, the commit was exported to `/tmp/run040-87ee2e1`. Instantiating the historical classes with the saved config and `load_encoder=False`, then loading on CPU with `strict=True`, produced:

| Module | State keys | Parameters | Result |
|---|---:|---:|---|
| Bottleneck `B` | 33 | 2,203,904 | strict load OK |
| EMA bottleneck `B_EMA` | 33 | 2,203,904 | strict load OK |
| CoarseFlow `F_c` | 70 | 7,643,904 | strict load OK |
| Decoder | 59 | 14,842,368 | strict load OK |

Checkpoint: `artifacts/checkpoints/inv011_fixed_position_decoder_phase1_step15000.pt`, 301,116,256 bytes, SHA-256 `969d197854320c71e774961de1039db69ab2eed682696d3b5297e96a43de7927`, saved step 15,000. Top-level entries are `bottleneck`, `target_bottleneck`, `coarse_flow`, `decoder`, `optimizer`, `config`, and `global_step`. It contains no run ID, source SHA, encoder revision, schema version, validation IDs, or fixed batch.

This is sufficient to establish exact learned-module compatibility with the historical commit without a forward pass.

## 3. Why the current Bottleneck rejects the historical state

The historical Bottleneck loads with 10 unexpected keys and the current Bottleneck reports 70 missing keys.

Historical-only keys:

- `cross_attn.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}`
- `out_mlp.0.{weight,bias}`
- `out_mlp.1.{weight,bias}`
- `out_mlp.3.{weight,bias}`

Current-only missing groups:

- `latent_blocks.0.*`: 23 keys
- `latent_blocks.1.*`: 23 keys
- `latent_blocks.2.*`: 23 keys
- `pos_emb`: 1 key

The old implementation performs one `nn.MultiheadAttention` cross-read followed by one output MLP. The current implementation has learned position embeddings and three latent blocks, each with normalized sharp cross-attention (`q/k/v/o` projections and logit scale), self-attention, multiple normalization layers, and an MLP. The old fused QKV tensor could be split mechanically, and the old MLP tensors have shape-level analogues in a current block, but that would not preserve behavior: positional information, cosine/sharp attention, self-attention, repeated cross-reads, and initialization/residual structure are new.

Therefore:

- **Exact adapter:** feasible by retaining the historical Bottleneck class (or an exact compatibility wrapper implementing it) and loading its ten-key topology.
- **Weight conversion into the current Bottleneck:** technically fabricable but behaviorally non-equivalent and unsuitable for historical evaluation.
- **Recommended:** run the historical source at the recorded commit; do not transform the checkpoint.

## 4. Validation batch and S3 provenance

Historical validation behavior at the recorded commit:

- Dataset files are `sorted((root/split).glob("*.webm"))`.
- Validation loader uses `shuffle=False`, 8 workers, and batch size `min(16, 64)=16`.
- The diagnostic batch is `next(iter(val_loader))`.
- Each sample uses 8 context and 8 target frames, stride 2, horizon 12; validation uses midpoint temporal start `max_start // 2` and center crop.
- The required temporal span is `(8-1)*2 + 12 = 26` frames; short videos repeat the last frame.
- The historical batch tuple does not retain source video IDs.

Read-only, capped S3 metadata listings found:

- Raw videos: `s3://hjepa-volume/ssv2_raw/20bn-something-something-v2/*.webm`
- Raw annotations: `s3://hjepa-volume/ssv2_raw/labels/labels/{labels.json,train.json,validation.json,test.json,test-answers.csv}`; validation JSON is 3,733,168 bytes
- Processed location: `s3://hjepa-volume/data/ssv2/train/*.webm` plus `data/ssv2/labels.json`
- No `data/ssv2/validation/` prefix was returned
- No `evaluation_results/run040/` objects were returned; the only discovered evaluation-results child was `run069/`
- No Run 040 fixed-batch/cache artifact was found in the targeted locations

The minimum payload for a **new reconstructed 16-sample diagnostic batch** is the validation annotation file plus the exact 16 `.webm` objects selected after recreating the historical validation directory and lexicographic ordering. Those 16 object IDs cannot be named confidently yet because the original `/workspace/data/validation` membership/symlink construction was not recorded and the S3 processed prefix currently exposes only `train/`. Downloading an arbitrary first 16 videos would not reproduce the historical batch.

No S3 objects were downloaded, modified, moved, deleted, or uploaded. No sync was run.

## 5. Encoder identity and uncertainty

Ordered recovery evidence:

1. **Confirmed:** W&B config names `facebook/vjepa2-vitl-fpc64-256`; it was frozen and loaded through the historical repository code.
2. **Confirmed:** S3 contains one snapshot prefix: `s3://hjepa-volume/hf_cache/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/`.
3. **Not confirmed:** no `refs/` object was returned, and neither W&B nor the checkpoint records a revision or cache snapshot hash.
4. **Inference only:** because exactly one relevant snapshot is currently visible in the bucket, `b3c1679…` is the strongest encoder candidate. It cannot be claimed as the exact snapshot used by Run 040 without contemporaneous cache metadata, logs containing resolved revision, or an immutable model-file hash match.

## 6. Safe next-phase reconstruction plan (not executed)

1. Create a detached worktree at the exact commit; leave the current branch and working tree untouched.
2. Create an isolated Python 3.11 environment and install the W&B-captured requirements after reviewing them for historical availability.
3. Verify the local checkpoint hash, then CPU-load metadata and all four modules strictly using the historical classes.
4. Recover encoder snapshot `b3c1679…` only after accepting that it is the strongest candidate rather than proven exact; pin loading to the snapshot and use offline/local-files-only mode.
5. Reconstruct the validation split from `validation.json`; create a manifest of the lexicographically first 16 historical validation paths and download only those 16 videos plus the annotation file. Do not use `aws s3 sync`.
6. Record the reconstructed IDs, object sizes, ETags, preprocessing parameters, dependency lock, commit, encoder snapshot, checkpoint hash, and deterministic settings.
7. Label all outputs `Run 040 reconstructed evaluation`; never call them the original diagnostic batch.
8. Only after all compatibility gates pass, run the requested no-training diagnostics. Keep forward evaluation as a separately authorized phase.

Proposed commands for that later authorized phase (shown only; **not executed**):

```bash
git worktree add --detach ../HJEPA-VWM-run040-87ee2e1 87ee2e1a9a8d1d084a6d7ca128374bfe108a8cc5
cd ../HJEPA-VWM-run040-87ee2e1
python3.11 -m venv .venv-run040
source .venv-run040/bin/activate
python -m pip install -r /path/to/wandb-run040-requirements.txt
sha256sum ../HJEPA-VWM/artifacts/checkpoints/inv011_fixed_position_decoder_phase1_step15000.pt
```

After a manifest identifies the exact reconstructed validation objects, use one explicit `aws s3 cp` per object, always with `--region ap-south-1 --no-cli-pager`; for example:

```bash
aws s3 cp s3://hjepa-volume/ssv2_raw/labels/labels/validation.json artifacts/run040/data/validation.json --region ap-south-1 --no-cli-pager
aws s3 cp s3://hjepa-volume/ssv2_raw/20bn-something-something-v2/ID.webm artifacts/run040/data/validation/ID.webm --region ap-south-1 --no-cli-pager
aws s3 cp s3://hjepa-volume/hf_cache/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/ artifacts/run040/hf_cache/snapshot/ --recursive --region ap-south-1 --no-cli-pager
```

The recursive encoder command is confined to one immutable snapshot prefix and is not `sync`, but its byte size should be listed and reviewed before authorization. The video commands must be expanded to exactly 16 explicit IDs; `ID.webm` is intentionally a placeholder and must not be executed.

## Final gate

**CONDITIONAL GO** for a separately authorized, isolated **reconstructed evaluation**. The learned-module environment is exact at commit `87ee2e1…`; the remaining blockers to claiming an exact historical replay are the unrecorded encoder revision and missing original validation sample identities/fixed batch.
