# Post-SigLIP setup for DINOv3 ViT-B/16

This is the handoff from the completed standard-ViT/SigLIP lane to the gated DINO lane. The
selected weights are already known:

```text
alias:      dinov3_vitb16
repository: facebook/dinov3-vitb16-pretrain-lvd1689m
model:      DINOv3 ViT-B/16, approximately 85.7M parameters
runtime:    Transformers 4.57.6, DINOv3ViTModel
target:     8 frames x 256 patches x 768 = (B,2048,768)
```

The repository does **not** yet have a usable DINO adapter. The alias is deliberately reserved
with no private factory and no default revision, so `--encoder dinov3_vitb16` currently fails
instead of falling back to mutable `main`. Do not fit DINO stats or start a DINO run until the
implementation and real-weight gates below are complete.

## The steps you personally need to do

### 1. Check whether Meta approved the gated checkpoint

On your normal browser, while logged into the same Hugging Face account you will use on RunPod:

1. Open
   [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
2. If the page says **You have access** and lets you open/download `config.json`, continue.
3. If it still says **Request pending**, stop. There is no code or token command that bypasses the
   approval.
4. If it asks you to accept/share details again, review the
   [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/), accept only
   if you agree, and wait for the page to show access.

An unauthenticated view of the official page still reports this as a gated model. A coding agent
on your Mac cannot see the approval state of your private Hugging Face account.

### 2. Log the RunPod into that exact Hugging Face account

SSH into the pod, then run:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
hf auth login
hf auth whoami
```

`hf auth login` opens a browser/device flow or asks for a personal token. A read-only token is
sufficient. Paste it only into the hidden authentication prompt—never into a repository file,
chat, W&B config, shell command, screenshot, or log. Do not run `hf auth token`, because that prints
the active secret.

Verify `hf auth whoami` prints the same account that received DINO access.

### 3. Prove access and capture the current candidate snapshot SHA

This downloads only the small config, not the 343MB weight file, and never prints credentials:

```bash
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache /workspace/preflight/dinov3

python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download

repo = "facebook/dinov3-vitb16-pretrain-lvd1689m"
info = HfApi().model_info(repo)
sha = info.sha
assert isinstance(sha, str) and len(sha) == 40, sha
path = hf_hub_download(
    repo_id=repo,
    filename="config.json",
    revision=sha,
    cache_dir="/workspace/hf_cache",
)
print("candidate_revision:", sha)
print("config_cached:", path)
PY
```

Save the printed 40-character `candidate_revision` and send only that SHA to the coding agent.
This is a candidate pin, not yet a tested default: the agent must validate the actual weights and
feature contract before committing it.

### 4. Send the offline-adapter prompt to the coding agent

Run this on the normal coding branch before trying a DINO launch:

```text
Read AGENT_FILES/AGENTS.md and every mandatory linked document first. Then read
AGENT_FILES/KNOWLEDGE/encoders/ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md,
DINOv3_ViT-B-16.md, GUIDE_encoders.md, POST_STD_VIT_SETUP_FOR_DINO.md, encoders.py,
the installed Transformers 4.57.6 DINOv3ViT implementation/config, and every encoder,
pipeline, stats, checkpoint, rank, and drift test.

Implement only the offline DINOv3 ViT-B/16 adapter lane, test-first:

1. Add one private DINOv3ViTModel adapter behind the existing FrozenEncoder interface in
   flat encoders.py. Do not expose a DINO class, processor, model ID, or special-token rule
   outside encoders.py.
2. Keep dinov3_vitb16 unavailable without an explicit immutable 40-character revision until
   real validation succeeds. Never resolve or fall back to main. Do not install a default
   revision merely because I supplied a candidate SHA.
3. Own ImageNet normalization exactly once, load through hf_cache_dir, preserve sticky
   eval/freeze and the selected precision/attention/frame-microbatch identity, encode every
   frame independently, assert config.num_register_tokens == 4, remove exactly one CLS plus
   four register tokens, assert 256 patch tokens per frame and width 768, restore time-major
   order, and return only (B,2048,768).
4. Add strict injected fake-model tests that prove token identity/order—not only shape—plus
   normalization-once, range/finiteness errors, special-token/grid/width failures,
   frame-microbatch equivalence, fp32/bf16 behavior, sticky eval/freeze, immutable revision
   capture, parameter count, and feature fingerprinting. Keep all common code encoder-agnostic.
5. Run focused tests, the full suite, Ruff, Black check, model/diagnostic smokes, and every
   offline two-layout regression. Do not download gated weights, fit stats, launch training,
   or claim real DINO success. Commit the offline lane and report the exact commit and the
   one real RunPod command I must run next.

Do not ask questions and do not change losses, schedules, latent dimensions, data manifests,
or gradient routing.
```

The coding agent owns all code and tests in that prompt. You should not hand-edit `encoders.py`.

### 5. Pull the offline adapter to RunPod and run the explicit-revision smoke

After the coding agent commits and you push that commit, SSH to RunPod:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
git status --short
git pull --ff-only
python -m pip install -r requirements.txt
pytest -q
hf auth whoami
```

Stop if the worktree is dirty, tests fail, or the account is wrong. Export the candidate SHA from
Step 3, then run the command the coding agent reports. Its expected shape is this:

```bash
read -r -p "Paste the 40-character candidate DINO revision: " DINO_SHA
[[ "$DINO_SHA" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "STOP: revision must be exactly 40 lowercase hexadecimal characters"; false; }
export DINO_SHA

python encoders.py --smoke \
  --encoder dinov3_vitb16 \
  --revision "$DINO_SHA" \
  --precision bf16 \
  --frame-microbatch 32 \
  --attention-implementation sdpa \
  --hf-cache-dir /workspace/hf_cache \
  --batch-size 1 \
  --device cuda \
  | tee /workspace/preflight/dinov3/real_adapter_smoke.json
```

The first real call downloads/caches `model.safetensors`; you do not need to find or manually
copy a separate pretrained-weights file. Expected evidence:

```text
requested_revision = resolved_revision = $DINO_SHA
transformers_version = 4.57.6
parameter_count approximately 85.7M
trainable_parameter_count = 0
output_shape = [1, 2048, 768]
output_finite = true
peak_encoder_memory_bytes is non-null on CUDA
```

If it fails, send the complete traceback and smoke JSON to the coding agent, but never send the
token. Do not work around a token-count/config mismatch by blindly slicing five tokens.

### 6. Send the real-validation-and-pin prompt

Once Step 5 passes, send this prompt with the candidate SHA and smoke output path/results:

```text
Continue from the committed offline DINO adapter. Real RunPod validation passed at the explicit
revision <PASTE_40_CHAR_SHA>; use the attached credential-free smoke JSON/error evidence. Read the
same mandatory encoder docs and all adapter/pipeline tests again.

Complete the gated-real DINO lane:

1. Pin <PASTE_40_CHAR_SHA> as dinov3_vitb16's immutable default only after reconciling the real
   config/output with the private adapter contract. Never use main.
2. Add/finish regression coverage for the real config seam, exact one-CLS/four-register removal,
   256 patches per frame, time-major (B,2048,768), zero trainables, sticky eval, feature
   fingerprint, and resolved revision. Contain every DINO quirk inside encoders.py.
3. Preserve Transformers 4.57.6. If any shared interface, preprocessing, dependency, or
   fingerprint field changes, explicitly invalidate and list every SigLIP/V-JEPA test, stats,
   provenance, cache, checkpoint, and preflight that must be regenerated.
4. Run the full offline suite, Ruff, Black check, and all locally possible pipeline/probe tests.
   Give me exact credential-safe RunPod commands for final CUDA present-only/full-mode,
   whitening, checkpoint-resume, rank, drift, and resource evidence. Do not fit final full stats
   or launch 15,000 steps yourself.
5. Commit the validated default and report the commit SHA and remaining external gates.

Do not change losses, schedules, abstract dimensions, manifests, or gradient routing. Do not
claim any real test I did not supply or you did not actually run.
```

### 7. Pull the pinned adapter and prove the stable alias

After that commit is pushed:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
git status --short
git pull --ff-only
python -m pip install -r requirements.txt
pytest -q

python encoders.py --smoke \
  --encoder dinov3_vitb16 \
  --precision bf16 \
  --frame-microbatch 32 \
  --attention-implementation sdpa \
  --hf-cache-dir /workspace/hf_cache \
  --batch-size 1 \
  --device cuda \
  | tee /workspace/preflight/dinov3/pinned_alias_smoke.json
```

The requested and resolved revisions must equal the tested SHA even though `--revision` is now
omitted. This proves the alias cannot drift with Hub `main`.

## What the coding agent still owns after the stable alias works

The coding agent must specify and verify the full real-adapter matrix before a paid run:

- present-only and full-prediction paths on `ssv2_tiny` and `ego4d_tiny`;
- strict checkpoint save/resume and mismatched-encoder rejection;
- DINO-specific whitening-envelope creation/reload and SigLIP↔DINO mismatch rejection;
- rank probe, drift probe, feature-cache identity, and diagnostic RNG isolation;
- V-JEPA and SigLIP regressions on the unchanged dependency/shared interface;
- unwhitened resource preflight to settle physical batch and frame microbatch;
- final DINO stats, whitening-active preflight, exact provenance, and 100-step smoke.

Use [GUIDE_encoders.md](GUIDE_encoders.md) Stages 9D–14 after the adapter is pinned, but regenerate
commands/artifacts if the shared implementation changed. Never reuse SigLIP whitening merely
because both encoders emit `(B,2048,768)`.

## Final join gate before DINO versus SigLIP research runs

Do not launch the paired 15,000-step experiment until all of these are true:

1. both aliases resolve immutable tested SHAs on the same clean Git commit;
2. both use exactly Transformers 4.57.6 and the same CUDA/runtime environment;
3. both real CUDA smoke reports pass;
4. both have encoder-specific stats built from the same deterministic 12,800 clip identities;
5. physical batch and frame microbatch are safe for both, then both stats/preflights are rerun at
   the final common identity-bearing values;
6. exact provenance comparison permits only encoder/stat/output-path differences;
7. both 100-step W&B smokes pass from byte-identical trainable initialization;
8. no DINO/SigLIP conditional exists outside the private adapter registry.

SigLIP can always be replaced by `--encoder vjepa2_vitl16` or `--encoder siglip2_vitb16`; adding
DINO does not remove either path. Checkpoints and whitening artifacts remain encoder-bound, so
switching the CLI alias also requires the matching stats and a fresh run—not a cross-encoder
resume.

## Official references

- [Meta DINOv3 ViT-B/16 checkpoint and model card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- [Hugging Face authentication quickstart](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
- [DINOv3 reference repository](https://github.com/facebookresearch/dinov3)
- [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
- [Project DINO encoder contract](DINOv3_ViT-B-16.md)
