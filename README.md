# HJEPA-VWM

Hierarchical JEPA-Flow video world model — planning docs and Phase 1 implementation for v0.

## Architecture

HJEPA-VWM trains a video world model whose internal state is a two-level latent hierarchy: a detailed latent `e_t` for local visual detail and a compressed abstract latent `c_t` for future-relevant structure. Phase 1 predicts future abstract latents with a coarse flow model `F_c` against a stop-gradient EMA target branch. This is not a video diffusion model; the compressed predictive state is the core object being tested.

## For coding agents

Start with `AGENT_FILES/AGENTS.md`. It defines the mandatory read order, document precedence, RunPod volume contract, and phase workflow. Do not infer architecture from generic ML patterns.

## Network volume layout

The repo is code-only. Runtime assets live as siblings under `/workspace`:

```text
/workspace/hierarchal-jepa-flow-world-model/   # git repo, code only
/workspace/data/ssv2/                          # full SSv2 symlink layout
/workspace/data/ssv2_tiny/                     # created by make_subset.py
/workspace/ssv2_raw/                           # raw .webm backing files
/workspace/checkpoints/                        # training checkpoints
/workspace/hf_cache/                           # Phase 3 VAE cache
```

Local development can override only the dataset parent with:

```bash
export JEPA_DATA_ROOT=/path/to/local/data
```

See `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md` and `AGENT_FILES/SETUPS/SETUP.md` step A8 for the one-time RunPod data migration.

## Phase 1: coarse hierarchy

Phase 1 implements:

- `config.py` — locked constants and path defaults
- `make_subset.py` — symlink-only SSv2-tiny creation
- `data.py` — SSv2 clips: 4 context frames + 1 future frame at 128x128
- `models.py` — patchifier, online encoder `E`, bottleneck `B`, EMA target branch, coarse flow `F_c`
- `losses.py` — rectified-flow matching and SIGReg
- `diagnostics.py` — latent std/effective-rank health, F_c baselines, gradient health
- `train.py` — Stage 0 synthetic sanity and Stage 1 training loop

Phase 1 does not include `FineFlow`, frame generation, VAE loading, Stage 2–4 logic, or shuffled-c diagnostics.

### What the Phase 1 baselines mean

`F_c` must predict the future abstract target `c_plus` better than trivial alternatives:

- copy baseline: pretend the current abstract latent `c_t` is already the future
- batch-mean baseline: predict every future abstract latent as the batch mean

After 10k+ steps, acceptance requires:

- model L_c <= 0.70 x copy-baseline L_c
- model L_c <= 0.50 x batch-mean-baseline L_c

## One-time SSv2-tiny creation

Run this once on the RunPod after `/workspace/data/ssv2` exists:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python make_subset.py --train-per-class 23 --val-per-class 2 --seed 42
```

Expected output is roughly 4,002 train symlinks and 348 validation symlinks, depending on undersized classes.

## Phase 1 commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Stage 0 synthetic sanity:

```bash
python train.py --stage0-only
```

500-step smoke run:

```bash
python train.py --data ssv2_tiny --steps 500
```

Full Phase 1 smoke/acceptance run on tiny:

```bash
python train.py --data ssv2_tiny --steps 30000
```

Full dataset run:

```bash
python train.py --data ssv2 --steps 30000
```

Expected runtime: about 4–5 hours for 30k steps on an A100 80GB with `ssv2_tiny`; measure and update after the first real RunPod run.

## Verification

Local lightweight tests:

```bash
pytest tests/test_phase1_contract.py -q
python -m py_compile config.py make_subset.py data.py models.py losses.py diagnostics.py train.py
```

RunPod checks after dependencies are installed:

```bash
python -c "from config import Config; print(Config())"
python -c "from data import smoke_test_dataloader; smoke_test_dataloader()"
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
python train.py --stage0-only
```

## Documentation layout

| Folder | Contents |
|---|---|
| `AGENT_FILES/AGENT-BEHAVIOUR/` | `PROTOCOL.md`, `CODE_DESIGN.md` |
| `AGENT_FILES/KNOWLEDGE/` | Architecture brief, `UNDERSTANDING.md` |
| `AGENT_FILES/PHASES/` | Phase 1–3 implementation specs |
| `AGENT_FILES/SETUPS/` | `VOLUME_LAYOUT.md`, `SETUP.md`, `SETUP_POD.md` |
