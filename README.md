# HJEPA-VWM

Hierarchical JEPA-Flow video world model — planning docs and Phase 1 implementation for v0.

## Architecture

HJEPA-VWM trains a video world model whose internal state is a two-level latent hierarchy: a detailed latent `e_t` (from a **frozen pretrained V-JEPA 2 ViT-L/16 encoder**, `D_e=1024`) for local visual detail and a compressed abstract latent `c_t` (from a trainable bottleneck `B`, 32×256) for future-relevant structure. Phase 1 predicts the future abstract latent `c⁺_{t+k}` with a conditional rectified-flow model `F_c`, against a stop-gradient **EMA bottleneck** target (`B_EMA`); collapse is prevented by a **variance floor on `c_t`** (no SIGReg). This is not a video diffusion model; the compressed predictive state is the core object being tested.

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

- `config.py` — locked constants and path defaults (frozen encoder repo, `D_e=1024`)
- `make_subset.py` — symlink-only SSv2-tiny creation
- `data.py` — SSv2 clips: 8 context frames + 8-frame future clip (horizon `k`) at 256×256, encoder-normalized
- `models.py` — frozen encoder `E` (V-JEPA 2 ViT-L/16), bottleneck `B`, EMA bottleneck `B_EMA`, coarse flow `F_c`
- `losses.py` — rectified-flow matching and the variance floor on `c_t`
- `diagnostics.py` — the three required `c_t` monitors (variance, cross-video cosine, effective rank), F_c baselines, gradient health
- `train.py` — Stage 0 synthetic sanity and Stage 1 training loop

> The frozen encoder downloads `facebook/vjepa2-vitl-fpc64-256` (~1.2 GB) from Hugging Face on first
> run and is never trained or checkpointed (reloaded from HF).

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

### Training flags

| Flag | Values | Use |
|---|---|---|
| `--data` | `ssv2`, `ssv2_tiny` | Selects the dataset split root. |
| `--steps` | integer | Sets max training steps for this launch. |
| `--resume` | checkpoint path | Loads model and optimizer state from a checkpoint. |
| `--seed` | integer | Sets Python/Torch RNG seed. |
| `--stage0-only` | boolean flag | Runs one synthetic sanity step instead of training. |
| `--log-every` | integer | Overrides console/W&B train metric frequency. |
| `--diag-every` | integer | Overrides validation diagnostic frequency. |
| `--checkpoint-dir` | path | Writes checkpoints to this directory. |
| `--horizon-k` | integer frames | Sets future offset in original video frames. |
| `--predict-residual` | boolean flag | Predicts `c_{t+k}-c_t` instead of full `c_{t+k}`. |
| `--present-recon-only` | boolean flag | Trains only `D(B(e_t))->e_t` and skips prediction losses. |
| `--recon-loss-mode` | `cosine`, `relative_mse` | Chooses new unit-normalized cosine recon or legacy `MSE/Var(e)`. |
| `--lambda-recon` | float >= 0 | Weights present reconstruction `D(c_t)->e_t`. |
| `--lambda-recon-pred` | float >= 0 | Weights predicted-future reconstruction `D(c_hat)->e_{t+k}`. |
| `--recon-warmup-steps` | integer | Linearly ramps reconstruction losses over this many steps. |
| `--lambda-var` | float >= 0 | Weights the `c_t` variance floor. |
| `--lambda-sigreg` | float >= 0 | Weights SIGReg isotropic-Gaussian regularization on `c_t`. |
| `--sigreg-warmup-steps` | integer | Linearly ramps SIGReg over this many steps. |
| `--lambda-cov` | float >= 0 | Weights optional VICReg-C covariance penalty. |
| `--lambda-slot` | float >= 0 | Weights optional slot-diversity penalty. |
| `--lr-bottleneck` | float | Sets peak LR for bottleneck `B`. |
| `--lr-coarse-flow` | float | Sets peak LR for coarse flow `F_c`. |
| `--no-agc` | boolean flag | Disables adaptive gradient clipping. |
| `--agc-lambda-bottleneck` | float > 0 | Sets AGC clip factor for `B`. |
| `--agc-lambda-coarse-flow` | float > 0 | Sets AGC clip factor for `F_c`. |
| `--grad-skip-threshold` | float | Skips optimizer steps above this post-AGC global norm. |
| `--decoder-dim` | integer | Sets reconstruction decoder width. |
| `--decoder-blocks` | integer | Sets reconstruction decoder depth. |
| `--n-c` | integer | Sets abstract latent slot count. |

## Verification

Local lightweight tests (no encoder download — `smoke_test_models` synthesizes `e_t`):

```bash
pytest tests/test_phase1_contract.py -q
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
python -m py_compile config.py make_subset.py data.py models.py losses.py diagnostics.py train.py
```

RunPod checks after dependencies are installed (these download the frozen encoder):

```bash
python -c "from config import Config; print(Config())"
python -c "from data import smoke_test_dataloader; smoke_test_dataloader()"
python -c "from models import smoke_test_encoder; smoke_test_encoder()"   # loads + verifies frozen E
python train.py --stage0-only
```

## Documentation layout

| Folder | Contents |
|---|---|
| `AGENT_FILES/AGENT-BEHAVIOUR/` | `PROTOCOL.md`, `CODE_DESIGN.md` |
| `AGENT_FILES/KNOWLEDGE/` | Architecture brief, `UNDERSTANDING.md` |
| `AGENT_FILES/PHASES/` | Phase 1–3 implementation specs |
| `AGENT_FILES/SETUPS/` | `VOLUME_LAYOUT.md`, `SETUP.md`, `SETUP_POD.md` |
