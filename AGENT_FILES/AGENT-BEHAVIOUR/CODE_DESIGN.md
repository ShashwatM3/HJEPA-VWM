# Hierarchical JEPA-Flow — Code Style & Readability Guide

> A reference for **how to write the code**: file structure, naming, docstrings, formatting.
> This is not an implementation plan. No build order, no phases, no sequencing — just the
> conventions that keep the codebase readable and the gradient boundaries auditable.

---

## 1. Guiding principles

1. **Readability beats cleverness.** Optimize for a tired version of you reading this later.
2. **One function per named concept in the brief.** If the brief names it (bottleneck, coarse flow,
   variance floor, EMA update, shuffled-c test), it gets exactly one function/class that owns it.
3. **Modular, but not fragmented.** A function maps to a step in the pipeline, not to three lines
   you felt like extracting. Avoid both mega-functions and one-line-wrapper sprawl.
4. **The gradient graph must read clearly.** Anything touching stop-gradient, EMA, or detached
   conditioning is written so a reader can see the boundary and why it exists.
5. **Config over magic numbers.** Every dimension, lambda, learning rate, and momentum value lives
   in a config object.

---

## 2. File layout (flat — 5 to 6 files, no packages, no folders)

```
config.py       # dataclass configs: dims, lambdas, LRs, momentum, runtime flags
data.py         # SSv2 dataset, preprocessing (no tubelet dropout in v0.2), dataloader
models.py       # nn.Modules: frozen encoder, bottleneck, coarse/fine flows, generator, B_EMA wrapper
losses.py       # flow-matching primitives + flow_matching_loss, variance_floor, total objective
diagnostics.py  # collapse/bypass probes, each a pure function returning a metrics dict
train.py        # training loop, EMA update, wandb logging, entry point
```

That's the whole repo. No `src/`, no nested packages, no productionization scaffolding. If you want
5 files, fold `diagnostics.py` into `train.py` — but keeping it separate keeps your eyes-on-collapse
easy to find.

Two boundaries worth respecting even in a flat layout:

- `models.py` holds **only nn.Modules** — the things that carry parameters.
- `losses.py` holds **only pure tensor math** — no parameters. The flow predictors are modules
  (→ `models.py`); the flow-matching loss is pure math (→ `losses.py`).

---

## 3. Naming map (single source of truth)

The brief uses terse symbols; code uses descriptive snake_case. Pick one identifier per symbol and
never deviate. Put this table at the top of `config.py`.

| Brief symbol | Code identifier      | Meaning                              |
|--------------|----------------------|--------------------------------------|
| `x`          | `context_clip`       | Input context frames (8 @256)        |
| `E`          | `encoder`            | **Frozen** pretrained V-JEPA 2 ViT-L/16 (shared) |
| `e_t`        | `detailed`           | Current detailed latent (frozen E)   |
| `B`          | `bottleneck`         | Trainable bottleneck module          |
| `c_t`        | `abstract`           | Current abstract latent              |
| `x_{≤t+k}`   | `target_clip`        | Future clip ending at t+k            |
| `B_EMA`      | `target_bottleneck`  | EMA copy of bottleneck (only EMA module) |
| `e_plus`     | `target_detailed`    | Target (stop-grad) detailed latent (frozen E on target clip) |
| `c_plus`     | `target_abstract`    | Target (stop-grad) abstract latent (= c⁺_{t+k}) |
| `F_c`        | `coarse_flow`        | Coarse flow predictor                |
| `F_e`        | `fine_flow`          | Fine flow predictor                  |
| `h_k`        | `horizon_embed`      | Learned horizon embedding (Phase 4)  |
| `c_hat`      | `pred_abstract`      | Predicted future abstract latent     |
| `e_hat`      | `pred_detailed`      | Predicted future detailed latent     |
| `D`          | `frame_generator`    | Frame generator (VAE latent space)   |
| `A`          | `vae_encoder`        | Frozen image VAE encoder             |

> **v0.2:** there is no `online_encoder`/`target_encoder` split — `E` is a single **frozen** module
> used by both branches. Only the **bottleneck** has an EMA copy (`B_EMA`).

Ambiguity between `c_t` / `c_plus` / `c_hat` is the likeliest source of a sign-or-routing bug. This
table is the cheapest defense.

---

## 4. Docstrings & shape contracts

Every function gets a docstring with three parts, in this order:

1. A one-line summary.
2. One or two lines of plain English on **what it does and why it matters**.
3. `Args` / `Returns` with tensor shapes using named dimensions.

Shape notation, defined once and reused everywhere:
`B`=batch, `T`=context frames, `N_e`=detailed tokens, `D_e`=detailed dim,
`N_c`=abstract tokens, `D_c`=abstract dim.

Template:

```python
def compress(detailed):
    """Compress the detailed perceptual latent into the small abstract latent.

    Drops most local appearance and keeps only future-relevant structure. This
    bandwidth squeeze is what stops the abstract state from just copying the detailed one.

    Args:
        detailed: (B, N_e, D_e) online detailed latent.
    Returns:
        abstract: (B, N_c, D_c) abstract latent, with N_c << N_e.
    """
```

During development, guard shapes with cheap asserts behind a `cfg.debug_shapes` flag so they cost
nothing in production:

```python
if cfg.debug_shapes:
    assert detailed.shape[1:] == (cfg.n_e, cfg.d_e), detailed.shape
```

---

## 5. Function granularity

The forward pass should read like the data path in the brief:

```python
with torch.no_grad():
    detailed = encoder(context_clip)              # frozen V-JEPA 2.1
abstract = bottleneck(detailed)                   # trainable
with torch.no_grad():
    target_detailed = encoder(target_clip)        # same frozen encoder
    target_abstract = as_target(target_bottleneck(target_detailed))  # B_EMA, detached
pred_abstract = coarse_flow(abstract)             # (+ horizon_embed in Phase 4)
pred_detailed = fine_flow(detailed, coarse_cond)
```

- **DO** make `flow_matching_loss(...)` one pure function, reused by the coarse, fine, and frame
  losses. They are structurally identical — one function, three call sites.
- **DO** give each diagnostic probe its own function.
- **DON'T** explode `flow_matching_loss` into four public functions; keep noise/tau/interpolant/
  velocity as private helpers inside `losses.py`, exposed as one coherent unit.
- **DON'T** write a 400-line `train_step`. It orchestrates named functions; it doesn't inline them.

Rule of thumb: if you can't name a function after a concept in the brief or a single clear verb,
it's at the wrong granularity.

---

## 6. Stop-gradient, written to be auditable

Do not scatter `.detach()` across the codebase. Centralize it in one named helper.

```python
def as_target(x):
    """Mark a tensor as a stop-gradient target, so it is predicted but never optimized.

    Used for the EMA-branch outputs and for detached conditioning, so every gradient
    boundary lives in one named place instead of scattered .detach() calls.
    """
    return x.detach()
```

Comment each boundary with the collapse it prevents:

```python
# Detach so the fine loss can't backprop into the coarse flow and turn the
# abstract latent into a texture carrier (brief sec. 6).
coarse_cond = as_target(pred_abstract)
```

---

## 7. Config

One flat `@dataclass` hierarchy, no inline literals on the hot path.

```python
@dataclass
class LossConfig:
    lambda_fine: float = 1.0
    lambda_e_reg: float = 0.02
    lambda_c_reg: float = 0.10
```

Push the whole config into `wandb.config` at startup so every run is self-describing and ablations
show up as config diffs.

---

## 8. PEP8 + tooling (enforced, not aspirational)

- **Formatter:** `black` (line length 100).
- **Linter:** `ruff` (covers flake8 + isort + pyflakes in one fast pass).
- **Type hints** on all public signatures. Tensors annotate as `Tensor`; their shapes live in the
  docstring (Python types can't express shapes).
- **Naming:** functions/modules `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- **Pre-commit hook** running `black --check` and `ruff` so style never reaches review.

---

## 9. Comments

Comment the **why**, never the **what** — the code already says what it does. Reference brief
sections on any non-obvious design choice so a reader can jump from a tricky line to its
justification.

---

## 10. Anti-patterns to avoid

- **Premature abstraction.** No base classes until two real things genuinely share behavior.
- **Scattered `.detach()`.** Centralize it (see sec. 6).
- **Clever one-liners on the gradient path.** Readability there is correctness.
- **Magic numbers.** Everything lives in the config object.
- **Config inheritance hell.** Flat dataclasses over deep YAML inheritance.