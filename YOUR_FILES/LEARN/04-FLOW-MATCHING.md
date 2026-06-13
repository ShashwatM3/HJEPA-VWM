# 04 — Flow Matching: How We Actually Predict the Future

> **What you'll understand after this file:** what rectified flow is, why
> the predictor outputs a "velocity" instead of the future latent directly,
> what every tensor in `train_step` is, what adaLN-Zero conditioning means,
> and why condition dropout exists.

---

## 1. The problem with predicting the future directly

The obvious predictor: a network `P(c_t) → ĉ_{t+k}`, trained with MSE
against the true `c_{t+k}`. This has the same flaw as pixel MSE from file
01, just moved into latent space: **the future is a distribution**, and
MSE regression converges to its *mean*. Given `c_t` of a hand mid-reach,
plausible futures include grasp/hover/knock-over — distinct points in
latent space. An MSE regressor outputs their average, which may be a
latent corresponding to *no* coherent future.

What we want instead is a **generative model of the conditional
distribution** `p(c_{t+k} | c_t)`: something that can produce *samples* of
plausible futures, sharp ones, and (implicitly) assign them probability.
The modern tool family for this is diffusion/flow models. We use the
simplest member: **rectified flow** (a.k.a. flow matching with straight
paths — the same formulation behind Stable Diffusion 3 and Flux).

## 2. Rectified flow in one picture

Imagine two clouds of points in latent space: a Gaussian noise cloud
`ε ~ N(0, I)` and the data cloud (real future latents `c⁺`). A flow model
learns a **velocity field**: at every point `z` and time `τ ∈ [0,1]`, an
arrow `v(z, τ)` such that if you start at a noise sample and follow the
arrows from τ=0 to τ=1, you land on a data sample.

Rectified flow makes the boldest possible simplification: the paths are
**straight lines**. Define the interpolation between a noise sample and a
data sample:

```
z_τ = (1 − τ)·ε + τ·c⁺          (τ=0: pure noise → τ=1: pure data)
```

The velocity along this straight path is constant — just the displacement:

```
u = dz_τ/dτ = c⁺ − ε
```

The training objective is then disarmingly simple. Sample a real pair
(ε, c⁺), pick a random τ, construct z_τ, and train the network to output
the straight-line velocity:

```
L_flow = ‖ F_c(z_τ, τ, c_t) − (c⁺ − ε) ‖²
```

That's literally all of `losses.py`'s flow code:

```47:82:losses.py
def interpolate(z_target: Tensor, eps: Tensor, tau: Tensor) -> Tensor:
    """Build the rectified-flow interpolation point.

    Args:
        z_target: (B, N, D) stop-gradient target latent.
        eps: (B, N, D) Gaussian noise sample.
        tau: (B,) flow time, broadcast to (B, 1, 1).
    Returns:
        z: (B, N, D) interpolated latent `(1 - tau) * eps + tau * z_target`.
    """
    tau_b = _broadcast_tau(tau, z_target)
    return (1.0 - tau_b) * eps + tau_b * z_target


def velocity_target(z_target: Tensor, eps: Tensor) -> Tensor:
    """Compute the rectified-flow target velocity.

    Args:
        z_target: (B, N, D) future latent target.
        eps: (B, N, D) Gaussian noise source.
    Returns:
        u: (B, N, D) velocity target `z_target - eps` (constant along the trajectory).
    """
    return z_target - eps


def flow_matching_loss(u_hat: Tensor, u_target: Tensor) -> Tensor:
    """Mean-squared error for coarse, fine, and frame flow matching."""
    return (u_hat - u_target).pow(2).mean()
```

**Wait — isn't this still MSE?** Yes, but on a different object, and that
changes everything. The network sees `z_τ` — a *specific* noisy point. Two
different futures (grasp vs knock-over) paired with different noise
samples produce different `z_τ`'s, and at each one the correct velocity is
well-defined. Averaging happens only where trajectories genuinely overlap
(near τ=0, deep in noise), and the theory guarantees the *learned field's
flow* still transports the noise distribution to the data distribution.
Determinism in the network + randomness in the input = a sampler of a
distribution, not a regressor to its mean. That's the trick.

**Inference** (not used in Phase 1 training, but what the model is *for*):
start at z = ε, integrate `dz/dτ = F_c(z, τ, c_t)` from 0 to 1 (e.g. a few
Euler steps). Because training encouraged straight paths, very few steps
suffice — the efficiency selling point of rectified flow over classic
diffusion. Different ε → different plausible future. Multimodality
preserved.

**Why flow matching and not DDPM-style diffusion?** Same family, but:
simpler math (one interpolation formula, no noise-schedule zoo), better
few-step sampling (straight paths), and one consistent recipe reusable for
the fine flow (Phase 2) and frame generator (Phase 3). One mental model,
three uses.

## 3. The predictor `F_c`: a conditional DiT

`F_c` must compute velocity *given* three things: where we are (`z_τ`),
what time it is (`τ`), and what the present looks like (`c_t` — the
conditioning, the entire reason this is a *prediction* model). The
architecture is a small **DiT** (Diffusion Transformer): 6 transformer
blocks, dim 256, 8 heads, with two non-obvious mechanisms:

### 3a. Conditioning by concatenation

```
x = concat([z_τ (32 tokens), c_t (32 tokens)])  → 64-token sequence
```

The noised future tokens and the conditioning tokens are processed as one
sequence; self-attention lets every noised token read every conditioning
token. After the blocks, only the first 32 tokens (the z positions) are
kept as the velocity output. Simple, symmetric, and it gives token-level
access (each future slot can attend to each present slot) rather than
squeezing `c_t` through a single global vector.

### 3b. Time conditioning via adaLN-Zero

`τ` is a scalar — too thin to concatenate as a token. Instead:

1. **Sinusoidal embedding** (`_timestep_embedding`): τ → 256-dim vector of
   sines/cosines at geometrically spaced frequencies — the standard
   "Fourier features" trick (same as transformer position encodings) that
   lets an MLP resolve both coarse and fine differences in τ.
2. **A small MLP** processes that embedding.
3. **adaLN** (adaptive LayerNorm): in each block, the time embedding
   generates six vectors — `shift`, `scale`, `gate` for the attention
   sub-block and again for the MLP sub-block. The block computes:

```
h = LayerNorm(x) · (1 + scale) + shift     ← time modulates normalization
x = x + gate · Attention(h)                ← time gates the residual
```

So time doesn't enter as data; it *modulates how every block processes
data*. Empirically (the DiT paper), this beats concatenation for
diffusion-style conditioning.

4. **The "-Zero" part** — the initialization story:

```258:260:models.py
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
```

The layer producing (shift, scale, gate) starts at **exactly zero**. Then
gate = 0 → every residual branch contributes nothing → **each block is the
identity function at step 0**. The whole 6-block network starts as
(approximately) a fixed map, and training *gradually opens the gates*,
letting each block earn its contribution. This is the single most careful
initialization in the codebase, and the reason: flow training feeds the
network pure-noise inputs at small τ; a randomly initialized deep stack
amplifies that noise through 6 blocks and produces wild early gradients.
Identity-at-init makes step 0 boring — and boring is stable. (Connect this
to Run 1: the explosion still happened at *peak LR*, months of steps later
— init protects the start, the LR schedule must protect the middle. File
07 and 10.)

### 3c. Condition dropout

```346:353:models.py
        if self.training or condition_drop is not None:
            if condition_drop is None:
                condition_drop = (
                    torch.rand(abstract.shape[0], device=abstract.device)
                    < self.cfg.condition_dropout
                )
            null = self.null_condition[None].expand_as(abstract)
            abstract = torch.where(condition_drop[:, None, None], null, abstract)
```

With probability 0.10 per sample during training, the true `c_t` is
replaced by a learned `null_condition` parameter. Two reasons:

- **Robustness/regularization:** the predictor must sometimes model the
  *unconditional* future distribution, so it cannot become brittle about
  its conditioning.
- **It's the classifier-free-guidance trick** from image diffusion: a
  model trained both with and without conditioning can, at inference,
  extrapolate *toward* the conditioned prediction for sharper/more
  faithful samples. We keep the door open even if Phase 1 never uses it.

Note the diagnostics pass an explicit all-False mask (`_no_drop`) so
measured baselines always use the real `c_t` — you never want your
acceptance metric randomly contaminated by dropped conditions.

## 4. One training step, end to end

From `train.py::train_step`, annotated:

```
context_clip, target_clip                    # (B,8,3,256,256) each, from dataloader
e_t  = encoder(context_clip)        no_grad  # (B,1024,1024)
c_t  = bottleneck(e_t)              GRAD     # (B,32,256)   ← online branch
e⁺   = encoder(target_clip)         no_grad
c⁺   = target_bottleneck(e⁺)        no_grad + detach        ← EMA target branch

ε    ~ N(0,I)  like c⁺                       # fresh noise every step
τ    ~ U(0,1)  per-sample                    # every step trains all noise levels
z_τ  = (1−τ)·ε + τ·c⁺
u    = c⁺ − ε                                # target velocity
û    = F_c(z_τ, τ, c_t)             GRAD     # predicted velocity

loss = ‖û − u‖² + 0.10 · L_var(c_t)
loss.backward()                              # grads flow to F_c AND through c_t to B
clip, maybe-skip, optimizer.step()           # file 07
EMA update of target_bottleneck              # file 05
```

Things worth internalizing:

- **`τ ~ U(0,1)` per sample** means every batch trains the model at all
  noise levels simultaneously — there is no curriculum over τ.
- **Gradients reach the bottleneck only through `c_t`'s conditioning
  path.** The target side is sealed (detach + no_grad + EMA). So the
  bottleneck is literally optimized to "be a useful conditioning signal" —
  that *is* its training signal.
- **L_flow's typical scale:** with c⁺ roughly unit-scale (LayerNorm) and ε
  unit Gaussian, ‖c⁺ − ε‖² per element ≈ 2 for an ignorant model. Run 1's
  L_flow hovering near ~1.0–1.1 means "better than ignorant, far from
  perfect" — which is why we judge against *baselines*, not raw values
  (file 06).

## 5. Why predict velocity and not the future directly — recap table

| | Direct regression `P(c_t)→ĉ⁺` | Rectified flow |
|---|---|---|
| Output on ambiguous future | mean of futures (possibly invalid) | a sample of one plausible future |
| Models distribution? | no, point estimate | yes, implicitly |
| Extra inputs needed | none | noise ε, time τ |
| Inference cost | 1 forward | few integration steps |
| Loss | MSE on latent | MSE on velocity (well-posed pointwise) |

The cost — extra inputs, integration at inference — is the price of
honesty about uncertainty. For a world model meant to imagine futures,
that honesty is the product.

## 6. Questions to test yourself

1. Why is the velocity target `c⁺ − ε` constant along the whole path?
   *(Straight-line interpolation: derivative of `(1−τ)ε + τc⁺` w.r.t. τ.)*
2. What happens at τ=0 and τ=1 in `z_τ`? *(Pure noise / pure target — the
   model learns to denoise from any point in between.)*
3. Why does zero-initializing the modulation layer make blocks identity?
   *(gate=0 kills every residual branch's contribution.)*
4. Where exactly do gradients from L_flow enter the bottleneck? *(Through
   the conditioning argument `abstract` of `F_c` — the only grad path.)*
5. Why must diagnostics disable condition dropout? *(Baseline ratios would
   be randomly corrupted by samples where the model was blindfolded.)*
