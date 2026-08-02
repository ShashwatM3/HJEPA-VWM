# Observations — coarse-flow architecture and metric audit

## Short answer

Your colleague's **off-manifold concern is plausible**, but increasing `lambda_recon` is not the
right intervention. In the frozen run it would do literally nothing to `F_c`. Your own instinct is
closer to the immediate problem: we should audit both `F_c` and the meaning of the coarse-copy
gate. The ratio is arithmetically correct, but it compares the learned vector field against a
direct target-space guess that is given the sampled noise by the evaluator. That makes the gate
useful, but not a clean measurement of whether the flow model has learned all it reasonably can.

## 1. Your colleague's hypothesis

### What increasing reconstruction would actually do

There are two different reconstruction weights, and they answer different questions.

`lambda_recon` trains the present code by decoding `c_t` back to the present encoder features. It
does not decode the predicted future. In Investigation 021 it was recorded as `1.0`, but `B` and
`D` were frozen and present reconstruction was excluded from the optimized loss. Raising it to
`10`, `100`, or any other value would therefore produce the same `F_c` updates: none.

`lambda_recon_pred` is the weight that matches your colleague's idea. It decodes the predicted
future code and compares it with future encoder features. That gradient can tell the predictor,
“produce a residual whose endpoint still decodes like a real future.” However, the current
`fc_only` contract explicitly requires `lambda_recon_pred=0`, and the training path only enables
that term when the bottleneck is trainable. Using it with fixed `B` and `D` would require an
intentional new optimization mode, not a configuration-only weight change.

The historical prediction-reconstruction run does not settle whether this could work now. In that
old run the decoder was nearly blind: correct, shuffled, and predicted codes decoded similarly.
The current decoder is different. In run `r0s6ouwd`, correct-future reconstruction was `0.118823`,
shuffled-code reconstruction was `0.538133`, and predicted-code reconstruction improved to about
`0.174`. The current decoder contains a meaningful signal that the old decoder did not.

So prediction-side reconstruction is now a legitimate later test. Simply turning present
reconstruction “way higher” is not.

### Is the off-manifold concern real?

The exact residual is not intrinsically in a wrong space. By construction,

```text
Delta = c_plus - c_t
c_t + Delta = c_plus
```

The exact endpoint is therefore a real bottleneck code. The problem begins when the predictor
produces an approximate residual. Then `c_t + Delta_hat` need not lie in the region occupied by
real codes.

Run 021 shows a one-step endpoint mismatch consistent with this concern, not proof of an
off-manifold code. The fixed decoder's loss was about `0.1188` on the real future code and `0.1741`
on the late one-step predicted code. The predicted code does not decode as well as the true code,
but it moved substantially closer from initialization.

A large prediction-reconstruction weight could pull the endpoint toward decoder-valid content.
It would still not be a true latent-density or manifold constraint: a decoder can assign a good
output to latent points that the bottleneck itself never produces. A very large weight can also
make the predictor exploit decoder directions while sacrificing the flow target. If we later use
this loss, we should keep `B` and `D` fixed, measure the two gradient scales, and judge both latent
endpoint error and decoded-future error.

My verdict on the colleague's proposal is therefore: **the concern is real; the literal
`lambda_recon` prescription is wrong; `lambda_recon_pred` is plausible but not yet the cleanest
next move.**

## 2. Your hypothesis

### Is the copy ratio wrong?

The code performs the stated arithmetic correctly.

For residual prediction, the flow target is `u = Delta - eps`. The zero-residual baseline is
entered as `u_copy = -eps`. Their difference is exactly `Delta`, so
`coarse_copy_loss = mean(Delta^2)`. The batch-mean baseline works the same way. The logged ratios
are exact quotients of model loss by those losses. There is no sign error, swapped tensor, or bad
division in this calculation.

The subtle issue is what the comparison means. The numerator asks `F_c` to infer `Delta - eps`
from the noisy mixture `z`, time `tau`, and condition `c_t`. The denominator directly inserts a
guess for `Delta` and uses the evaluator's exact sampled `eps` to translate that guess into a
velocity. It does not construct a realizable vector field that must infer the same information
from `z`, `tau`, and `c_t`.

That difference creates a useful sanity check. Suppose `Delta` and the scaled noise are independent,
centered Gaussians with the same variance, and suppose the condition supplies no useful temporal
information. The best possible predictor from `z` and `tau` then has conditional error

```text
variance / (tau^2 + (1 - tau)^2).
```

Averaging over uniform `tau` and dividing by the batch-mean residual loss gives

```text
integral[0,1] 1 / (tau^2 + (1 - tau)^2) d tau = pi / 2 = 1.5708.
```

Run 021's late batch-mean ratio was `1.575406`. The agreement is not a proof because the real
residuals are not ideal Gaussians. It is nevertheless a strong clue: `F_c` may have learned almost
exactly the **unconditional denoising solution** while extracting little predictive information
from `c_t`. That would produce a repeatable floor even when optimization is healthy.

The gate is therefore not useless. It still asks whether the final error is better than directly
guessing no change or the population mean. But the current one-step velocity ratio cannot, by
itself, distinguish “bad architecture” from “good unconditional flow model with unused or
uninformative conditioning.” We should stop treating it as the only number that matters.

### Is `F_c` architecturally suspicious?

Yes, but not for the same reason as the old reconstruction bottleneck.

The old bottleneck projected down before enough mixing, so it discarded information before the
model could decide what to preserve. The current `F_c` does not do that. It keeps the full
`D_c=512` width, adds slot and stream identities, concatenates 64 noisy-residual tokens with 64
conditioning tokens, and mixes all 128 tokens through six transformer blocks. There is no early
dimension-reduction smoking gun here.

There is a different structural concern at the output. After the six blocks, `F_c` takes the noisy
stream and returns a final `LayerNorm` directly as the predicted velocity. It has no learned final
velocity projection. The normalization removes each token's hidden mean and scale before applying
one shared learned affine transform. The true velocity can have example-, token-, and
time-dependent magnitude, so the network must encode magnitude indirectly in the normalized
direction instead of emitting an unconstrained vector. This is a real restriction, even though it
is not yet proven to be the observed floor. A normal flow transformer usually has a dedicated
final mapping from hidden state to velocity, often after a conditioned normalization.

The conditioning path is the second concern. Concatenated self-attention is valid, but it gives the
network an easy route to solve the denoising part from `z` and `tau` while mostly ignoring `c_t`.
The 10% condition dropout makes unconditional competence useful. The near-`pi/2` plateau makes
ignored conditioning a more immediate hypothesis than insufficient width or depth.

The next action should therefore be an audit, not another long run:

1. Evaluate the same fixed `(z, tau, eps)` samples with real, shuffled, and null `c_t`. This
   isolates whether the trained predictor uses conditioning at all.
2. Integrate the learned flow from noise to an endpoint `Delta_hat`, then compare endpoint MSE
   against zero residual and batch mean. This compares actual inference procedures instead of an
   oracle velocity construction.
3. Compare target velocity means and norms with `F_c` output means and norms across `tau`. This
   tests whether the final normalization/no-head parameterization is clipping a degree of freedom
   the target needs.

**Decision:** do not raise `lambda_recon` and do not launch a larger model yet. First determine
whether `F_c` ignores `c_t`, whether its integrated endpoint beats direct residual baselines, and
whether the output normalization restricts target scale. If conditioning is unused despite being
predictive, the next architectural experiment should fix the conditioning/output interface. If
conditioning helps but endpoints leave the real-code region, fixed-decoder
`lambda_recon_pred` becomes the principled larger test.
