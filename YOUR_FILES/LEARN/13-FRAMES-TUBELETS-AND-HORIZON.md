# Frames, Tubelets, And Horizon

## Current Pipeline (`horizon_k = 4`)

**1. Pick one video.**

We choose a starting point, call it frame `0`.

**2. Build the context clip.**

We take 8 sampled frames, every 2 frames:

```text
context frames = 0, 2, 4, 6, 8, 10, 12, 14
```

So the context clip is:

```text
8 sampled frames = 4 V-JEPA tubelets
```

because V-JEPA groups every 2 frames into 1 tubelet.

**3. Build the target clip.**

Current `horizon_k = 4`, so the target starts 4 frames later:

```text
target frames = 4, 6, 8, 10, 12, 14, 16, 18
```

The target clip is also:

```text
8 sampled frames = 4 V-JEPA tubelets
```

**4. V-JEPA encodes the context.**

Input:

```text
8 context frames
```

V-JEPA turns that into:

```text
4 temporal tubelet groups × 16 × 16 spatial patches = 1024 tokens
```

So:

```text
context frames -> context tubelet tokens e_t
```

**5. Bottleneck compresses context tokens.**

Input:

```text
1024 context tubelet tokens
```

Output:

```text
c_t = 32 abstract tokens
```

This is the model’s compressed understanding of the context clip.

**6. V-JEPA encodes the target.**

Input:

```text
8 target frames
```

V-JEPA turns it into:

```text
1024 target tubelet tokens
```

So:

```text
target frames -> target tubelet tokens e_plus
```

**7. EMA bottleneck compresses target tokens.**

Input:

```text
1024 target tubelet tokens
```

Output:

```text
c_plus = 32 target abstract tokens
```

This is the thing we want the model to predict.

**8. Flow model learns prediction.**

Input condition:

```text
c_t = abstract latent of context clip
```

Target:

```text
c_plus = abstract latent of target clip
```

So the model learns:

```text
given the context clip's 4 tubelets,
predict the target clip's 4 tubelets
```

More precisely:

```text
given 4 context tubelets starting at frame 0,
predict 4 target tubelets starting at frame 4
```

That is why there is high overlap, as observed via the metrics from the previous run.

---

## Proposed `horizon_k = 12`

Only step 3 changes.

Context stays:

```text
context frames = 0, 2, 4, 6, 8, 10, 12, 14
```

Target becomes:

```text
target frames = 12, 14, 16, 18, 20, 22, 24, 26
```

So now the model learns:

```text
given 4 context tubelets starting at frame 0,
predict 4 target tubelets starting at frame 12
```

Simple difference:

```text
current:  target starts 4 frames later
proposed: target starts 12 frames later
```

Everything else stays the same.