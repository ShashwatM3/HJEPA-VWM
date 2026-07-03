# Fixed-Position Decoder Proposal

**Status:** implemented in `models.Decoder` and covered by `tests/test_decoder.py`.
**Investigation:** 011.
**Purpose:** record the proposed decoder architecture change for making reconstruction depend more
strongly on `c`.

## 1. Problem

Since investigation 006, reconstruction loss has followed nearly the same shape across runs:

- fast drop during the reconstruction warmup;
- plateau around the same old `relative_mse` floor;
- small movement under decoder-size, reconstruction-weight, SIGReg, and residual-prediction changes;
- small `L_recon_chat - L_recon_cplus` gap even when the prediction branch is only tying copy.

The most likely explanation is not a detach or optimizer bug. The reconstruction path is wired so
gradients reach the decoder and bottleneck. The more likely problem is architectural: the decoder can
learn a strong output-token prior without using much information from `c`.

## 2. Current decoder loophole

Current reconstruction decoder:

```text
c                -> kv projection -> memory
learned queries  -> cross-attend to memory -> MLP -> e_hat
```

The decoder owns a trainable table:

```text
queries: 1024 x decoder_dim
```

There is one learned vector per V-JEPA output tubelet position. That table can become a stored
per-position V-JEPA feature template:

```text
learned output query -> decoder MLP -> average/typical e feature for that tubelet
```

Cross-attention into `c` still exists, but the learned queries give the decoder an unconditional
content path. If that unconditional path explains a large fraction of the target variance, all runs
can show the same reconstruction trajectory even when `c` changes.

## 3. Goal

Keep the useful part:

```text
the decoder knows which output tubelet position it is reconstructing
```

Remove the dangerous part:

```text
the decoder has a trainable content vector for every output tubelet
```

The intended rule is:

```text
position tells the decoder where to write;
c tells the decoder what to write.
```

## 4. Proposed architecture

Replace learned output queries with fixed positional queries.

The decoder still maps:

```text
c:     (B, 32, 256)
e_hat: (B, 1024, 1024)
```

but the output-token starting point changes.

### 4.1 Fixed tubelet position codes

V-JEPA produces 1024 detailed tokens:

```text
4 temporal positions x 16 rows x 16 columns = 1024 tubelets
```

Create a fixed 3D positional code for each tubelet:

```text
pos[t, y, x] -> decoder_dim
```

This can be sinusoidal or another deterministic non-trainable encoding. It should be registered as a
buffer, not as an `nn.Parameter`.

The fixed position code is allowed to identify the output location. It is not allowed to become a
learned content template.

### 4.2 First decoder step

Project `c` into decoder memory:

```text
memory = kv_proj(c)          # (B, 32, decoder_dim)
pos    = fixed_pos           # (1024, decoder_dim)
```

Use position only as the attention query:

```text
query_p = W_q(pos_p)
key_j   = W_k(memory_j)
value_j = W_v(memory_j)

attention_weight[p, j] = softmax_j(query_p dot key_j)
hidden_p = sum_j attention_weight[p, j] * value_j
```

Important point:

```text
hidden starts as a weighted sum of c-derived values.
```

The fixed position code is not added into `hidden` before the output head.

### 4.3 Deeper blocks

For later blocks, the decoder can keep using fixed position to decide how each output token should
read from `c`, but the residual hidden state must remain content derived from `c`.

One acceptable block pattern:

```text
query_input = norm(hidden) + fixed_pos
attn_out    = cross_attn(query_input, memory, memory)
hidden      = hidden + attn_out
hidden      = hidden + mlp(norm(hidden))
```

In PyTorch `MultiheadAttention`, the query controls attention weights. The attention output is a
weighted sum of values, and the values come from `c`. Since `fixed_pos` is not added to `hidden`, it
does not directly become output content.

### 4.4 Output head

The output head remains simple:

```text
e_hat = out_proj(out_norm(hidden))
```

The output projection and MLP are trainable. They can learn how to convert `c`-derived hidden states
into V-JEPA feature vectors, but they no longer receive a trainable per-position content table.

## 5. How fixed queries still learn where to read

The fixed query does not learn. The learned parts around it learn.

Attention score:

```text
score(position p, slot j) = W_q(pos_p) dot W_k(c_j)
```

What is fixed:

- `pos_p`, the identity of output tubelet `p`.

What is trainable:

- `W_q`, the projection that interprets a tubelet position as an attention query;
- `W_k`, the projection that interprets a `c` slot as an attention key;
- `W_v`, the projection that turns a `c` slot into readable content;
- the bottleneck `B`, which decides what each `c` slot contains;
- the decoder MLP and output projection.

Training gradients can therefore teach:

```text
for output tubelet p, use this attention pattern over the 32 c slots
```

Example behavior that could emerge:

- early temporal tubelets read more from slots carrying present object/layout state;
- later temporal tubelets read more from slots carrying motion-sensitive state;
- spatial regions containing hands/objects read from different slots than background regions.

None of those assignments are hard-coded. They emerge because the reconstruction loss rewards
routing that produces the correct frozen V-JEPA feature.

The exact sentence to keep in mind:

```text
The fixed position does not learn; the attention projections learn how each fixed position reads c.
```

## 6. Why this should strengthen dependence on c

With the current decoder, this path exists:

```text
learned query for tubelet p -> output feature for tubelet p
```

With the proposed decoder, the main path becomes:

```text
fixed position p -> attention weights over c -> weighted c values -> output feature for tubelet p
```

If `c` carries little information, the decoder loses the easy way to emit a detailed per-position
template. It may still learn small global defaults through biases and shared weights, but it no
longer owns 1024 trainable content vectors.

This should make `L_recon_present` a more honest measure of information in `c`.

It should also make `L_recon_chat - L_recon_cplus` more meaningful. If predicted future `c_hat` is
bad, decoding from it should become visibly worse than decoding from true future `c_plus`.

## 7. What this proposal is not

### Not a VITA-style flow lift yet

A direct flow from `c` to `e` is not the first fix. Our dimensional jump is large:

```text
c: 32 x 256 = 8,192 numbers
e: 1024 x 1024 = 1,048,576 numbers
```

A flow still needs a way to lift `c` into an `e`-shaped object. If that lift uses trainable
per-position content queries, the same loophole returns with more machinery.

### Not mutual-information maximization

An MI or contrastive objective is extra loss design. The current problem has a simpler direct cause:
the decoder has an unconditional per-position template path. Remove that first.

### Not batch-mean residualization

Residualizing `e` against a saved training-set mean is possible:

```text
D(c) -> e - mu_e
e_hat = mu_e + D(c)
```

but it adds a stored dataset statistic and a new inference contract. It is secondary to removing the
learned-query escape route.

## 8. Expected metric behavior

This change may make reconstruction loss look worse at first. That is acceptable.

Expected if the diagnosis is correct:

- initial `L_recon_present` may be higher or descend more slowly;
- late `L_recon_present` should become more sensitive to whether `c` actually carries information;
- `L_recon_chat - L_recon_cplus` should widen when `c_hat` is worse than true `c_plus`;
- decoder capacity changes should matter less than `c` quality;
- a present-only reconstruction run should reveal whether `B` can encode enough detail when the
  decoder cannot lean on learned output templates.

Failure modes:

- if `L_recon_present` stays high and `c` rank/cosine do not improve, the bottleneck may not be able
  to carry enough detailed information under this objective;
- if `L_recon_present` returns to the same floor with tiny `chat-cplus` gap, another unconditional
  path remains;
- if training becomes unstable, the decoder may need a gentler warmup or smaller LR, but that should
  be treated as optimization tuning, not a new architectural idea.

## 9. Implementation notes for a later patch

This note began as a design proposal. The implemented patch is intentionally narrow:

- replace `Decoder.queries` with fixed positional codes registered as a buffer;
- ensure no `nn.Parameter` exists with shape `(N_ctx, decoder_dim)` for output-token content;
- use position only as the cross-attention query, not as residual output content;
- keep the decoder input/output shapes unchanged;
- keep `--decoder-dim`, `--decoder-blocks`, `--lambda-recon`, `--lambda-recon-pred`, and
  `--recon-loss-mode` behavior unchanged;
- do not resume old decoder checkpoints across this architecture change;
- update tests to assert that fixed position codes are buffers, not parameters;
- keep smoke tests that prove reconstruction gradients still reach `D` and `B`.

The smallest implementation target:

```text
Decoder(c) still returns (B, 1024, 1024),
but output content must come from c-derived values, not learned per-token queries.
```
