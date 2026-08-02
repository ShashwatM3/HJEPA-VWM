# Appendix — Parameter and memory formulas

All counts below include trainable affine biases and normalization parameters. Buffers are excluded.
Frozen encoder parameters are listed separately from the trainable Phase 1 stack.

## Linear and attention primitives

For width `D`:

```text
Linear(A→B) = A*B + B
affine LayerNorm(D) = 2D
non-affine LayerNorm(D) = 0
PyTorch MHA(D) = 3D²+3D + D²+D = 4D²+4D
MLP(D→4D→D) = (4D²+4D)+(4D²+D) = 8D²+5D
```

Multi-head count does not change dense projection parameter count as long as total width stays fixed.

## One ConvNeXt block at width `M`

```text
depthwise 7×7 conv = 49M weights + M bias = 50M
LayerNorm          = 2M
M→4M              = 4M²+4M
4M→M              = 4M²+M
total              = 8M²+57M
```

There are two shipped blocks.

## One bottleneck latent block

Components:

```text
sharpened cross-attention projections + scalar
self-attention
latent MLP
affine pre-normalizations
```

Combined exact count:

```text
16M² + 19M + 1
```

There are three shipped blocks. The `+1` is learned logarithmic attention scale.

## Full bottleneck

Variables:

```text
D_e detailed width
N_e detailed token count
M   internal width
N_c abstract query count
D_c external abstract width
C   ConvNeXt blocks = 2
L   latent blocks = 3
```

Formula:

```text
input projection       D_e*M + M
ConvNeXt               C*(8M²+57M)
token KV prep          M²+M
detailed position      N_e*M
queries                N_c*M
latent processor       L*(16M²+19M+1)
abstract projection    0 if M=D_c else M*D_c+D_c
final LayerNorm        2D_c
```

So:

```text
P_B =
 D_e*M+M
+C(8M²+57M)
+M²+M
+N_e*M
+N_c*M
+L(16M²+19M+1)
+I[M≠D_c](M*D_c+D_c)
+2D_c
```

### Default V-JEPA breakdown

`D_e=N_e=1024`, `M=D_c=256`, `N_c=32`:

| Component | Parameters |
|---|---:|
| input projection | 262,400 |
| two ConvNeXt blocks | 1,077,760 |
| token KV prep | 65,792 |
| learned detailed position | 262,144 |
| queries | 8,192 |
| three latent blocks | 3,160,323 |
| abstract projection | 0 |
| final norm | 512 |
| **total** | **4,837,123** |

### Default frame-encoder breakdown

`D_e=768,N_e=2048,M=D_c=256,N_c=32`:

| Component | Parameters |
|---|---:|
| input projection | 196,864 |
| two ConvNeXt blocks | 1,077,760 |
| token KV prep | 65,792 |
| learned detailed position | 524,288 |
| queries | 8,192 |
| three latent blocks | 3,160,323 |
| abstract projection | 0 |
| final norm | 512 |
| **total** | **5,033,731** |

The frame model saves 65,536 input-projection parameters but adds 262,144 position parameters, a net
increase of 196,608.

## One AdaLN-Zero flow block

At width `D`:

```text
MHA                    4D²+4D
MLP                    8D²+5D
time modulation D→6D   6D²+6D
two non-affine norms   0
total                 18D²+15D
```

Zero initialization changes starting values, not parameter count.

## Full coarse flow

For `N=N_c,D=D_c,L=6`:

```text
null condition    N*D
slot position     N*D
two type codes    2D
time MLP          8D²+5D
blocks            L*(18D²+15D)
final norm        2D
```

At `N=32,D=256`:

| Component | Parameters |
|---|---:|
| null condition | 8,192 |
| slot position | 8,192 |
| two type codes | 512 |
| time MLP | 525,568 |
| six blocks | 7,100,928 |
| final norm | 512 |
| **total** | **7,643,904** |

## One decoder block

At decoder width `D`:

```text
query LayerNorm   2D
cross MHA         4D²+4D
MLP LayerNorm     2D
MLP               8D²+5D
total             12D²+13D
```

## Full decoder

For abstract width `D_c`, decoder width `D_d`, detailed width `D_e`, and `L=2`:

```text
KV projection        D_c*D_d+D_d
initial nonaffine LN 0
initial MHA          4D_d²+4D_d
fixed position       0 parameters
decoder blocks       L*(12D_d²+13D_d)
final LayerNorm      2D_d
output projection    D_d*D_e+D_e
```

At `D_c=D_d=256`:

| Component | V-JEPA | frame encoder |
|---|---:|---:|
| KV projection | 65,792 | 65,792 |
| initial MHA | 263,168 | 263,168 |
| two blocks | 1,579,520 | 1,579,520 |
| final norm | 512 | 512 |
| output projection | 263,168 | 197,376 |
| **total** | **2,172,160** | **2,106,368** |

## Complete trainable formula

```text
P_trainable = P_B + P_Fc + P_D
P_bundle = P_trainable + P_B_EMA
P_B_EMA = P_B
```

Frozen encoder is not included.

## Parameter storage thought experiment

For default V-JEPA:

```text
trainable parameters = 14,653,187
EMA parameters       = 4,837,123
frozen encoder       = 325,971,328
```

Assuming FP32 trainable/EMA parameters:

| State | Approximate binary MiB |
|---|---:|
| trainable weights | 55.9 |
| trainable gradients | 55.9 |
| two FP32 Adam moments | 111.8 |
| EMA bottleneck | 18.5 |
| frozen V-JEPA weights if BF16 | 621.7 |

These are simple scalar-count products. Actual GPU/checkpoint memory differs because:

- some weights may remain FP32 under autocast;
- optimizer implementation/serialization adds metadata;
- frozen model dtype/loading can differ;
- activations and allocator workspaces dominate at times;
- checkpoints do not include gradients or frozen encoder;
- PyTorch serialization adds container overhead.

Use actual CUDA peak and file size for operational claims.

## Tensor memory formulas

For dtype bytes `q`:

```text
raw clip           = T*3*H*W*q
detailed features  = B*N_e*D_e*q
abstract features  = B*N_c*D_c*q
cross-attn logits  = B*heads*N_c*N_e*q
flow-attn logits   = B*heads*(2N_c)^2*q
feature mean       = N_e*D_e*4
whitener           = (D_e+2D_e²)*4 + 1 boolean
```

These do not count backward saved tensors.

## FLOP-shape intuition

No exact hardware FLOP claim is made, but leading attention terms are:

```text
bottleneck cross score/value  O(B*N_c*N_e*M)
bottleneck latent self-attn   O(B*N_c²*M)
flow self-attn/block          O(B*(2N_c)²*D_c)
decoder cross-attn/block      O(B*N_e*N_c*D_d)
```

Thus frame encoders roughly double cross-attention sequence work versus V-JEPA; increasing `N_c`
raises flow attention quadratically; increasing width raises projection/MLP work quadratically.
