# Fast-repetition flashcards

Read the left side, cover everything after `→`, answer, then reveal.

## Shapes and identity

- Raw batch → `(64,8,3,256,256)`.
- V-JEPA detailed → `(64,1024,1024)`.
- SigLIP/DINO detailed → `(64,2048,768)`.
- Abstract → `(64,32,256)`.
- V-JEPA lattice → `4×16×16`.
- Frame lattice → `8×16×16`.
- DINO special tokens removed → one CLS + four registers.
- Default abstract scalars/video → 8,192.
- V-JEPA scalar compression → 128:1.
- Frame-encoder scalar compression → 192:1.
- Runtime tuple → `E,B,B_EMA,F_c,D`.
- Trainable modules → `B,F_c,D`.
- EMA module → `B` copy only.
- Current decoder output → detailed encoder features.
- Current fine flow → absent.
- Horizon embedding → absent.

## Time/data

- Context indices → `a+[0,2,4,6,8,10,12,14]`.
- Default target → `a+[4,6,8,10,12,14,16,18]`.
- Default shared frames → 6.
- `k=12` shared frames → 2.
- first even disjoint horizon → 16.
- transform version → `raw-rgb-resize-crop-jitter-v2`.
- jitter range → `[0.6,1.4]`.
- SSv2 tiny exact counts → 4002/348.
- Full SSv2 exact counts → 168913/24777.
- EGO tiny exact counts → 4000/350.
- tiny batches/epoch at64 → 62.
- tiny 15k epochs → 241+58 batches.
- full 15k epochs → about 5.684.

## Modules

- Bottleneck internal width default → 256.
- selected experiment width → 512.
- Bottleneck local blocks → two ConvNeXt.
- Latent blocks → three.
- Cross heads → eight.
- Query init → orthogonal unit rows.
- Bottleneck pos init → trunc-normal std .5.
- cosine temperature init → .07.
- flow blocks/heads → six/eight.
- flow sequence length → 64.
- condition dropout → .10/example.
- decoder width/depth/heads → 256/2/8.
- decoder axis position split → 85/85/86.
- V-JEPA B count M256 → 4,837,123.
- frame B count M256 → 5,033,731.
- flow count → 7,643,904.
- V-JEPA D count → 2,172,160.
- frame D count → 2,106,368.
- V-JEPA trainable total → 14,653,187.

## Training

- steps/warmup → 15,000/1,500.
- peak LR B/Fc/D → 1e-4/2e-4/1e-4.
- Adam betas → .9/.95.
- weight decay → .05 eligible matrices.
- global clip → .5.
- skip threshold → 150.
- warning threshold → grad30 and flow loss1.
- AGC B/Fc/D → .2/.1/.2.
- EMA start/end/denominator → .996/.9999/105k.
- LR scale step0 → 1/1500.
- LR scale step1500 → 1.
- LR scale step7500 → .586824.
- recon/SIG ramp step1000 → .5.
- log/diag/ckpt cadence → 50/500/2500.
- 15k log/diag/ckpt counts → 300/30/6.
- default example presentations → 960,000.

## Losses and gates

- default active optional regularizer → variance floor at .1.
- variance target → population std1.
- SIG projections/rows/beta → 128/512/1.
- mean tracker momentum → .99.
- whitening epsilon/clips → 1e-4/12,800.
- rank gate → greater than60/256.
- cross-video cosine guide → below.5.
- copy-ratio gate → ≤.70.
- batch-mean gate → ≤.50.
- run37 verdict → healthy representation, no predictor.
- copy loss trained? → no.
- residual-mode copy → zero residual.
- present recon gradients → B and D.
- predicted recon gradients → B,Fc,D.
- target gradients → none.

## Reproducibility/MLOps

- checkpoint schema → `hjepa-phase1-checkpoint-v2`.
- provenance schema → `hjepa-run-provenance-v1`.
- checkpoint step meaning → next update.
- train order seed → `seed+epoch*1,000,003`.
- step seed → `seed*1,000,003+step`.
- diagnostic seed → `seed*1,000,003+900,001`.
- exact resume needs → model+EMA+optimizer+buffers+sampler+RNG+identities.
- W&B checkpoint bytes by default → not uploaded.
- active checkpoint policy → unique `/workspace/ckpt/<tag>`.
- remote completion → exit0 + final step/W&B + checkpoint hash + identities + verdict.
