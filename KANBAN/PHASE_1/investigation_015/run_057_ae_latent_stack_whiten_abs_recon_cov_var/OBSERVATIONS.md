# Observations — run 057 `ae_latent_stack_whiten_abs_recon_cov_var`

**W&B:** `cdvp6hou`  
**State:** finished, full 15,000-step schedule  
**Mode:** present-reconstruction-only  
**Verdict:** **Strong present representation**

## Reading Cycle B

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive and correctly wired? | PASS | Prediction stayed off; whitening stayed on; SIGReg scale stayed zero; reconstruction reached full warmup. Zero skipped updates, NaN flags, or instability warnings. |
| Q2 | Is `c_t` alive and source-specific? | PASS | Final std `1.117`, dead fraction `0`, cross-video cosine `0.0585`. |
| Q3 | Is the latent rich and are slots distinct? | PASS | Effective rank `208.22/256`; centered slot rank `30.71/32`. Neither contracted late. |
| Q4 | Does reconstruction use the correct code? | PASS | Present `0.71285`, shuffled `0.93966`, gap `0.22681`; about `78.0%` of learned improvement is code-conditioned. |
| Q5 | Do geometry and content cooperate? | PASS | Rank and slot diversity held while reconstruction fell. Removing SIGReg improved reconstruction and gap relative to run 056 without losing geometry. |
| Q6 | What did the ablation settle? | Covariance is operative | Run 057 differs from run 056 only by `lambda_sigreg=0`. Rank rose slightly, reconstruction improved, attention broadened, and gradients calmed. |

## Trajectory landmarks

| Step | Rank | Centered slot rank | Cross-video cosine | Std | Present recon | Shuffled recon | Gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31.0* | 31.0* | 1.000 | ~0 | 1.004 | 1.004 | 0.000 |
| 2,000 | 146.3 | 29.9 | 0.019 | 1.043 | 0.827 | 0.929 | 0.102 |
| 7,500 | 212.3 | 30.5 | 0.017 | 1.146 | 0.729 | 0.939 | 0.210 |
| 14,500 | 208.2 | 30.7 | 0.059 | 1.117 | 0.713 | 0.940 | 0.227 |

\* Step-zero rank is mechanical slot identity before the code carries input information.

## Comparison with run 056

| Metric | Run 056: cov+var+SIGReg | Run 057: cov+var | Read |
|---|---:|---:|---|
| Effective rank | 201.58 | 208.22 | SIGReg not needed for rank |
| Centered slot rank | 29.87 | 30.71 | no slot-diversity cost |
| Present reconstruction | 0.73076 | 0.71285 | improved without SIGReg |
| Video gap | 0.20362 | 0.22681 | more correct-code dependence |
| Conditioned share | 74.6% | 78.0% | modest honesty recovery |
| Attention mean / minimum | 0.469 / 0.006 | 0.721 / 0.278 | no forced near-delta head |

The run is the strongest SSv2 present-only point in the sequence, but `F_c` was inactive. It
cannot be counted against the Phase 1 copy or batch-mean acceptance gates. Full interpretation:
[`ANALYSIS.md`](ANALYSIS.md).
