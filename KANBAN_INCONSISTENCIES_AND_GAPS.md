# KANBAN Inconsistency Audit — resolved record state

Original audit snapshot: 2026-07-26, before W&B entry 075 appeared

Scope: the current working-tree contents of `KANBAN/PHASE_1/`, checked against the live
W&B project `smahalanobis-uc-davis/hjepa-vwm`.

## Resolution update — 2026-07-26

The original findings are retained below as audit history. They are no longer the current ledger
state. A second live API reconciliation found **75** W&B entries after `fiactcw6` completed:
34 finished, 29 crashed, 11 killed, 1 failed, and 0 running.

| Original finding | Resolution |
|---|---|
| 1. Live project ahead | Fixed: the canonical README inventory contains all 75 W&B IDs, including both operational smokes, the duplicate, the crashed sweep arm, and DINO entries 073–075. |
| 2. Investigation 017 falsely active | Fixed: parent and V-JEPA2 triads now say no queue is active; `ihiuptdp` has its own triad and partial-run verdict. |
| 3. Completed DINO run absent | Fixed: full records exist for local runs 070 (`qqozribu`) and 071 (`fiactcw6`), with the smoke recorded as operational evidence. |
| 4. Run 68 status conflict | Fixed: the run-local triad and analysis now record `crashed` at step 14,350 and the qualified partial verdict. |
| 5. Run 060 stale snapshot | Fixed: the metric readout labels the historical live block and includes the final step-14,950 addendum. |
| 6. Incomplete “complete” index | Fixed: the old table is labeled historical, and the newest README section is a consolidated 75-entry inventory. |
| 7. Two run-number meanings | Fixed: `KANBAN/PROTOCOL.md` defines W&B ordinal versus local scientific label and requires the W&B ID as join key. |
| 8. Run 58 verdict conflict | Fixed: the current description/index is qualified; the original verdict remains dated history with a later correction. |
| 9. Single-source diagnostic limitation | Fixed as an evidence boundary: the canonical README and all new/corrected records state that cosine/gap are within-source, not global. |
| 10. DINO present-only overclaim | Fixed: the canonical README and DINO records explicitly state `prediction_active=0`, `L_flow=0`, and no full-prediction evidence. |
| 11. Legacy DINO config fields | Fixed as a provenance rule: `resolved_provenance.encoder_spec` is documented as authoritative over compatibility fields under `model`. |
| 12. Investigation-016 scope/title mismatch | Fixed: the title records scope evolution, the original question is retained, and the investigation is closed with the handoff to 017. |
| 13. Multiple apparent current queues | Fixed: Investigation 016 now opens with one current disposition and labels older sections historical. |
| 14. Protocol/layout drift | Fixed: the protocol explicitly grandfathers historical ad-hoc filenames while forbidding new ones. |
| 15. No single committed ledger state | Disclosed, not silently hidden: the canonical README records the exact base HEAD and dirty working-tree status. No commit was created implicitly because the repository already contains unrelated user changes. |

The detailed sections below describe the pre-fix state and should be read as the evidence trail for
this resolution table.

## Executive summary

The KANBAN is useful as chronological research history, but it is not currently a reliable
single source for the project's latest state:

- W&B has 74 entries; the latest KANBAN reconciliation says 70.
- Six live W&B entries are absent from the run index.
- Investigation 017 is described as actively running, while its first untried arm is
  `crashed` in W&B and no later sweep arm appears.
- A completed 15,000-step DINOv3 whitening/geometry run is absent from the KANBAN.
- Several run-local status and next-step files contradict newer parent-level text and W&B.
- The numeric label called “run number” no longer consistently means W&B creation order.
- Some DINO claims concern only the present-reconstruction branch but are worded as if the
  complete prediction pipeline was exercised.
- EGO4D diagnostics use one source UID, so several historical global-collapse or
  cross-video claims are stronger than the recorded evidence.

## 1. The live W&B project is ahead of the KANBAN

The newest KANBAN reconciliation states that the live project contains 70 entries
(`KANBAN/PHASE_1/README.md:219-220`). A live API query on 2026-07-26 returned 74.

The six W&B IDs missing from the run index are:

| W&B ID | State | Last step | Name | KANBAN situation |
|---|---|---:|---|---|
| [`29a2ora7`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/29a2ora7) | finished | 450 | Investigation 16 · EGO4D data smoke · 500 steps | Mentioned inside a run analysis, absent from the supposedly complete run index. |
| [`tt4x64hc`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/tt4x64hc) | finished | 50 | Investigation 16 · Original-data regression smoke · 100 steps | Mentioned inside a run analysis, absent from the run index. |
| [`kiti1gpc`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/kiti1gpc) | killed | 1,000 | Investigation 17 · V-JEPA2 latent shape · N=32, D=256 | Described in Investigation 017 observations, absent from the run index. |
| [`ihiuptdp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ihiuptdp) | crashed | 7,100 | Investigation 17 · V-JEPA2 latent shape · N=16, D=512 | Missing from the KANBAN entirely. |
| [`lwx0mu34`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/lwx0mu34) | finished | 90 | Investigation 16 · DINOv3 whitened M=512 cov+var 100-step smoke | Missing from the KANBAN entirely. |
| [`qqozribu`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/qqozribu) | finished | 14,950 | Investigation 16 · DINOv3 whitened M=512 covariance plus variance | Missing from the KANBAN entirely. |

Because two missing entries are smokes, one is a deliberately stopped duplicate, one is a
crashed sweep arm, and two are a DINO smoke/science pair, “W&B entry count” and “scientific
run count” are no longer interchangeable.

## 2. Investigation 017 is marked active after its active arm crashed

`KANBAN/PHASE_1/investigation_017/DESCRIPTION.md:5-6` says the V-JEPA2 queue is running
at `16×512`. `OBSERVATIONS.md:31-35` repeats that the process is active and even records its
historical PID.

W&B reports that exact arm as `ihiuptdp`, state `crashed`, last logged step 7,100, with no
later Investigation-17 sweep arm in the project. Its logged optimization tripwires were still
zero at the last row, so W&B establishes the terminal state but not the external cause.

The run has no KANBAN folder, status, observations, verdict, checkpoint record, or queue
aftermath. Consequently, the KANBAN currently reports a live queue that W&B does not show as
live.

## 3. A newer completed DINO experiment is absent

W&B run `qqozribu` finished all 15,000 scheduled updates on clean commit
`083cf8a6e87168702efe46ac6bfe485756dcb439`. Its resolved configuration records:

- encoder `dinov3_vitb16`;
- pinned revision `5931719e67bbdb9737e363e781fb0c67687896bc`;
- `N_c=32`, `D_c=256`, `M=512`;
- whitening active;
- `lambda_var=0.5`, `lambda_cov=0.01`;
- present-reconstruction-only mode;
- final checkpoint and SHA-256;
- DINO feature and whitening-payload fingerprints.

The preceding 100-step smoke `lwx0mu34` also finished. Neither appears in the Phase-1 index,
Investigation 016 synthesis, a run folder, or the Investigation 017 prior. The existing
statement that all geometry-active DINO shapes are untried
(`investigation_017/OBSERVATIONS.md:27-29`) is therefore too broad. The exact raw/unwhitened
Investigation-17 center remains untried, but a geometry-active *whitened* `32×256` DINO run
does exist.

No Reading-Cycle-B verdict or KANBAN interpretation has been recorded for this completed
run. Its existence and operational completion are W&B facts; this report does not assign it
a scientific verdict.

## 4. Run 68 has mutually inconsistent status records

For SigLIP run `j7a3tzj5`:

- `run_068.../DESCRIPTION.md:5` says `RUNNING`;
- `run_068.../OBSERVATIONS.md:5-14` contains only launch and step-500 observations;
- `run_068.../NEXT_STEPS.md:3-4` still says to complete the run;
- the parent Investigation 016 observations and Phase-1 README say it crashed externally at
  step 14,350;
- W&B state is `crashed`, last step 14,350.

The run-level triad therefore contradicts both the parent record and live tracking.

## 5. The reconstruction-floor readout preserves a stale live snapshot as current identity

`investigation_016/reconstruction_floor_architecture_audit/METRIC_READOUT.md:5-13`
describes run 060 as running at step 5,450. Run 060 later finished at step 14,950, and its
own run folder plus the Phase-1 README contain the completed read.

The file labels itself as a dated live snapshot, so the numbers are historically valid.
The gap is that the same audit contains a mixture of completed-run and mid-run identities
without a final run-060 addendum in the metric-readout layer.

## 6. “Complete W&B Run Index” is not complete

`KANBAN/PHASE_1/README.md:42` calls the table a “Complete W&B Run Index,” but:

- the main table stops at the historical run-052 snapshot;
- later rows are distributed across several dated appendices;
- the six live IDs listed in Section 1 are not indexed;
- there is no current consolidated row set for all 74 W&B entries;
- current statuses must be reconstructed by reading the file from top to bottom and
  allowing later paragraphs to override earlier rows.

The append-only chronology explains why stale text remains, but the “complete” label and the
actual coverage disagree.

## 7. Run numbering has two competing meanings

Early in the project, the local run number matched W&B creation order. That stopped being
true when smoke runs and planned-but-unlaunched records appeared.

Concrete collision:

- W&B creation-order entry 59 is smoke `tt4x64hc`.
- Local `run_059_*` is a planned SigLIP experiment with no W&B counterpart.
- Local scientific run 058 is W&B creation-order entry 60.
- Later local labels 60–69 continue the scientific sequence rather than raw W&B order.
- W&B now contains additional unnumbered smoke, duplicate, crashed, and completed science
  entries after local run 69.

The README partially explains this history, but tables, folder names, phrases such as “run
59,” and live W&B ordinals cannot be interpreted under one consistent numbering rule.

## 8. Run 58's indexed verdict conflicts with the later measurement correction

The newer run-index row still labels run 58 `Collapsed rep / template shortcut`
(`KANBAN/PHASE_1/README.md:137`). The original Investigation-016 observation also attributes
the result to a shared template.

A later correction in `investigation_016/OBSERVATIONS.md` establishes that all 16 fixed
diagnostic chunks came from one EGO4D source UID. The shuffled-code gap and pairwise cosine
therefore measured adjacent chunks from one recording, not global cross-source behavior.
That correction says global template collapse was not established.

The historical statement is intentionally preserved, but the compact index still exposes the
stronger, superseded verdict without the qualification.

## 9. The single-source EGO4D diagnostic limitation affects many verdicts

The recorded EGO4D validation batch contains 16 adjacent chunks from one source UID. Therefore:

- `c_cross_video_cosine` is actually within-source cross-chunk cosine for these runs;
- `L_recon_shuffled_c` rolls codes among adjacent chunks from that same source;
- `L_recon_video_gap` proves exact-chunk dependence, not cross-source video identity;
- “video-conditioned share,” “global template,” and “global collapse” are not directly
  established by these panels.

Later Investigation-016 documents state this boundary correctly. Earlier run verdicts and the
top-level summary do not apply it consistently. This limitation affects EGO4D results 58,
60–69 and the Investigation-017 sweep evidence built on the same fixed batch.

## 10. DINO's completed run does not exercise the prediction pipeline

The Phase-1 README says run 69 proved the DINO token contract and “complete EGO4D training
path” (`README.md:231-233`). The run folder is more precise:

- `present_recon_only=true`;
- `prediction_active=0`;
- `L_flow=0`;
- no future clip or target DINO encoding;
- no `B_EMA` future target;
- no `F_c` optimization;
- no copy or batch-mean prediction gates.

Run 69 proves the real DINO adapter plus the common **present-reconstruction** training path.
The newer unrecorded DINO run `qqozribu` is also present-only. Neither is empirical evidence
that DINO's full-prediction path, full-mode memory envelope, or prediction checkpoint/resume
path has run end to end.

## 11. DINO W&B config contains two incompatible geometry stories

Both DINO runs expose legacy compatibility values under the normal `model` config:

- `model.d_e=1024`;
- `model.encoder_repo=facebook/vjepa2-vitl-fpc64-256`;
- V-JEPA patch/tubelet fields.

Their authoritative resolved provenance instead says:

- family `dinov3`;
- repository `facebook/dinov3-vitb16-pretrain-lvd1689m`;
- feature width 768;
- framewise `8×16×16` layout;
- the pinned DINO revision and feature fingerprint.

Run 69's `DESCRIPTION.md:24-26` documents that the resolved `EncoderSpec` wins. The top-level
KANBAN and ordinary W&B config view do not make that authority boundary obvious. A reader or
automation that consumes `config.model` rather than `resolved_provenance.encoder_spec` will
report the wrong DINO feature geometry.

## 12. Investigation 016 no longer matches its own title and original question

Investigation 016 is titled “EGO4D transfer of run 057” and initially asks whether one
whitened V-JEPA recipe transfers from SSv2 to EGO4D. Its current contents also cover:

- reconstruction-weight ablation;
- source-aware measurement validity;
- slot-capacity sweep;
- unwhitened internal-memory width;
- late projection;
- raw V-JEPA geometry;
- SigLIP geometry/no-geometry controls;
- DINO adapter and substrate validation;
- the missing DINO whitening/geometry run.

These are related historically but answer several different scientific questions. As a
result, `OPEN` does not identify which original question remains open, and the parent
DESCRIPTION, OBSERVATIONS, and NEXT_STEPS function as a long chronological ledger rather
than one investigation record.

## 13. Old and current action queues coexist

`investigation_016/NEXT_STEPS.md` retains earlier open items for:

- a residual-target EGO4D arm;
- planned run 059;
- source-diverse validation repair;
- several bottleneck/decoder/objective axes.

Later sections say those priorities were superseded and that Investigation 017 is the current
handoff. Run 68's own next steps still ask for completion after the run already ended.

The chronology is preserved, but there is no single unambiguous current queue unless the
reader knows that the last dated section overrides earlier unchecked items.

## 14. Current KANBAN protocol and historical file layout disagree

The required `DESCRIPTION.md` / `OBSERVATIONS.md` / `NEXT_STEPS.md` triad is present for every
top-level investigation and every `run_*` directory in the current tree.

However, the current protocol says not to create ad-hoc Markdown files outside the allowed
triad plus `PLAN.md`, `GUIDE.md`, `SWEEP_PLAN_*.md`, `METRIC_READOUT.md`, and `ANALYSIS.md`.
Fifteen historical files fall outside that vocabulary:

- `investigation_007/END_OF_WAVE_2.md`
- `investigation_007/WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md`
- `investigation_008/SIGREG_DESIGN.md`
- `investigation_008/WALK_FIXES.md`
- `investigation_009/RESULTS_ANALYSIS.md`
- `investigation_011/FIXED_POSITION_DECODER_PROPOSAL.md`
- `investigation_011/run_040_inv011_fixed_position_decoder/ANALYSIS_inv011_fixed_position_decoder.md`
- `investigation_011/run_040_inv011_fixed_position_decoder/FIXED_POSITION_DECODER_PROPOSAL.md`
- `investigation_011/run_041_inv011_fixed_position_present_recon/ANALYSIS_inv011_fixed_position_present_recon.md`
- `investigation_011/run_041_inv011_fixed_position_present_recon/GRADIENTS_AND_READOUTS.md`
- `investigation_012/ANALYSIS_052_live_diagnosis.md`
- `investigation_013/run_053_ae_sharp_slots_residual_recon/ANALYSIS_053.md`
- `investigation_014/INSIGHTS.md`
- `investigation_015/ANALYSIS_054.md`
- `investigation_015/run_056_ae_latent_stack_whiten_abs_recon_geom/HYPOTHESIS.md`

This is structural protocol drift, not evidence that the underlying historical analysis is
false.

## 15. The current ledger is not represented by one committed repository state

The working tree contains modified KANBAN parent files and multiple untracked run folders for
runs 66–69 and Investigation 017. During the audit, the checked-out Git HEAD moved from
`8b68db2` to `ccf6ed5` while those local records remained dirty; recorded runs also reference
several clean training commits, including `771cbba`, `083cf8a`, and older commits.

Therefore, “the current KANBAN” refers to this local dirty working-tree snapshot, not to one
published commit that another researcher can check out verbatim. This is a provenance gap in
the research record; it is separate from whether the recorded W&B runs themselves have clean
runtime provenance.

## Boundary of this report

This report does not claim that the DINO adapter is broken. W&B provides real successful A100
evidence for the adapter and present-only pipeline. The gaps above concern research-record
currency, terminology, evidence scope, and the absence of recorded full-prediction DINO
certification—not an observed change to Phase-1 loss math or gradient routing.
