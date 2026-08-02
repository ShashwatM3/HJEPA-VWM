# Analysis — fixed residual coordinates

## Conclusion

Freezing the representation helped, but it did not solve prediction. The evidence rejects the
claim that moving bottleneck coordinates were the dominant obstruction. It supports the narrower
claim that coordinate motion was one obstruction, because the fixed run was materially better than
the matched joint run. The remaining failure is now on the temporal predictor/objective side, or
in the amount of predictable information available in the fixed present code.

The registered run stopped early, so it is formally incomplete. The negative conclusion is still
strong. The run reached step 12,450, had already consumed 99.02% of the cumulative learning-rate
mass, and had shown no new downward trend in either baseline ratio for roughly the last 8,000
steps.

## 1. What this experiment isolated

Investigation 020 could not isolate predictor learning from representation learning. Its residual
arm updated online `B` and `F_c` together. Its target residual came from detached `B_EMA`. This
created two coupled processes:

1. `B` changed the coordinates used to condition `F_c`.
2. `B_EMA` changed the coordinates used to define the regression target.

The predictor therefore learned a vector field while both ends of the problem evolved.

Investigation 021 removed that nonstationarity. It loaded the same present-only checkpoint, made
`B_EMA` an exact copy of `B`, froze `B`, `B_EMA`, and `D`, and optimized only `F_c` with `L_flow`.
The encoder, data, order, seed, residual construction, fresh `F_c`, architecture, horizon, batch,
and schedule stayed fixed.

This is the correct causal test for the moving-coordinate hypothesis. Any improvement relative to
the matched joint run can be attributed to removing representation updates, subject to ordinary
finite-run noise. Any remaining failure cannot be blamed on coordinate drift.

## 2. The stationarity intervention worked

The diagnostic representation did not merely look stable. Every fixed-batch representation value
was exactly constant across all 25 diagnostic rows. The following quantities did not change by one
logged bit:

- online and target spread;
- online and target effective rank;
- encoder-side and latent-side cross-video cosine;
- slot-diversity ranks and attention entropy;
- copy and batch-mean residual losses;
- correct-code and shuffled-code reconstruction losses.

The most direct temporal check is `coarse_copy_loss`. It remained exactly `0.9625443220`. This loss
is `mean(Delta^2)`. If the bottleneck coordinates moved, the residual distribution could move and
this value could drift. It did not. The run therefore tested one fixed residual distribution.

The optimization contract also behaved correctly. `loss` equaled `L_flow` on every row. `B` and
`D` AGC metrics stayed zero. Exactly 70 `F_c` parameter tensors had gradients. No step was skipped,
no NaN appeared, and no instability warning fired.

The experiment is consequently valid as a fixed-coordinate trajectory through the human stop.
Its negative prediction result is not an execution or gradient-routing artifact.

## 3. The fixed representation was healthy

The fixed substrate was not a weak or collapsed code.

`c_std_mean=1.103687` and `c_dead_dim_frac=0` show that feature dimensions had healthy numerical
spread. `c_effective_rank=364.281/512` shows that the distribution occupied about 71% of the
available feature directions. `c_plus_effective_rank=353.867/512` shows that the target branch was
similarly rich. The target rank was only 2.86% below online rank.

The cross-video result is stronger than rank alone. Raw DINOv3 features had
`e_cross_video_cosine=0.402768`. The bottleneck reduced this to `0.109722`, a 72.76% reduction on
the same source-diverse batch. Different videos therefore mapped to substantially different
abstract directions.

The slot axis was also healthy. The raw and centered slot-diversity ranks were 54.45 and 58.79 out
of 64. Mean attention entropy was 0.418, with a minimum-head value of 0.116. The bottleneck was not
producing 64 copies of one uniform pooled vector.

The decoder supplied an independent content check. `L_recon_present=0.124881`, while decoding
another video's code gave `0.538133`. The gap of `0.413252` rules out the historical
video-independent decoder template. The fixed code carries video-specific decodable information.

The predictor failure therefore cannot be reduced to low variance, rank collapse, cross-video
collapse, slot collapse, or reconstruction blindness.

## 4. What `F_c` did learn

It would be wrong to say that `F_c` learned nothing.

On the fixed diagnostic problem, model loss fell from `3.021331` to a best value of `1.401737`.
This is a 53.61% reduction. The last-six available median was `1.432718`, which is 52.58% below
initialization. The stochastic training loss also fell from `2.202297` to `0.682081` at the cutoff.

The fixed decoder saw corresponding movement. `L_recon_chat` fell from `0.231562` to a last-six
median of `0.174081`. This is a 24.82% improvement. The gap between the predicted-code decode and
the true-future-code decode closed by 50.99%.

These two signals agree. The trained output is substantially better than the random initialization,
and it moves toward a region that the honest fixed decoder recognizes as more future-like.

The correct statement is therefore not “optimization failed.” Optimization succeeded at reducing
its own error. The correct statement is “the learned predictor remained worse than trivial target
guesses.”

## 5. Why the predictor still fails the project claim

The copy and batch-mean losses were fixed at `0.962544` and `0.909427`. The batch mean is already
5.52% better than zero residual. This means the residual distribution has an unconditional mean
component. A video-conditioned predictor must beat that mean before it can claim to use the
current video's state.

`F_c` never crossed either baseline.

Its best copy ratio was `1.456283`. Its best batch-mean ratio was `1.541340`. In the last six
available diagnostic rows, the medians were `1.488469` and `1.575406`. Thus the late model was
48.85% worse than zero residual and 57.54% worse than the batch mean.

This failure was not a brief warmup effect. By step 4,500, the copy ratio had reached 1.496. It then
fluctuated between roughly 1.46 and 1.65 through step 12,000. The best point appeared at step
10,500. The ratio worsened to 1.512 by step 12,000 while the learning rate continued to decay.

The model had reached a stable suboptimal basin under this schedule. The missing tail was very
unlikely to move the ratio from about 1.5 to below 1.0, and still less likely to reach the formal
0.70/0.50 gates, because only 0.98% of the schedule's cumulative learning-rate mass remained.
This is a probabilistic research judgment, not a mathematical impossibility claim.

## 6. What freezing changed relative to joint training

The matched comparison is Investigation-020 residual run `3y2hxj5t`. Both runs began with the same
fixed diagnostic values and the same fresh coarse-flow component. At step 0, both had model loss
`3.021331`, copy loss `0.962544`, copy ratio `3.138901`, and batch ratio `3.322235`.

The fair late comparison uses the same diagnostic steps, 9,500 through 12,000:

| Metric, matched six-step median | Joint `B/F_c/D` | Fixed `F_c` only | Relative change |
|---|---:|---:|---:|
| `coarse_model_loss` | 2.164067 | 1.432718 | 33.80% lower |
| `coarse_vs_copy_ratio` | 1.701472 | 1.488469 | 12.52% lower |
| `coarse_vs_batch_mean_ratio` | 1.799931 | 1.575406 | 12.47% lower |
| `L_recon_chat` | 0.184183 | 0.174081 | 5.49% lower |

The comparison supports one causal statement: moving representation coordinates made the
predictor's job harder. Removing that motion improved absolute model error, both baseline ratios,
and the decoder readout.

The comparison rejects a stronger statement: moving coordinates were not the dominant obstruction.
After removing them completely, the predictor still lost badly to both baselines. Coordinate
stationarity was helpful but insufficient.

The joint run also changed the task itself. Its copy loss rose from 0.962544 to a late median near
1.300293 because joint `B` training made residuals larger. The fixed run kept the original smaller
residual scale. A smaller residual makes the absolute regression target smaller, but it also makes
zero residual a stronger baseline. The ratio comparison controls for that scale change, and the
fixed run still remained above 1.0.

## 7. What the experiment proves

The evidence supports the following statements.

1. The Investigation-019 DINOv3 bottleneck supplies a stable, high-rank, video-specific, decodable
   present representation.
2. The same fixed bottleneck supplies nonzero temporal residuals at horizon 12.
3. Freezing `B` and `B_EMA` improves `F_c` training relative to the matched joint recipe.
4. The current six-block `F_c`, rectified-flow objective, condition-dropout recipe, optimizer, and
   15,000-step schedule do not learn the fixed residual task well enough to beat zero residual or
   the batch mean.
5. More work on bottleneck variance, covariance, SIGReg, slot count, decoder size, or encoder
   geometry is not the first response to this result. Those axes do not address the isolated
   failure.

## 8. What the experiment does not prove

The run does not prove that the residual dynamics are fundamentally unpredictable. It tests one
present code, one horizon, one `F_c` architecture, and one training objective.

The run also does not cleanly separate predictor capacity from objective geometry. `F_c` receives a
noised residual `z`, a random flow time, and `c_t`. It learns the velocity field
`Delta - eps`. A direct deterministic residual regressor would solve a simpler problem:

```text
Delta_hat = G(c_t)
minimize mean((Delta_hat - Delta)^2).
```

Failure of the present flow model can therefore come from at least three remaining sources:

1. `c_t` does not contain enough information to predict the particular future residual.
2. The six-block predictor does not have the right capacity or inductive bias.
3. Rectified-flow denoising and time conditioning make optimization harder than direct residual
   regression under the current budget.

There is also a diagnostic limitation. `coarse_model_loss` is velocity-field MSE at one sampled
`tau` and one sampled noise tensor. The copy and batch baselines are direct target-space guesses
written as velocities using that known noise tensor. They use the same scalar ruler, so the ratios
are reproducible operational gates. They are not an inference-time comparison between a numerically
integrated flow rollout and two deterministic transition models.

This limitation does not rescue the current run under the project's declared gates. It does limit
the broader interpretation. The 24.82% improvement in `L_recon_chat` shows that the learned vector
field contains some useful signal even though its velocity MSE loses to the fixed guesses.

## 9. Recommended next investigation

Do not run another bottleneck ablation. Open a full-scale temporal-objective investigation on the
same frozen checkpoint.

The decisive paid arm should train a direct residual predictor on the fixed coordinates for the
same full schedule:

```text
c_t -> G -> Delta_hat
target = B_EMA(e_{t+12}) - B_EMA(e_t)
loss = mean((Delta_hat - target)^2)
```

Its evaluation should use direct residual MSE against zero and batch mean on the same fixed batch.
This is a larger and cleaner next test than merely widening the existing flow. It asks whether the
present code contains deterministic predictive signal without mixing that question with denoising,
flow time, or one-step endpoint approximations.

Before paying for that run, add an offline evaluation of the current checkpoint that numerically
integrates the learned flow from noise to residual and compares the resulting endpoint against the
same direct baselines. This is not a small training ablation. It is a measurement needed to tell
whether the current velocity-loss gate understates usable rollout quality.

The decision after the direct-regression run is simple:

- If direct regression beats both baselines, the representation contains predictive information
  and the current flow objective/training formulation is the main problem.
- If direct regression still loses, the next large move should jointly address predictor capacity
  and temporal information, such as a stronger transition architecture or a representation trained
  explicitly for predictable state. More static geometry regularization would not be justified.

## Final verdict

The experiment answered its causal question despite the early stop. Fixed coordinates remove a
real source of difficulty, but they do not produce a passing predictor. The project should now stop
treating representation motion as the main explanation and move to an objective-level test of
whether these fixed residuals are directly predictable from `c_t`.
