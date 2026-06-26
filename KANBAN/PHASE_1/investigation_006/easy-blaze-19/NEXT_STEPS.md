# Next steps — easy-blaze-19

## Status

Run **finished the full 15000 steps**. Analysis complete.

Option 3, as tested here (`lambda_recon=0.05` + `lambda_recon_pred=0.05`), is a
negative result: it ran stably but did not improve prediction, did not lower
`L_recon_pred`, and mildly worsened representation metrics.

## What this run settled

- **Do not rerun the same option-3 configuration.** The copy-gate curve sat on top of
  `fanciful-lake-18` and ended slightly worse.
- **The limiting issue is not cheap-vs-clean predicted recon.** The reconstruction
  readouts are saturated near ~0.60, so the channel is blind to prediction quality.
- **The feared `F_c` stability problem did not occur.** Through-`F_c` recon is
  technically stable in this stabilized regime; it is just not informative enough.
- **The extra branch can hurt `c`.** Rank, cross-video cosine, `c_std_mean`, and
  `L_var` drifted in the wrong direction without buying prediction improvement.

## Forks implied by the result

1. **Break the reconstruction floor first.** If reconstruction is still the desired
   lever, it needs an unsaturated channel before it can train prediction:
   - larger `c` capacity,
   - larger / stronger decoder `D`,
   - higher reconstruction weight,
   - or an explicit capacity ablation to determine whether the ~0.60 floor belongs to
     `c`, `D`, or both.
2. **Question reconstruction as the copy-gate lever.** Over horizon 12, `c` changes
   little enough that copy-forward keeps getting stronger. If that is a task/horizon
   property, the lever may be the prediction target or horizon design rather than a
   decoder-side auxiliary loss.

## Do not do next

- Do not spend a run on only "clean" option 3 unless the reconstruction floor is
  addressed first. The evidence says even trained predicted recon cannot see
  prediction quality at this capacity.
- Do not treat the stable completion as success. Stability was a guardrail; the
  target metric was the copy gate, and it failed.

## Parent update

Update `investigation_006` to record option 3 as a completed negative result and to
close or redirect the investigation toward a new, separately scoped capacity /
target-horizon question.
