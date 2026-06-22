# Observations — slot-loss-k12-low-run-4

## Finding

**`L_slot` glued near ~1.0** — gradient did not move the slot metric meaningfully.

## Root cause (code)

`slot_diversity_loss` used **raw** slot vectors while `slot_diversity_rank` diagnostic
used **centered** slots — loss/metric mismatch.

## Interpretation

Not a verdict on slot loss in general; exposed a **bug/ inconsistency** to fix before
any further slot experiments.
