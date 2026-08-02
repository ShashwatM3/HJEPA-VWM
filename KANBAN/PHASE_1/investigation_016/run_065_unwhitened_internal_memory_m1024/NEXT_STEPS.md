# Next steps — Run 65, EGO4D internal memory M=1024

1. Keep this completed arm immutable; its W&B run, provenance, and final checkpoint are valid.
2. Do not promote M=1024 as the default: it missed both material-effect thresholds despite its
   higher recorded-batch rank. Keep M=512 for the next architecture arm.
3. Repair the fixed EGO4D diagnostic to use source-unique clips and cross-source derangement before
   turning the within-source conditioning result into a global preservation claim.
4. Test the remaining final-rate/objective bottlenecks separately: external `D_c` or final code
   rate, decoder form, and residual/cosine-plus-magnitude targets must not be bundled together.
5. If healthy representation geometry is required, restore one geometry force in a dedicated arm;
   zero regularizer weights intentionally allowed both width arms to become low-rank decodable.
