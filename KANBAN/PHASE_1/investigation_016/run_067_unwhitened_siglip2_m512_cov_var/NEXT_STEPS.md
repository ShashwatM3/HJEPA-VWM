# Next steps — Run 67

1. Preserve the stopped W&B history, step-10,000 checkpoint, provenance, and logs; do not resume or
   relabel this run.
2. Use Run 68 as the exact SigLIP2 M=512 control with `lambda_var=lambda_cov=0`, holding every other
   scientific and runtime parameter fixed.
3. Compare Run 67 and Run 68 only on overlapping steps when attributing differences to the
   regularizers; retain the within-source diagnostic-batch qualification.
