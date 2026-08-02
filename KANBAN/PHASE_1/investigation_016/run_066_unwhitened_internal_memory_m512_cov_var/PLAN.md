# Plan — Run 66, unwhitened M=512 with covariance plus variance

## Hypothesis

Run 64 proved that the unwhitened, late-projection M=512 compressor can reach low raw cosine loss
and a nonzero correct-versus-shuffled gap, but every geometry weight was zero and the code
contracted to late rank `10.75`, std `0.205`, and cross-example cosine `0.954`. Historical controlled
SSv2 evidence identifies covariance as the operative rank lever and the variance floor as the
spread guard, without requiring SIGReg.

Activating `lambda_var=0.5` and `lambda_cov=0.01` should directly oppose that contraction. The
central tradeoff is whether the broader code remains decodable under the same raw reconstruction
objective, not whether geometry metrics can be improved in isolation.

## Locked comparison

The baseline is Run 64 (`4biwq87o`). No architecture, data, target, decoder, optimizer, or schedule
field changes. The two geometry weights are treated as one settled bundle; this run does not
separately attribute variance versus covariance.

## Readout

Apply present-only Reading Cycle B over the final six diagnostic points (steps 12,000–14,500).
Primary reads are fixed and active reconstruction, correct-versus-shuffled gap and conditioned
share, std, cross-example cosine, effective rank, centered slot rank, stability, and the realized
`L_var`/`L_cov` terms.

Healthy-geometry reference gates remain std `0.8–1.2`, cosine below `0.5`, and effective rank above
`60`. A scientific success requires reconstruction to remain clearly code-dependent while those
geometry gates improve; pretty geometry with weak content is not a success. The single-source
fixed batch prohibits a global cross-source verdict.
