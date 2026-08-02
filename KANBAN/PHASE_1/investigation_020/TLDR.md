# TL;DR

We gave two models the same strong video memory and asked them to guess what comes next.
One guessed the whole future memory; the other guessed only what should change.
Both trained cleanly for all 15,000 steps, so this is a learning result, not a crashed-run result.
The change-only model kept a rich, video-specific code, but its guess was still worse than guessing “nothing changes.”
The whole-future model made present and future codes look more alike, so copying got easier while its predictor got relatively worse.
Technically, the late model/copy ratios were `1.726` residual and `3.133` full-latent, versus the required `<=0.70`; both also failed the `<=0.50` batch-mean gate.
So there is no winning predictor: residual avoids the static-`c` trap, and the clean next test is residual prediction with the pretrained bottleneck frozen so Fc learns in fixed coordinates.
