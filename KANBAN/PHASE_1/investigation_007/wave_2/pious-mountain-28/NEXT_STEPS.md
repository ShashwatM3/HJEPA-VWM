# Next steps — pious-mountain-28

**TIER 0 — re-run this run.** It is one of the two bookends (`n_c=64` + `n_c=256`) that decide the
latent-capacity question; the reduced re-run wave should include it. Use the exact command in
[DESCRIPTION.md](DESCRIPTION.md); harden the launch (tmux detach + step-600 tripwire) per
[Wave 2 NEXT_STEPS](../NEXT_STEPS.md) so the synchronized death can't recur.

Read alongside [classic-yogurt-29](../classic-yogurt-29/) (n_c=128) and
[earnest-dragon-25](../earnest-dragon-25/) (n_c=256) — the latent ladder it begins. Outcome metric:
`coarse_vs_copy_ratio` → <1 (with `c_effective_rank` rising) = latent capacity is the lever;
anything else = pivot to a temporal objective ([`../END_OF_WAVE_2.md`](../END_OF_WAVE_2.md) §2.6).
