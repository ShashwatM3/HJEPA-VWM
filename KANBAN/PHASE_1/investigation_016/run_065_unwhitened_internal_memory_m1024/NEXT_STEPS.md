# Next steps — Run 65, EGO4D internal memory M=1024

1. Pass the largest-shape Stage 0 and resource gate in the parent queue.
2. Wait for the paired 512 arm to finish successfully; never run both on the one GPU.
3. Complete 15,000 steps with valid W&B/provenance/checkpoint evidence.
4. Compare late loss and improved correct-versus-shuffled gap against 512.
5. Prefer 1,024 only if it clears the parent effect/cost rule without instability or shortcutting.
