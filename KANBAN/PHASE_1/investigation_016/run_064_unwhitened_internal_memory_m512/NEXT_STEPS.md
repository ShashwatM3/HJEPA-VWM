# Next steps — Run 64, EGO4D internal memory M=512

1. Launch through the fail-fast queue in [`../GUIDE.md`](../GUIDE.md).
2. Require Stage 0 and the 1,024-wide resource gate to pass before this arm starts.
3. Verify this arm finishes 15,000 steps with valid W&B/provenance/checkpoint evidence.
4. Let the queued 1,024 sibling start only after this arm exits successfully.
5. Compare both arms under the parent sweep's loss-plus-gap rule.
