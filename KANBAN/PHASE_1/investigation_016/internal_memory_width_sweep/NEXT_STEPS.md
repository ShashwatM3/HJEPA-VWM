# Next steps — unwhitened internal-memory width sweep

1. Pass local tests and publish the exact implementation commit.
2. Synchronize that commit to the RunPod without disturbing unrelated state.
3. Launch [`RUN_SWEEP.sh`](RUN_SWEEP.sh) through the single queue in [`GUIDE.md`](GUIDE.md).
4. Verify Stage 0 and the 1,024-wide batch-64 resource report before interpreting either arm.
5. Let 512 and 1,024 finish sequentially; record W&B IDs, checkpoint hashes, and provenance.
6. Run Reading Cycle B separately, then compare late loss together with shuffled-code separation.
7. Choose 1,024 only if its preservation gain clears [`PLAN.md`](PLAN.md)'s effect/cost guard;
   otherwise retain 512 as the practical width.
