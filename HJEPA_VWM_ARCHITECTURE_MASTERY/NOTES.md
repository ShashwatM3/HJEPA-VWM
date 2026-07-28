# Teaching notes

## Learner preferences

- Exhaustive, end-to-end coverage is preferred over brevity.
- No discovery interview or clarification round was requested.
- The material must support oral quizzing on shapes, numbers, data flow, modules, training, and
  MLOps.
- Code-grounded distinctions matter more than a smooth but historically blended narrative.
- Every delivered course artifact must be a Markdown file; use Markdown tables, code blocks, and
  Mermaid diagrams rather than separate HTML/CSS/JavaScript assets.

## Instructional approach

- Start with one concrete sample and one concrete default shape.
- Introduce symbols before formulas.
- Repeat important invariants from different angles: tensor flow, gradient flow, parameter ownership,
  diagnostics, and failure mode.
- Use explicit “implemented/default/experiment/planned” labels.
- Put dense memorization material in reference sheets and keep lessons narrowly scoped.
- Keep answers separate from drills to preserve retrieval practice.

## Scope boundary

This corpus teaches the current Phase 1 repository and the documented future roadmap. It does not
invent implementations for `F_e`, inference ODE integration, or the frame generator. Where the
original brief conflicts with current code, the conflict is taught explicitly.

## Snapshot caveat

The working tree was already dirty. Existing user changes were treated as source material and were
not modified. The new corpus is additive under `HJEPA_VWM_ARCHITECTURE_MASTERY/`.
