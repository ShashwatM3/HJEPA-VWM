# HJEPA-VWM

Hierarchical JEPA-Flow video world model — planning docs and agent implementation specs (v0).

## For coding agents

**Start here:** [`AGENT_FILES/AGENTS.md`](AGENT_FILES/AGENTS.md)

That file is the focused entry point: what the project is, the full filesystem map, mandatory read order, and where behavior rules live. Do not infer the architecture from generic patterns.

**Priority reads (in order):**

1. [`AGENT_FILES/AGENTS.md`](AGENT_FILES/AGENTS.md) — compass
2. [`AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`](AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md) — how to operate
3. [`AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md`](AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md) — how to write code

Then execute the phase the human assigns under [`AGENT_FILES/PHASES/`](AGENT_FILES/PHASES/).

Many tools also auto-discover the root [`AGENTS.md`](AGENTS.md), which redirects to the full doc above.

## For human operators

Network volume layout (current vs target) → [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](AGENT_FILES/SETUPS/VOLUME_LAYOUT.md).

First RunPod setup → [`AGENT_FILES/SETUPS/SETUP.md`](AGENT_FILES/SETUPS/SETUP.md) Path A.

Push/pull training cycle → Path B in the same file.

## Documentation layout

| Folder | Contents |
|---|---|
| [`AGENT_FILES/AGENT-BEHAVIOUR/`](AGENT_FILES/AGENT-BEHAVIOUR/) | `PROTOCOL.md`, `CODE_DESIGN.md` |
| [`AGENT_FILES/KNOWLEDGE/`](AGENT_FILES/KNOWLEDGE/) | Architecture brief, `UNDERSTANDING.md` |
| [`AGENT_FILES/PHASES/`](AGENT_FILES/PHASES/) | Phase 1–3 implementation specs |
| [`AGENT_FILES/SETUPS/`](AGENT_FILES/SETUPS/) | `VOLUME_LAYOUT.md`, `SETUP.md`, `SETUP_POD.md` |
