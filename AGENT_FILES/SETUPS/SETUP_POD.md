# SETUP_POD.md — RunPod Infrastructure Reference

> **Start here instead:** [`SETUP.md`](SETUP.md) — full end-to-end operator guides (Path A / Path B).
>
> This file is **reference only** — volume layout, path defaults, troubleshooting.
>
> **Provenance:** RunPod-specific claims cite official docs (linked below). Folder layout (`/workspace/data/ssv2`) is **this project's convention**, not RunPod's. Symlink counts are from **your** volume inspection in [`CHAT.md`](../../CHAT.md) — verify on the pod.

Official RunPod docs (verified):
- [Connect to a Pod with SSH](https://docs.runpod.io/pods/configuration/use-ssh)
- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- [Transfer files (SCP/rsync)](https://docs.runpod.io/pods/storage/transfer-files)

**Agent docs layout:** see [`../AGENT-BEHAVIOUR/PROTOCOL.md`](../AGENT-BEHAVIOUR/PROTOCOL.md) §0.

---

## Volume layout (target state)

After first-time setup ([`SETUP.md`](SETUP.md) Path A, steps A7–A8):

```
/workspace/                                          ← network volume (persists across pod stops)
├── hierarchal-jepa-flow-world-model/                ← git repo — CODE ONLY
├── data/
│   ├── ssv2/                                        ← full SSv2 (~169k train symlinks)
│   │   ├── train/
│   │   ├── validation/
│   │   └── labels.json
│   └── ssv2_tiny/                                   ← smoke subset (~4k train); created by make_subset.py
│       ├── train/
│       ├── validation/
│       └── manifest.json
├── ssv2_raw/                                        ← raw .webm files (read-only)
│   └── 20bn-something-something-v2/
├── checkpoints/                                     ← all training checkpoints
└── hf_cache/                                        ← Hugging Face cache (Phase 3 VAE)
```

### Path defaults (hardcoded in `config.py`)

| Path | Purpose |
|---|---|
| `/workspace/hierarchal-jepa-flow-world-model` | Repo root |
| `/workspace/data` | Datasets |
| `/workspace/checkpoints` | Checkpoints |
| `/workspace/hf_cache` | HF model cache |
| `JEPA_DATA_ROOT` env var | Replaces `/workspace/data` for local dev only |

---

## What “data migration” means

**Not** RunPod documentation — **project convention** to match `config.py` paths.

**Not** re-downloading or re-processing video.

**Is** moving prepared SSv2 on your volume:

- **From:** `/workspace/hierarchal-jepa-flow-world-model/data/something-something-v2/`
- **To:** `/workspace/data/ssv2/`

Instant `mv` — symlinks still point to `/workspace/ssv2_raw/`. Run once per volume. Full commands in [`SETUP.md`](SETUP.md) step **A8**.

---

## SSH methods (RunPod Connect tab)

| Method | Example | SCP/SFTP |
|---|---|---|
| Basic (proxied) | `ssh xxxxx@ssh.runpod.io -i ~/.ssh/id_ed25519` | No |
| Full (public IP) | `ssh root@IP -p PORT -i ~/.ssh/id_ed25519` | Yes |

First-time SSH key setup: [`SETUP.md`](SETUP.md) step **A5**.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| SSH asks for password | Wrong key, bad permissions (`chmod 600 ~/.ssh/id_ed25519`), or key not in RunPod Settings → SSH Public Keys. [Docs](https://docs.runpod.io/pods/configuration/use-ssh#troubleshooting-ssh-key-authentication) |
| `/workspace` empty | Wrong volume attached at pod deploy. Use the volume that contains `ssv2_raw/`. |
| SSv2 symlink count 0 | Migration not run or broken symlinks. Check `/workspace/ssv2_raw/20bn-something-something-v2/*.webm`. |
| OOM on 40GB A100 | Use `--data ssv2_tiny`; see README for grad accumulation when implemented. |
| Training killed when laptop sleeps | Use tmux ([`SETUP.md`](SETUP.md) A15). |
| Pod stopped mid-run | Checkpoints on volume persist — resume with `--resume`. |

---

## Agent vs human responsibilities

| Task | Who |
|---|---|
| Path A / Path B operator steps | **You** ([`SETUP.md`](SETUP.md)) |
| Write Python codebase | **Agent** ([`../PHASES/PHASE_1.md`](../PHASES/PHASE_1.md) …) |
| Pass phase acceptance gates | **Agent** reports; **you** approve |

---

## Related docs

| Doc | Role |
|---|---|
| [`SETUP.md`](SETUP.md) | **Main operator guide** (first time + iterative) |
| [`../PHASES/PHASE_1.md`](../PHASES/PHASE_1.md) | Agent implementation spec |
| [`../AGENT-BEHAVIOUR/PROTOCOL.md`](../AGENT-BEHAVIOUR/PROTOCOL.md) | Agent behavior |
| [`../KNOWLEDGE/UNDERSTANDING.md`](../KNOWLEDGE/UNDERSTANDING.md) §13 | SSv2 dataset constants |
