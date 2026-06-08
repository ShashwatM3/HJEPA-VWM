# SETUP_POD.md — RunPod Infrastructure Reference

> **Start here instead:** [`SETUP.md`](SETUP.md) — full end-to-end operator guides (Path A / Path B).
>
> **Volume structure (current vs target):** [`VOLUME_LAYOUT.md`](VOLUME_LAYOUT.md) — canonical tree, path contract, migration map.
>
> This file is **reference only** — SSH, troubleshooting. RunPod-specific claims cite official docs (linked below).

Official RunPod docs (verified):
- [Connect to a Pod with SSH](https://docs.runpod.io/pods/configuration/use-ssh)
- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- [Transfer files (SCP/rsync)](https://docs.runpod.io/pods/storage/transfer-files)

**Agent docs layout:** see [`../AGENT-BEHAVIOUR/PROTOCOL.md`](../AGENT-BEHAVIOUR/PROTOCOL.md) §0.

---

## Volume layout

See [`VOLUME_LAYOUT.md`](VOLUME_LAYOUT.md) for the full current-state inspection, target tree, path defaults for `config.py`, and the migration map (`something-something-v2` → `data/ssv2`). Operator commands for migration: [`SETUP.md`](SETUP.md) step **A8**.

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
