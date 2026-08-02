# HJEPA-VWM survivor ledger

- Audit directory:
- Audit UTC start:
- Old RunPod volume ID: `4hzrwzk8ja`
- RunPod termination confirmed by:
- Selected case: A / B / C / D
- Operator:

## Source locations

| Source | Account/device | Exact path/bucket/run | Read-only evidence captured | Notes |
|---|---|---|---|---|
| Mac repository | | | | |
| GitHub | | | | |
| AWS S3 | | | | |
| External disk/NAS | | | | |
| W&B runs | | | | |
| W&B artifacts | | | | |
| Qualcomm SSv2 access | | | | |
| EGO4D access | | | | |
| Hugging Face access | | | | |

## Artifact dispositions

Allowed dispositions: `exact-verified`, `exact-unverified`, `partial`, `new-derivation`,
`licensed-reacquisition`, `metadata-only`, `irrecoverable`, `blocked-pending`.

| Artifact family | Surviving source | Disposition | Object/file count | Logical bytes | SHA/inventory identity | Required action | Provenance caveat |
|---|---|---|---:|---:|---|---|---|
| `/workspace/ckpt` | | | | | | | |
| `/workspace/checkpoints` | | | | | | | |
| `/workspace/stats` | | | | | | | |
| `/workspace/preflight` | | | | | | | |
| `/workspace/logs` | | | | | | | |
| `/workspace/archive` | | | | | | | |
| Pod repo worktree | | | | | | | |
| Pod repo `.git` | | | | | | | |
| Pod repo `wandb/` | | | | | | | |
| Pod repo ignored logs/caches | | | | | | | |
| `/workspace/hf_cache` | | | | | | | |
| SSv2 raw videos | | | | | | | |
| SSv2 official labels/splits | | | | | | | |
| SSv2 full view | | | | | | | |
| SSv2 tiny view/manifest | | | | | | | |
| EGO4D metadata/tier manifest | | | | | | | |
| EGO4D UID/selection manifests | | | | | | | |
| EGO4D generated chunks | | | | | | | |
| EGO4D tiny view/manifest | | | | | | | |
| Unknown inventory paths | | | | | | | |

## Irrecoverable facts

List each lost artifact explicitly. Do not write “everything else.”

| Lost artifact | Why no surviving source contains it | Scientific consequence | Replacement/new-run decision |
|---|---|---|---|
| | | | |

## Blocked access

| Asset | Owner/provider | Request date | Expected decision/expiry | Safe work that can continue |
|---|---|---|---|---|
| | | | | |

## Path decision

- [ ] Case A: complete backup restore
- [ ] Case B: partial recovery
- [ ] Case C: ground-up fresh start
- [ ] Case D: one or more lanes remain blocked
- Primary implementation file:
- Secondary component files:
- Decision rationale:

## Evidence integrity

- `SHA256SUMS` path:
- Independent copy 1:
- Independent copy 2:
- Audit frozen UTC:
