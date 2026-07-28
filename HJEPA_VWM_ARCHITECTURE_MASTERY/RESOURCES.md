# Resources

## Repository-primary sources

Read these before external literature when answering a project-specific question.

| Purpose | Source |
|---|---|
| Agent entry point and complete project contract | [`AGENT_FILES/AGENTS.md`](../AGENT_FILES/AGENTS.md) |
| Coding behavior | [`PROTOCOL.md`](../AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md) and [`CODE_DESIGN.md`](../AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md) |
| Shipped values | [`config.py`](../config.py) |
| Raw video path | [`data.py`](../data.py) |
| Encoder adapter contract | [`encoders.py`](../encoders.py) |
| B, B_EMA, F_c, D, mean tracker, whitener | [`models.py`](../models.py) |
| Objective definitions | [`losses.py`](../losses.py) |
| Health metrics and AGC | [`diagnostics.py`](../diagnostics.py) |
| Step, schedule, checkpoint, resume, W&B | [`train.py`](../train.py) |
| Artifact and dataset identities | [`provenance.py`](../provenance.py) |
| Offline whitening | [`whiten_stats.py`](../whiten_stats.py) |
| Rank/drift probes | [`rank_probe.py`](../rank_probe.py), [`drift_probe.py`](../drift_probe.py) |
| Experiment reading cycle | [`GUIDES/READING_EXPERIMENTS.md`](../GUIDES/READING_EXPERIMENTS.md) |
| Research chronology | [`KANBAN/PHASE_1/README.md`](../KANBAN/PHASE_1/README.md) |
| Fresh-pod setup | [`NEW_POD.md`](../AGENT_FILES/SETUPS/NEW_POD.md) |
| Remote safety and autonomy | [`GUIDE_AGENT_SSH_ACCESS.md`](../AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md), [`GUIDE_AUTONOMOUS_REMOTE_RUN.md`](../AGENT_FILES/GUIDE_AUTONOMOUS_REMOTE_RUN.md) |
| Volume topology | [`VOLUME_LAYOUT.md`](../AGENT_FILES/SETUPS/VOLUME_LAYOUT.md) |
| Disaster recovery | [`NETWORK_VOLUME_RECOVERY/README.md`](../NETWORK_VOLUME_RECOVERY/README.md) |
| Planned architecture origin | [`GUIDES/original_brief.pdf`](../GUIDES/original_brief.pdf) |

## External technical foundations

These explain the ideas that the repository adapts. They do not override the local implementation.

- [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) and
  [official V-JEPA2 ViT-L checkpoint](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256).
- [SigLIP 2 paper](https://arxiv.org/abs/2502.14786) and
  [official Base/16 256 checkpoint](https://huggingface.co/google/siglip2-base-patch16-256).
- [DINOv3 paper](https://arxiv.org/abs/2508.10104) and
  [official ViT-B/16 checkpoint](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
- [Perceiver](https://arxiv.org/abs/2103.03206) for asymmetric attention into a tight latent array.
- [ConvNeXt](https://arxiv.org/abs/2201.03545) for the local depthwise-convolution/MLP mixer pattern.
- [Flow Matching](https://arxiv.org/abs/2210.02747) for regression of vector fields along conditional
  probability paths.
- [DiT](https://arxiv.org/abs/2212.09748) for transformer diffusion/flow blocks and adaLN-Zero.
- [VICReg](https://arxiv.org/abs/2105.04906) for variance and covariance anti-collapse terms.
- [LeJEPA](https://arxiv.org/abs/2511.08544) and its
  [reference implementation](https://github.com/galilai-group/lejepa) for SIGReg background.

## Runtime and operations references

- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
- [PyTorch `clip_grad_norm_`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
- [Transformers DINOv3 docs](https://huggingface.co/docs/transformers/model_doc/dinov3).
- [Transformers SigLIP docs](https://huggingface.co/docs/transformers/model_doc/siglip).
- [W&B run resume semantics](https://docs.wandb.ai/models/runs/resuming).
- [W&B artifacts](https://docs.wandb.ai/models/artifacts).
- [RunPod SSH](https://docs.runpod.io/pods/configuration/use-ssh).
- [RunPod storage types](https://docs.runpod.io/pods/storage/types).

## How to use sources during a quiz

Answer from memory first. Then verify in this order:

1. current code;
2. tests;
3. `AGENT_FILES/AGENTS.md`;
4. current KANBAN evidence;
5. narrative briefs;
6. external papers.

If two sources disagree, do not average them. Name the disagreement and apply the authority order.
