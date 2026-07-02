"""Optimizer grouping and coarse-flow identity tests for Phase 1 fixes."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def _small_cfg():
    config = importlib.import_module("config")
    cfg = config.Config()
    cfg.model.h = 32
    cfg.model.w = 32
    cfg.model.n_c = 4
    cfg.model.d_e = 16
    cfg.model.d_c = 8
    cfg.model.bottleneck_mixer_dim = 8
    cfg.model.bottleneck_cross_attn_heads = 2
    cfg.model.f_c_blocks = 1
    cfg.model.f_c_heads = 2
    cfg.model.decoder_dim = 8
    cfg.model.decoder_blocks = 1
    cfg.model.decoder_heads = 2
    return cfg


def test_make_optimizer_keeps_geometry_and_zero_init_out_of_weight_decay():
    """AdamW decays normal weights but not queries, norms, biases, gates, or types."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    _, bottleneck, _, coarse_flow, decoder = models.build_phase1_modules(cfg, load_encoder=False)

    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)

    weight_decay_by_id = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            weight_decay_by_id[id(param)] = group["weight_decay"]

    assert len(optimizer.param_groups) == 6
    assert id(bottleneck.in_proj.weight) in weight_decay_by_id
    assert weight_decay_by_id[id(bottleneck.in_proj.weight)] == cfg.train.weight_decay
    assert weight_decay_by_id[id(bottleneck.cross_attn.q_proj.weight)] == cfg.train.weight_decay
    assert weight_decay_by_id[id(coarse_flow.time_mlp[0].weight)] == cfg.train.weight_decay
    assert weight_decay_by_id[id(decoder.kv_proj.weight)] == cfg.train.weight_decay

    no_decay_params = (
        bottleneck.queries,
        bottleneck.pos_emb,
        bottleneck.in_proj.bias,
        bottleneck.norm.weight,
        bottleneck.cross_attn.logit_scale,
        bottleneck.cross_attn.o_proj.weight,
        bottleneck.cross_attn.o_proj.bias,
        bottleneck.out_mlp[-1].weight,
        bottleneck.out_mlp[-1].bias,
        coarse_flow.null_condition,
        coarse_flow.slot_pos,
        coarse_flow.z_type,
        coarse_flow.cond_type,
        coarse_flow.blocks[0].mod[-1].weight,
        coarse_flow.blocks[0].mod[-1].bias,
    )
    for param in no_decay_params:
        assert weight_decay_by_id[id(param)] == 0.0

    trainable_ids = {
        id(param)
        for module in (bottleneck, coarse_flow, decoder)
        for param in module.parameters()
        if param.requires_grad
    }
    assert set(weight_decay_by_id) == trainable_ids


def test_coarse_flow_slot_type_embeddings_are_trainable_and_used():
    """CoarseFlow stamps slot and stream identity before concatenated attention."""
    models = importlib.import_module("models")
    cfg = _small_cfg()
    coarse_flow = models.CoarseFlow(cfg.model)
    coarse_flow.eval()

    z_c = torch.zeros(2, cfg.model.n_c, cfg.model.d_c)
    abstract = torch.zeros_like(z_c)
    tau = torch.zeros(2)

    assert coarse_flow.slot_pos.shape == (cfg.model.n_c, cfg.model.d_c)
    assert coarse_flow.z_type.shape == (1, 1, cfg.model.d_c)
    assert coarse_flow.cond_type.shape == (1, 1, cfg.model.d_c)
    assert coarse_flow.slot_pos.requires_grad
    assert coarse_flow.z_type.requires_grad
    assert coarse_flow.cond_type.requires_grad

    with torch.no_grad():
        stamped = coarse_flow(z_c, tau, abstract)
        coarse_flow.slot_pos.zero_()
        coarse_flow.z_type.zero_()
        unstamped = coarse_flow(z_c, tau, abstract)

    assert stamped.shape == (2, cfg.model.n_c, cfg.model.d_c)
    assert stamped.abs().sum() > 0.0
    assert torch.allclose(unstamped, torch.zeros_like(unstamped), atol=1e-6)


def test_linear_ramp_scale_handles_sigreg_warmup_edges():
    """SIGReg warmup is linear, clamped, and can be disabled with zero steps."""
    train = importlib.import_module("train")

    assert train.linear_ramp_scale(0, 2_000) == 0.0
    assert train.linear_ramp_scale(1_000, 2_000) == 0.5
    assert train.linear_ramp_scale(3_000, 2_000) == 1.0
    assert train.linear_ramp_scale(0, 0) == 1.0


def test_load_checkpoint_skips_incompatible_optimizer_state(tmp_path):
    """Old 3-group optimizer checkpoints resume model weights with a fresh optimizer."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    modules = models.build_phase1_modules(cfg, load_encoder=False)
    _, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)

    path = tmp_path / "old_optimizer.pt"
    torch.save(
        {
            "global_step": 123,
            "bottleneck": bottleneck.state_dict(),
            "target_bottleneck": target_bottleneck.state_dict(),
            "coarse_flow": coarse_flow.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": {"state": {}, "param_groups": [{"params": []}]},
        },
        path,
    )

    assert train.load_checkpoint(path, modules, optimizer) == 123


def test_load_checkpoint_rejects_learned_query_decoder_state(tmp_path):
    """Old decoder checkpoints fail clearly instead of loading per-position templates."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    modules = models.build_phase1_modules(cfg, load_encoder=False)
    _, bottleneck, target_bottleneck, coarse_flow, decoder = modules

    old_decoder_state = decoder.state_dict()
    old_decoder_state.pop("fixed_pos")
    old_decoder_state["queries"] = torch.randn(cfg.model.n_ctx, cfg.model.decoder_dim)

    path = tmp_path / "old_decoder.pt"
    torch.save(
        {
            "global_step": 123,
            "bottleneck": bottleneck.state_dict(),
            "target_bottleneck": target_bottleneck.state_dict(),
            "coarse_flow": coarse_flow.state_dict(),
            "decoder": old_decoder_state,
        },
        path,
    )

    with pytest.raises(RuntimeError, match="old learned-query reconstruction decoder"):
        train.load_checkpoint(path, modules)
