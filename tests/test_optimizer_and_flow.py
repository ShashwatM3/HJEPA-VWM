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


def test_flow_endpoint_math_and_present_residual_rejection():
    """Endpoint interpolation is exact and ambiguous present-plus-residual mode fails."""
    config = importlib.import_module("config")
    losses = importlib.import_module("losses")
    train = importlib.import_module("train")
    source = torch.tensor([[[1.0, 2.0]]])
    target = torch.tensor([[[5.0, 8.0]]])
    tau = torch.tensor([0.25])
    z_c = losses.interpolate(target, source, tau)
    velocity = losses.velocity_target(target, source)
    assert torch.equal(z_c, torch.tensor([[[2.0, 3.5]]]))
    assert torch.equal(z_c + 0.75 * velocity, target)

    cfg = config.Config()
    cfg.train.flow_source = "present"
    cfg.train.predict_residual = True
    with pytest.raises(ValueError, match="incompatible"):
        train.finalize_training_config(cfg)


@pytest.mark.parametrize("flow_source", ["present", "noise"])
def test_fixed_training_keeps_single_bottleneck_immutable(flow_source):
    """Neither fixed-arm update can mutate bottleneck parameters or buffers."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    cfg.train.flow_source = flow_source
    cfg.train.flow_bottleneck_checkpoint = "fixed-contract-active"
    cfg.train.lambda_var = 0.0
    _, bottleneck, target_bottleneck, coarse_flow, decoder = models.build_phase1_modules(
        cfg, load_encoder=False
    )
    bottleneck.eval()
    for parameter in bottleneck.parameters():
        parameter.requires_grad_(False)
    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids.isdisjoint(id(parameter) for parameter in bottleneck.parameters())

    before_parameters = {
        name: value.detach().clone() for name, value in bottleneck.named_parameters()
    }
    before_buffers = {name: value.detach().clone() for name, value in bottleneck.named_buffers()}
    modules = (
        torch.nn.Identity(),
        bottleneck,
        target_bottleneck,
        coarse_flow,
        decoder,
    )
    batch = (
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
    )
    train.train_step(batch, modules, optimizer, 0, cfg, torch.device("cpu"))

    assert not bottleneck.training
    assert all(
        torch.equal(value, before_parameters[name]) for name, value in bottleneck.named_parameters()
    )
    assert all(
        torch.equal(value, before_buffers[name]) for name, value in bottleneck.named_buffers()
    )


def test_present_euler_rollouts_start_from_present_and_report_all_contract_metrics():
    """Primary endpoints never begin from a teacher-forced future-containing state."""
    diagnostics = importlib.import_module("diagnostics")

    class RecordingFlow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.states = []

        def forward(self, state, tau, condition, *, condition_drop):
            self.states.append((state.clone(), tau.clone(), condition.clone()))
            return torch.ones_like(state)

    flow = RecordingFlow()
    present = torch.zeros(2, 3, 4)
    future = torch.ones(2, 3, 4)
    metrics = diagnostics.flow_euler_rollouts(flow, present, future, present, present)

    starts = [record for record in flow.states if torch.equal(record[1], torch.zeros(2))]
    assert len(starts) == 12
    assert all(torch.equal(record[0], present) for record in starts)
    for count in (1, 2, 4, 8):
        for condition in ("normal", "zero", "shuffled"):
            prefix = f"rollout_{count}step_{condition}"
            assert f"{prefix}_endpoint_mse" in metrics
            assert f"{prefix}_coarse_to_copy_loss_ratio" in metrics
            assert f"{prefix}_endpoint_cosine_distance" in metrics
            assert f"{prefix}_displacement_norm" in metrics
            assert f"{prefix}_displacement_alignment" in metrics


def test_noise_and_present_arms_share_fresh_coarse_flow_initialization():
    """Flow-source selection cannot alter the seed-matched fresh CoarseFlow weights."""
    models = importlib.import_module("models")
    provenance = importlib.import_module("provenance")
    cfg_noise = _small_cfg()
    cfg_present = _small_cfg()
    cfg_present.train.flow_source = "present"
    hashes = []
    for cfg in (cfg_noise, cfg_present):
        torch.manual_seed(1042)
        modules = models.build_phase1_modules(cfg, load_encoder=False)
        hashes.append(provenance.trainable_state_hash((modules[3],)))
    assert hashes[0] == hashes[1]


def test_present_build_loads_only_online_bottleneck_and_leaves_flow_fresh(tmp_path):
    """The warm start restores exact online B state without restoring checkpoint F_c."""
    models = importlib.import_module("models")
    provenance = importlib.import_module("provenance")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    torch.manual_seed(7)
    source = models.build_phase1_modules(cfg, load_encoder=False)
    with torch.no_grad():
        next(source[1].parameters()).add_(3.0)
        next(source[3].parameters()).add_(5.0)
    checkpoint = tmp_path / "run60.pt"
    torch.save(
        {"bottleneck": source[1].state_dict(), "coarse_flow": source[3].state_dict()},
        checkpoint,
    )

    cfg.train.flow_source = "present"
    cfg.train.flow_bottleneck_checkpoint = str(checkpoint)
    built = train._build_and_init(cfg, torch.device("cpu"), load_encoder=False)

    assert provenance.state_dict_hash(built[1].state_dict()) == provenance.state_dict_hash(
        source[1].state_dict()
    )
    assert provenance.state_dict_hash(built[3].state_dict()) != provenance.state_dict_hash(
        source[3].state_dict()
    )
    assert not built[1].training
    assert all(not parameter.requires_grad for parameter in built[1].parameters())


def test_fixed_noise_optimizer_excludes_bottleneck_and_uses_one_target_coordinate_system():
    """Fixed Gaussian control freezes B and uses it for condition and future target."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    cfg.train.flow_source = "noise"
    cfg.train.flow_bottleneck_checkpoint = "fixed-contract-active"
    _, bottleneck, _, coarse_flow, decoder = models.build_phase1_modules(cfg, load_encoder=False)
    for parameter in bottleneck.parameters():
        parameter.requires_grad_(False)
    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids.isdisjoint(id(parameter) for parameter in bottleneck.parameters())

    present, future, _, _ = train._fixed_flow_forward(
        torch.nn.Identity(),
        bottleneck,
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
    )
    assert not present.requires_grad
    assert not future.requires_grad
    assert present.shape == future.shape == (2, cfg.model.n_c, cfg.model.d_c)


def test_fixed_arm_rng_matches_tau_and_dropout_without_global_rng_coupling():
    """The Gaussian stream cannot shift tau, dropout, or the caller's RNG state."""
    train = importlib.import_module("train")
    cfg = _small_cfg()
    reference = torch.zeros(16, cfg.model.n_c, cfg.model.d_c)
    torch.manual_seed(99)
    before = torch.random.get_rng_state().clone()
    tau_present, drop_present, no_noise = train._fixed_flow_randomness(
        reference, 17, cfg, sample_noise=False
    )
    after_present = torch.random.get_rng_state().clone()
    tau_noise, drop_noise, noise = train._fixed_flow_randomness(
        reference, 17, cfg, sample_noise=True
    )
    after_noise = torch.random.get_rng_state().clone()

    assert torch.equal(tau_present, tau_noise)
    assert torch.equal(drop_present, drop_noise)
    assert no_noise is None
    assert noise is not None
    assert torch.equal(before, after_present)
    assert torch.equal(before, after_noise)

    future = torch.randn_like(reference)
    cfg.train.flow_source = "present"
    present_target, present_source, present_tau, present_drop = train._fixed_flow_training_inputs(
        reference, future, 17, cfg
    )
    cfg.train.flow_source = "noise"
    noise_target, noise_source, noise_tau, noise_drop = train._fixed_flow_training_inputs(
        reference, future, 17, cfg
    )
    assert present_target is future
    assert noise_target is future
    assert present_source is reference
    assert not torch.equal(noise_source, reference)
    assert torch.equal(present_tau, noise_tau)
    assert torch.equal(present_drop, noise_drop)


@pytest.mark.parametrize("flow_source", ["present", "noise"])
def test_inv020_zero_dropout_never_substitutes_null_condition(flow_source):
    """Investigation 20 records zero dropout and always passes the real condition."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    cfg.train.flow_source = flow_source
    cfg.train.flow_bottleneck_checkpoint = "fixed-contract-active"
    cfg.model.condition_dropout = 0.0
    reference = torch.randn(8, cfg.model.n_c, cfg.model.d_c)
    _, condition_drop, _ = train._fixed_flow_randomness(
        reference, 11, cfg, sample_noise=flow_source == "noise"
    )
    assert not condition_drop.any()

    flow = models.CoarseFlow(cfg.model)
    flow.train()
    with torch.no_grad():
        flow.null_condition.fill_(123.0)
    captured = []
    handle = flow.blocks[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    flow(reference, torch.zeros(reference.shape[0]), reference, condition_drop=condition_drop)
    handle.remove()
    expected_condition = reference + flow.slot_pos[None] + flow.cond_type
    assert torch.allclose(captured[0][:, cfg.model.n_c :], expected_condition)
    assert cfg.model.condition_dropout == 0.0


def test_make_optimizer_keeps_geometry_and_zero_init_out_of_weight_decay():
    """AdamW decays normal weights but not queries, norms, biases, gates, or types."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    cfg.model.bottleneck_mixer_dim = 16
    _, bottleneck, _, coarse_flow, decoder = models.build_phase1_modules(cfg, load_encoder=False)

    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)

    weight_decay_by_id = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            weight_decay_by_id[id(param)] = group["weight_decay"]

    first_block = bottleneck.latent_blocks[0]
    assert len(optimizer.param_groups) == 6
    assert id(bottleneck.in_proj.weight) in weight_decay_by_id
    assert weight_decay_by_id[id(bottleneck.in_proj.weight)] == cfg.train.weight_decay
    assert weight_decay_by_id[id(bottleneck.abstract_proj.weight)] == cfg.train.weight_decay
    assert weight_decay_by_id[id(first_block.cross_attn.q_proj.weight)] == cfg.train.weight_decay
    assert weight_decay_by_id[id(first_block.self_attn.in_proj_weight)] == cfg.train.weight_decay
    assert weight_decay_by_id[id(first_block.mlp[1].weight)] == cfg.train.weight_decay
    assert weight_decay_by_id[id(coarse_flow.time_mlp[0].weight)] == cfg.train.weight_decay
    assert weight_decay_by_id[id(decoder.kv_proj.weight)] == cfg.train.weight_decay

    no_decay_params = (
        bottleneck.queries,
        bottleneck.pos_emb,
        bottleneck.in_proj.bias,
        bottleneck.abstract_proj.bias,
        bottleneck.norm.weight,
        first_block.cross_attn.logit_scale,
        first_block.cross_attn.o_proj.weight,
        first_block.cross_attn.o_proj.bias,
        first_block.self_attn.out_proj.weight,
        first_block.self_attn.out_proj.bias,
        first_block.mlp[-1].weight,
        first_block.mlp[-1].bias,
        bottleneck.latent_blocks[-1].mlp[-1].weight,
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


def test_make_optimizer_fc_only_excludes_bottleneck_and_decoder():
    """Fc-only optimization freezes the representation and exposes only F_c groups."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    cfg.train.optimization_scope = "fc_only"
    _, bottleneck, _, coarse_flow, decoder = models.build_phase1_modules(cfg, load_encoder=False)

    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)

    assert [group["name"] for group in optimizer.param_groups] == [
        "F_c/decay",
        "F_c/no_decay",
    ]
    assert not any(param.requires_grad for param in bottleneck.parameters())
    assert all(param.requires_grad for param in coarse_flow.parameters())
    assert not any(param.requires_grad for param in decoder.parameters())
    optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert optimized == {id(param) for param in coarse_flow.parameters()}
    assert train.peak_base_lrs(bottleneck, coarse_flow, decoder, cfg) == [
        cfg.train.lr_coarse_flow,
        cfg.train.lr_coarse_flow,
    ]


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


def test_stage0_ema_transition_accepts_fp32_rounding_to_no_visible_change():
    """Tiny warmup-step EMA deltas may correctly round back to the old fp32 value."""
    train = importlib.import_module("train")
    target_before = [torch.tensor([0.02], dtype=torch.float32)]
    online_after = [torch.tensor([0.0200001], dtype=torch.float32)]
    target_after = [target_before[0].clone()]

    train._assert_ema_transition(
        online_after,
        target_before,
        target_after,
        momentum=0.996,
        updated=True,
    )


def test_stage0_ema_transition_rejects_a_missing_representable_update():
    """The Stage-0 check still fails when a required EMA change is representable."""
    train = importlib.import_module("train")
    target_before = [torch.tensor([0.02], dtype=torch.float32)]
    online_after = [torch.tensor([0.03], dtype=torch.float32)]
    target_after = [target_before[0].clone()]

    with pytest.raises(AssertionError, match="B_EMA transition incorrect"):
        train._assert_ema_transition(
            online_after,
            target_before,
            target_after,
            momentum=0.996,
            updated=True,
        )


def test_load_checkpoint_requires_explicit_incompatible_optimizer_reset(tmp_path):
    """An incompatible legacy optimizer resets only when the caller says so."""
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

    with pytest.raises(RuntimeError, match="reset_optimizer=True"):
        train.load_checkpoint(path, modules, optimizer)
    assert train.load_checkpoint(path, modules, optimizer, reset_optimizer=True) == 123


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
