from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest
import torch

from InfiniDepth.model import InfiniDepth
from InfiniDepth.model.disparity_refiner import DisparitySparseRefiner
from InfiniDepth.model.model import acc_dtype, _InferenceState, _make_dense_query_coord


pytestmark = pytest.mark.cuda


def _checkpoint() -> Path:
    value = os.environ.get("INFINIDEPTH_CHECKPOINT")
    if not value:
        pytest.skip("INFINIDEPTH_CHECKPOINT is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"InfiniDepth checkpoint does not exist: {path}")
    return path


def _model() -> InfiniDepth:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = InfiniDepth(model_path=str(_checkpoint()))
    model.attach_disparity_refiner(backend="spconv", voxel_resolution=200)
    return model


def _image(height: int = 32, width: int = 48) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.rand(1, 3, height, width, generator=generator, device="cuda")


def _parameters(module: torch.nn.Module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _clear(model: InfiniDepth) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def test_encode_decode_matches_legacy_head_and_chunking() -> None:
    model = _model().eval()
    image = _image()
    query = torch.rand(1, 257, 2, device="cuda") * 2 - 1
    with torch.no_grad():
        encoding = model.encode_image(image)
        legacy = model.depth_implicit_head._decode_dpt(
            encoding.dino_features, encoding.basic_features, query
        )
        direct = model.decode_disparity(encoding, query)
        chunked = model.decode_disparity(encoding, query, chunk_size=31)
    torch.testing.assert_close(direct, legacy, rtol=0, atol=0)
    torch.testing.assert_close(chunked, legacy, rtol=1e-6, atol=1e-6)
    assert encoding.dino_features.shape == (1, 1024, 2, 3)
    assert encoding.basic_features.shape == (1, 128, 8, 12)
    assert encoding.patch_size == (2, 3)
    assert encoding.basic_features.dtype == torch.float32
    if torch.cuda.get_device_capability()[0] >= 8:
        assert acc_dtype == torch.bfloat16


def test_fixed_384x512_k0_batch_forward_regression_is_within_one_e_minus_six() -> None:
    model = _model().eval()
    image = _image(384, 512)
    query = _make_dense_query_coord(1, 384, 512, image.device)
    with torch.no_grad():
        features, basic, patch_h, patch_w, _ = model._prepare_backbone_features(
            image, _InferenceState()
        )
        feature_map = model.depth_implicit_head._encode_feat(
            features, patch_h, patch_w
        )
        expected_chunks = []
        for start in range(0, query.shape[1], 10000):
            expected_chunks.append(
                model.depth_implicit_head._decode_dpt(
                    feature_map,
                    basic,
                    query[:, start : start + 10000],
                )
            )
        expected = torch.cat(expected_chunks, dim=1)
        actual = model.batch_forward(image, query, bsize=10000)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_real_spconv_zero_initialization_is_bitwise_identity_for_all_k() -> None:
    model = _model().eval()
    with torch.no_grad():
        output = model.forward_dense_refined(
            _image(), query_hw=(32, 48), num_refinement_steps=5, chunk_size=256
        )
    for iteration in (1, 3, 5):
        assert torch.equal(output.disparity_sequence[0], output.disparity_sequence[iteration])
    assert all(torch.equal(raw, torch.zeros_like(raw)) for raw in output.raw_residuals)


def test_reference_and_spconv_backends_agree_on_coordinates_and_zero_identity() -> None:
    disparity = torch.rand(1, 16, 16, device="cuda")
    visual = torch.rand(1, 8, 2, 2, device="cuda")
    sparse = DisparitySparseRefiner(visual_dim=8, backend="spconv").cuda().eval()
    reference = DisparitySparseRefiner(visual_dim=8, backend="reference").cuda().eval()
    with torch.no_grad():
        sparse_raw, sparse_stats = sparse(disparity, visual)
        reference_raw, reference_stats = reference(disparity, visual)
    assert torch.equal(sparse_raw, reference_raw)
    assert sparse_stats["active_voxels"] == reference_stats["active_voxels"] == 256


def test_detached_stage_blocks_refined_gradients_but_k0_updates_base() -> None:
    model = _model().train()
    image = _image()
    target = torch.rand(1, 32, 48, device="cuda")
    output = model.forward_dense_refined(
        image, query_hw=(32, 48), num_refinement_steps=3,
        detach_base_from_refiner=True, chunk_size=256,
    )
    sum((value - target).abs().mean() for value in output.disparity_sequence[1:]).backward()
    assert all(parameter.grad is None for parameter in _parameters(model.pretrained))
    assert all(parameter.grad is None for parameter in _parameters(model.basic_encoder))
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and bool(parameter.grad.abs().sum())
        for parameter in _parameters(model.disparity_refiner)
    )

    _clear(model)
    output = model.forward_dense_refined(
        image, query_hw=(32, 48), num_refinement_steps=3,
        detach_base_from_refiner=True, chunk_size=256,
    )
    (output.disparity_sequence[0] - target).abs().mean().backward()
    assert any(parameter.grad is not None for parameter in _parameters(model.pretrained))
    assert any(parameter.grad is not None for parameter in _parameters(model.basic_encoder))


def test_joint_stage_reaches_ssr_head_and_dino_and_one_step_is_nonzero() -> None:
    model = _model().train()
    target = torch.rand(1, 32, 48, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model.forward_dense_refined(
        _image(), query_hw=(32, 48), num_refinement_steps=3,
        detach_base_from_refiner=False, chunk_size=256,
    )
    loss = sum((value - target).abs().mean() for value in output.disparity_sequence)
    loss.backward()
    for module in (model.disparity_refiner, model.basic_encoder, model.depth_implicit_head, model.pretrained):
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() and bool(parameter.grad.abs().sum())
            for parameter in _parameters(module)
        )
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        updated = model.forward_dense_refined(
            _image(), query_hw=(32, 48), num_refinement_steps=1, chunk_size=256
        )
    assert bool(updated.raw_residuals[0].abs().max() > 0)
    assert float(updated.bounded_residuals[0].abs().max()) <= 0.1


def test_eval_preserves_batch_norm_buffers_and_repeated_inference_state() -> None:
    model = _model().eval()
    batch_norms = [module for module in model.disparity_refiner.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    before = [(module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone()) for module in batch_norms]
    image = _image()
    with torch.no_grad():
        first = model.forward_dense_refined(image, query_hw=(32, 48), num_refinement_steps=3, chunk_size=256)
        second = model.forward_dense_refined(image, query_hw=(32, 48), num_refinement_steps=3, chunk_size=256)
    assert torch.equal(first.disparity, second.disparity)
    for module, snapshot in zip(batch_norms, before):
        assert torch.equal(module.running_mean, snapshot[0])
        assert torch.equal(module.running_var, snapshot[1])
        assert torch.equal(module.num_batches_tracked, snapshot[2])


def test_checkpoint_reload_reproduces_refined_output() -> None:
    model = _model().eval()
    image = _image()
    with torch.no_grad():
        expected = model.forward_dense_refined(image, query_hw=(32, 48), num_refinement_steps=1, chunk_size=256).disparity
    temporary_root = os.environ.get("INFINIDEPTH_TEST_TMP_ROOT")
    if not temporary_root:
        pytest.skip("INFINIDEPTH_TEST_TMP_ROOT must point to the personal server tmp directory")
    root = Path(temporary_root).resolve()
    safe = Path("/mnt/data/home/zhuzichao")
    if safe not in root.parents:
        raise PermissionError(f"Test temporary root is outside the personal directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        path = Path(directory) / "reload.pt"
        torch.save(model.state_dict(), path)
        with torch.no_grad():
            model.disparity_refiner.unet.output_layer.bias.add_(0.5)
        model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True), strict=True)
    with torch.no_grad():
        actual = model.forward_dense_refined(image, query_hw=(32, 48), num_refinement_steps=1, chunk_size=256).disparity
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
