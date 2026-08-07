from __future__ import annotations

from dataclasses import fields
import inspect

import pytest
import torch

from InfiniDepth.model.model import (
    DisparityAlignment,
    InfiniDepthEncoding,
    align_reference_disparity,
)
from InfiniDepth.model.ssr import (
    InfiniDepthSSR,
    factorize_points,
    smooth_bound_log_depth_residual,
    unfactorize_points,
    voxelize_factorized,
)
from InfiniDepth.model.ssr_geometry import (
    InfiniDepthSSRInputs,
    build_ssr_inputs,
    depth_to_points,
    fill_invalid_depth_with_median,
    make_dense_query_coord,
    scale_pixel_center_intrinsics,
)
from InfiniDepth.model.ssr_losses import affine_invariant_global_loss
from InfiniDepth.ssr_experiment import safe_path


def fake_inputs(device: torch.device, height: int = 4, width: int = 4) -> InfiniDepthSSRInputs:
    depth = torch.linspace(1.0, 1.3, height * width, device=device).reshape(1, height, width)
    intrinsics = torch.tensor(
        [[[8.0, 0.0, (width - 1) / 2], [0.0, 8.0, (height - 1) / 2], [0.0, 0.0, 1.0]]],
        device=device,
    )
    query = make_dense_query_coord(1, height, width, device=device)
    return InfiniDepthSSRInputs(
        image=torch.zeros(1, 3, height, width, device=device),
        query_coord=query,
        depth0=depth,
        valid_mask=torch.ones_like(depth, dtype=torch.bool),
        intrinsics=intrinsics,
        points0=depth_to_points(depth, intrinsics),
        dino_feature=torch.zeros(1, 1024, 1, 1, device=device),
        basic_feature=torch.zeros(1, 128, 1, 1, device=device),
        raw_disparity=depth.reciprocal(),
        alignment=(DisparityAlignment(1.0, 0.0, height * width, True, "ok"),),
        source_tags={"metric_reference": "test"},
    )


def test_dense_query_uses_yx_pixel_centers():
    query = make_dense_query_coord(1, 2, 4, device="cpu")
    assert query.shape == (1, 2, 4, 2)
    assert torch.allclose(query[0, 0, 0], torch.tensor([-0.5, -0.75]))
    assert torch.allclose(query[0, 1, 3], torch.tensor([0.5, 0.75]))


def test_pixel_center_intrinsics_scaling():
    intrinsics = torch.tensor([[100.0, 0.0, 49.5], [0.0, 120.0, 39.5], [0.0, 0.0, 1.0]])
    scaled = scale_pixel_center_intrinsics(intrinsics, (80, 100), (40, 50))
    assert torch.allclose(
        scaled,
        torch.tensor([[50.0, 0.0, 24.5], [0.0, 60.0, 19.5], [0.0, 0.0, 1.0]]),
    )


def test_backprojection_preserves_z_depth():
    depth = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    intrinsics = torch.tensor([[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]])
    points = depth_to_points(depth, intrinsics)
    assert torch.equal(points[..., 2], depth)
    expected = torch.tensor([-0.5, -0.5, 2.0])
    assert torch.allclose(points[0, 0, 0], expected)


def test_invalid_depth_fill_keeps_mask_separate():
    depth = torch.tensor([[[1.0, 0.0], [3.0, float("nan")]]])
    mask = torch.tensor([[[True, False], [True, False]]])
    filled = fill_invalid_depth_with_median(depth, mask)
    assert torch.isfinite(filled).all()
    assert filled[0, 0, 1] == 1.0
    assert torch.equal(mask, torch.tensor([[[True, False], [True, False]]]))


def test_reference_disparity_alignment_is_deterministic():
    pred = torch.linspace(0.1, 1.0, 100).reshape(10, 10)
    reference = 2.5 * pred + 0.3
    first, first_meta = align_reference_disparity(pred, reference)
    second, second_meta = align_reference_disparity(pred, reference)
    assert torch.equal(first, second)
    assert first_meta == second_meta
    assert first_meta.success
    assert first_meta.valid_count == 100
    assert torch.allclose(first, reference, rtol=1e-5, atol=1e-5)


def test_factorization_round_trip_and_voxel_order():
    points = torch.tensor([[[[-0.5, -0.25, 2.0], [0.5, 0.25, 4.0]]]])
    factorized = factorize_points(points)
    assert torch.allclose(unfactorize_points(factorized), points)
    shell = voxelize_factorized(factorized, voxel_resolution=10, num_downsamples=1)
    batch, depth, row, col = shell.coordinates.long().unbind(-1)
    assert torch.equal(batch, torch.zeros_like(batch))
    assert torch.equal(row, torch.zeros_like(row))
    assert torch.equal(col, torch.tensor([0, 1]))
    assert depth[1] > depth[0]


def test_ground_truth_is_not_an_ssr_input_or_forward_argument():
    names = {field.name for field in fields(InfiniDepthSSRInputs)}
    assert "gt" not in names
    assert "gt_depth" not in names
    parameters = inspect.signature(InfiniDepthSSR.forward).parameters
    assert "gt" not in parameters
    assert "gt_depth" not in parameters


def test_cached_base_state_is_detached_but_not_an_inference_tensor():
    class FakeBase:
        def encode_image(self, image):
            return InfiniDepthEncoding(
                dino_feature=torch.zeros(1, 1024, 24, 32),
                basic_feature=torch.zeros(1, 128, 96, 128),
                patch_height=24,
                patch_width=32,
                dino_tokens=torch.zeros(1, 24 * 32, 1024),
            )

        def decode_queries(self, encoding, query_coord, chunk_size):
            return (0.8 + 0.1 * query_coord[..., 0:1]).contiguous()

    height, width = 384, 512
    image = torch.zeros(1, 3, height, width)
    rows = torch.linspace(0, 1, height).reshape(1, height, 1)
    reference_depth = (1.0 / (1.6 + 0.2 * rows)).expand(1, height, width)
    reference_mask = torch.ones_like(reference_depth)
    intrinsics = torch.tensor(
        [[[400.0, 0.0, 255.5], [0.0, 400.0, 191.5], [0.0, 0.0, 1.0]]]
    )
    inputs = build_ssr_inputs(
        FakeBase(), image, reference_depth, reference_mask, intrinsics
    )
    for tensor in (inputs.points0, inputs.depth0, inputs.dino_feature, inputs.basic_feature):
        assert not tensor.requires_grad
        assert not torch.is_inference(tensor)


def test_masked_loss_ignores_invalid_supervision():
    pred = torch.ones(1, 2, 2, 3)
    pred[..., 2] = 2.0
    gt = pred.clone()
    gt[0, 0, 0] = torch.tensor([999.0, 999.0, 999.0])
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    mask[0, 0, 0] = False
    loss, alignment = affine_invariant_global_loss(pred, gt, mask)
    assert alignment.valid.all()
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_smooth_residual_bound_is_local_identity_and_bounded():
    raw = torch.tensor([-100.0, -1e-6, 0.0, 1e-6, 100.0])
    bounded = smooth_bound_log_depth_residual(raw, 0.1)
    assert float(bounded.abs().max()) <= 0.1 + torch.finfo(raw.dtype).eps
    assert torch.allclose(bounded[1:4], raw[1:4], rtol=1e-5, atol=1e-9)


def test_reference_backend_zero_init_and_parameter_update(tmp_path):
    inputs = fake_inputs(torch.device("cpu"))
    model = InfiniDepthSSR(
        backend="reference", channels=(8, 16), visual_channels=8, blocks_per_level=1
    )
    identity = model(inputs, K=1)
    assert torch.equal(identity.points, inputs.points0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs, K=1)
    output.depth.mean().backward()
    optimizer.step()
    changed = model(inputs, K=1)
    assert not torch.equal(changed.points, inputs.points0)

    checkpoint = tmp_path / "ssr.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = InfiniDepthSSR(
        backend="reference", channels=(8, 16), visual_channels=8, blocks_per_level=1
    )
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.allclose(restored(inputs, K=1).points, changed.points, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA server test")
def test_spconv_backend_zero_init_matches_reference_identity():
    pytest.importorskip("spconv.pytorch")
    device = torch.device("cuda")
    inputs = fake_inputs(device)
    sparse = InfiniDepthSSR(
        backend="spconv", channels=(8, 16), visual_channels=8, blocks_per_level=1
    ).to(device)
    output = sparse(inputs, K=1)
    assert torch.equal(output.points, inputs.points0)
    assert output.voxel_statistics[0]["active_voxels"] == 16


def test_safe_path_rejects_writes_outside_personal_root(tmp_path):
    root = tmp_path / "personal"
    root.mkdir()
    assert safe_path(root / "experiment", root) == (root / "experiment").resolve()
    with pytest.raises(ValueError):
        safe_path(tmp_path / "other", root)
