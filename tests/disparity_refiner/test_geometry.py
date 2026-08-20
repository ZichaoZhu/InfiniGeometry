from __future__ import annotations

from pathlib import Path
import inspect
import random
from types import SimpleNamespace

import pytest
import torch

from InfiniDepth.model.disparity_refiner import (
    DisparitySparseRefiner,
    bound_disparity_residual,
    gather_features_at_coordinates,
    make_disparity_features,
    voxelize_disparity,
    VoxelDisparitySpanError,
)
from InfiniDepth.model.model import _make_dense_query_coord
from InfiniDepth.model.model import _BaseInfiniDepthModel
from training.disparity_refiner.data import ensure_within, normalize_radial_disparity
from training.disparity_refiner.losses import (
    disparity_detail_metrics,
    masked_disparity_mae,
    multiscale_gradient_loss,
    supervised_iteration_loss,
)
from training.disparity_refiner.train import _restore_checkpoint, _save_checkpoint


def test_dense_query_is_pixel_center_yx_row_major() -> None:
    query = _make_dense_query_coord(1, 384, 512, torch.device("cpu"))
    assert query.shape == (1, 384 * 512, 2)
    assert torch.equal(
        query[0, 0],
        torch.tensor([2 * 0.5 / 384 - 1, 2 * 0.5 / 512 - 1]),
    )
    assert torch.equal(
        query[0, 511],
        torch.tensor([2 * 0.5 / 384 - 1, 2 * 511.5 / 512 - 1]),
    )
    assert torch.equal(
        query[0, 512],
        torch.tensor([2 * 1.5 / 384 - 1, 2 * 0.5 / 512 - 1]),
    )


def test_features_are_x_y_disparity_while_sparse_coordinates_are_b_z_row_col() -> None:
    disparity = torch.tensor([[[-0.01, 0.00], [0.50, 1.00]]])
    features = make_disparity_features(disparity)
    assert torch.equal(features[..., 2], disparity)
    assert features[0, 0, 0, 0] < features[0, 0, 1, 0]
    assert features[0, 0, 0, 1] < features[0, 1, 0, 1]
    shell = voxelize_disparity(disparity, voxel_resolution=200, num_downsamples=1)
    assert shell.logical_disparity_bins.flatten().tolist() == [-2, 0, 100, 200]
    assert shell.disparity_offsets.tolist() == [-2]
    assert shell.coordinates.tolist() == [
        [0, 0, 0, 0],
        [0, 2, 0, 1],
        [0, 102, 1, 0],
        [0, 202, 1, 1],
    ]


def test_revoxelization_changes_only_the_logical_disparity_axis() -> None:
    base = torch.zeros(1, 2, 2)
    updated = base + torch.tensor([[[0.0, 0.01], [0.02, -0.01]]])
    before = voxelize_disparity(base, num_downsamples=0)
    after = voxelize_disparity(updated, num_downsamples=0)
    assert torch.equal(before.coordinates[:, [0, 2, 3]], after.coordinates[:, [0, 2, 3]])
    assert not torch.equal(before.logical_disparity_bins, after.logical_disparity_bins)


def test_sparse_features_are_restored_to_pixel_order() -> None:
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 2, 0, 1], [0, 1, 1, 0]], dtype=torch.int32
    )
    permutation = torch.tensor([2, 0, 1])
    sparse_coordinates = coordinates[permutation]
    sparse_features = torch.tensor([[30.0], [10.0], [20.0]])
    restored = gather_features_at_coordinates(
        sparse_features,
        sparse_coordinates,
        coordinates,
        spatial_shape=(4, 2, 2),
    )
    assert restored[:, 0].tolist() == [10.0, 20.0, 30.0]


def test_disparity_span_limit_fails_before_sparse_allocation() -> None:
    disparity = torch.tensor([[[0.0, 1.0]]])
    with pytest.raises(VoxelDisparitySpanError):
        voxelize_disparity(
            disparity,
            voxel_resolution=200,
            num_downsamples=4,
            max_disparity_span=200,
        )


def test_dense_reference_is_zero_initialized_identity() -> None:
    disparity = torch.rand(1, 3, 4)
    visual = torch.rand(1, 6, 2, 2)
    refiner = DisparitySparseRefiner(visual_dim=6, backend="reference")
    raw, stats = refiner(disparity, visual)
    bounded = bound_disparity_residual(raw)
    assert torch.equal(raw, torch.zeros_like(raw))
    assert torch.equal(disparity + bounded, disparity)
    assert stats["active_voxels"] == 12


def test_residual_bound_preserves_nonfinite_values_for_failure_detection() -> None:
    raw = torch.tensor([-100.0, 0.0, 100.0, float("nan")])
    bounded = bound_disparity_residual(raw)
    assert bounded[:3].abs().max() <= 0.1
    assert torch.isnan(bounded[-1])


def test_radial_depth_normalization_uses_inverse_depth_quantiles() -> None:
    radial = torch.tensor([[1.0, 2.0], [4.0, float("nan")]])
    mask = torch.isfinite(radial)
    normalized, (low, high) = normalize_radial_disparity(radial, mask, quantile=0.0)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(1.0)
    assert normalized[0, 0] == pytest.approx(1.0)
    assert normalized[1, 0] == pytest.approx(0.0)
    assert normalized[1, 1] == 0


def test_masked_disparity_objective_ignores_invalid_pixels() -> None:
    target = torch.tensor([[[0.0, 1.0], [2.0, 100.0]]])
    prediction = torch.tensor([[[0.0, 0.0], [1.0, -100.0]]], requires_grad=True)
    mask = torch.tensor([[[True, True], [True, False]]])
    mae = masked_disparity_mae(prediction, target, mask)
    gradient = multiscale_gradient_loss(prediction, target, mask)
    assert mae.detach().item() == pytest.approx(2 / 3)
    assert torch.isfinite(gradient)
    sequence = [prediction] * 4
    loss, values = supervised_iteration_loss(sequence, target, mask)
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 1, 1] == 0
    assert set(values) == {
        "k0_mae", "k0_gradient", "k1_mae", "k1_gradient",
        "k2_mae", "k2_gradient", "k3_mae", "k3_gradient", "loss",
    }


def test_detail_metrics_reward_aligned_depth_edges() -> None:
    radial = torch.ones(9, 9)
    radial[:, 4:] = 2.0
    target = radial.reciprocal().sub(0.5).div(0.5)
    valid = torch.ones_like(radial, dtype=torch.bool)
    perfect = disparity_detail_metrics(target, target, radial, valid, (0.5, 1.0))
    shifted = disparity_detail_metrics(
        target.roll(2, dims=1), target, radial, valid, (0.5, 1.0)
    )
    assert perfect["multiscale_gradient_error"] == pytest.approx(0.0)
    assert perfect["boundary_f1"] == pytest.approx(1.0)
    assert perfect["edge_band_mae"] == pytest.approx(0.0)
    assert shifted["multiscale_gradient_error"] > 0
    assert shifted["boundary_f1"] < 1
    assert shifted["edge_band_mae"] > 0
    assert shifted["edge_pixels"] == perfect["edge_pixels"] > 0
    flat = disparity_detail_metrics(
        torch.ones(9, 9),
        torch.ones(9, 9),
        torch.ones(9, 9),
        valid,
        (0.5, 1.0),
    )
    assert flat["boundary_f1"] is None
    assert flat["edge_band_mae"] is None
    assert flat["edge_pixels"] == 0


def test_output_path_guard_rejects_shared_or_other_user_paths() -> None:
    safe = Path("/mnt/data/home/zhuzichao")
    assert ensure_within(safe / "tmp/infinidepth_disparity_ssr", safe, name="test")
    with pytest.raises(PermissionError):
        ensure_within(Path("/nas1/datasets/hypersim/raw/output"), safe, name="test")


def test_refined_model_interface_has_no_gt_or_camera_input() -> None:
    parameters = inspect.signature(_BaseInfiniDepthModel.forward_dense_refined).parameters
    forbidden = {"gt", "target", "depth", "reference_depth", "intrinsics", "camera"}
    assert forbidden.isdisjoint(parameters)


def test_residual_scale_damps_each_iterative_update() -> None:
    class ConstantRefiner:
        def __call__(self, disparity, visual):
            return torch.full_like(disparity, 0.1), {}

    class Model:
        disparity_refiner = ConstantRefiner()
        forward_dense_refined = _BaseInfiniDepthModel.forward_dense_refined

        def encode_image(self, image):
            return SimpleNamespace(dino_features=torch.zeros(1, 1, 1, 1))

        def decode_disparity(self, encoding, query, chunk_size):
            return torch.zeros(query.shape[0], query.shape[1], 1)

    image = torch.zeros(1, 3, 2, 2)
    full = Model().forward_dense_refined(
        image, query_hw=(2, 2), num_refinement_steps=2
    )
    half = Model().forward_dense_refined(
        image, query_hw=(2, 2), num_refinement_steps=2, residual_scale=0.5
    )
    torch.testing.assert_close(
        half.disparity_sequence[1], full.disparity_sequence[1] * 0.5
    )
    torch.testing.assert_close(
        half.disparity_sequence[2], full.disparity_sequence[2] * 0.5
    )
    torch.testing.assert_close(
        half.bounded_residuals[0], full.bounded_residuals[0] * 0.5
    )
    with pytest.raises(ValueError, match="residual_scale"):
        Model().forward_dense_refined(image, residual_scale=float("nan"))


def test_best_checkpoint_is_lightweight_and_only_last_is_resumable(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    evaluation = {"aggregate": {"k3": {"full_mae": 0.1}}}
    best_path = tmp_path / "joint_best.pt"
    last_path = tmp_path / "last.pt"
    common = {
        "model": model,
        "optimizer": optimizer,
        "stage": "joint",
        "stage_step": 1000,
        "total_step": 6000,
        "config_sha256": "config-sha",
        "evaluation": evaluation,
    }
    _save_checkpoint(best_path, **common, include_optimizer=False)
    _save_checkpoint(
        last_path,
        **common,
        include_optimizer=True,
        generator=random.Random(17),
        stage_state={
            "best_evaluation": evaluation,
            "best_score": 0.1,
            "best_step": 1000,
            "final_evaluation": evaluation,
        },
    )
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    assert best["optimizer_included"] is False
    assert "optimizer" not in best
    assert last["optimizer_included"] is True
    assert "optimizer" in last
    with pytest.raises(ValueError, match="last.pt"):
        _restore_checkpoint(best_path, model, optimizer, "config-sha")
    restored = _restore_checkpoint(last_path, model, optimizer, "config-sha")
    assert restored["total_step"] == 6000
