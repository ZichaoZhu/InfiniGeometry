import torch

from training.disparity_refiner.losses import point_cloud_metrics


def test_point_cloud_metrics_match_perfect_geometry() -> None:
    target = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]], [[0.0, 1.0, 1.0], [1.0, 1.0, 2.0]]]
    )
    metrics = point_cloud_metrics(
        target, target, torch.ones((2, 2), dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)
    )
    assert metrics["full"]["point_rel"] == 0.0
    assert metrics["full"]["depth_rel"] == 0.0
    assert metrics["full"]["depth_delta_1.01"] == 1.0
    assert metrics["full"]["boundary_f1"] == 1.0
