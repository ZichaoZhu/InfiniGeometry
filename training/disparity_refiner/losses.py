from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def masked_disparity_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(valid.any()):
        raise ValueError("Disparity MAE received no valid pixels")
    return (prediction[valid] - target[valid]).abs().mean()


def multiscale_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scales: int = 4,
) -> torch.Tensor:
    total = prediction.new_zeros(())
    for scale in range(int(scales)):
        step = 2**scale
        pred = prediction[:, ::step, ::step]
        gt = target[:, ::step, ::step]
        valid = mask[:, ::step, ::step].bool()
        difference = torch.where(valid, pred - gt, torch.zeros_like(pred))
        horizontal_valid = valid[:, :, 1:] & valid[:, :, :-1]
        vertical_valid = valid[:, 1:, :] & valid[:, :-1, :]
        horizontal = (difference[:, :, 1:] - difference[:, :, :-1]).abs()
        vertical = (difference[:, 1:, :] - difference[:, :-1, :]).abs()
        numerator = (
            horizontal[horizontal_valid].sum()
            + vertical[vertical_valid].sum()
        )
        denominator = valid.sum().clamp_min(1).to(prediction.dtype)
        total = total + numerator / denominator
    return total


def supervised_iteration_loss(
    disparity_sequence: Sequence[torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    gradient_weight: float = 0.5,
    gradient_scales: int = 4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if len(disparity_sequence) != 4:
        raise ValueError("Training must supervise exactly K=0,1,2,3")
    losses = []
    metrics: Dict[str, torch.Tensor] = {}
    for iteration, prediction in enumerate(disparity_sequence):
        mae = masked_disparity_mae(prediction, target, mask)
        gradient = multiscale_gradient_loss(
            prediction, target, mask, scales=gradient_scales
        )
        value = mae + float(gradient_weight) * gradient
        losses.append(value)
        metrics[f"k{iteration}_mae"] = mae.detach()
        metrics[f"k{iteration}_gradient"] = gradient.detach()
    total = torch.stack(losses).mean()
    metrics["loss"] = total.detach()
    return total, metrics


@torch.no_grad()
def disparity_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    structure_mask: torch.Tensor | None = None,
) -> Dict[str, float | None]:
    full = float(masked_disparity_mae(prediction, target, valid_mask).item())
    result = {"full_mae": full}
    if structure_mask is not None:
        structure = float(
            masked_disparity_mae(prediction, target, structure_mask).item()
        )
        result["structure_mae"] = structure
        result["composite_score"] = full + structure
    return result


def _depth_edges(
    depth: torch.Tensor,
    valid: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    safe = torch.where(valid, depth.clamp_min(1e-6), torch.ones_like(depth))
    log_depth = safe.log()
    horizontal = (log_depth[:, 1:] - log_depth[:, :-1]).abs()
    vertical = (log_depth[1:, :] - log_depth[:-1, :]).abs()
    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    vertical_valid = valid[1:, :] & valid[:-1, :]
    edges = torch.zeros_like(valid)
    edges[:, 1:] |= (horizontal > threshold) & horizontal_valid
    edges[:, :-1] |= (horizontal > threshold) & horizontal_valid
    edges[1:, :] |= (vertical > threshold) & vertical_valid
    edges[:-1, :] |= (vertical > threshold) & vertical_valid
    return edges


def depth_boundary_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold: float = 0.03,
) -> float:
    """MoGe3-compatible 1-pixel-tolerant log-depth boundary F1."""
    prediction_edges = _depth_edges(prediction, valid_mask, threshold)
    target_edges = _depth_edges(target, valid_mask, threshold)
    prediction_count = int(prediction_edges.sum())
    target_count = int(target_edges.sum())
    if prediction_count == 0 or target_count == 0:
        return float(prediction_count == target_count)
    prediction_dilated = F.max_pool2d(
        prediction_edges.float()[None, None], kernel_size=3, stride=1, padding=1
    )[0, 0].bool()
    target_dilated = F.max_pool2d(
        target_edges.float()[None, None], kernel_size=3, stride=1, padding=1
    )[0, 0].bool()
    precision = (prediction_edges & target_dilated).sum().float() / prediction_edges.sum()
    recall = (target_edges & prediction_dilated).sum().float() / target_edges.sum()
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-8)).item())


@torch.no_grad()
def disparity_detail_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    radial_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    disparity_quantiles: Tuple[float, float],
    *,
    gradient_scales: int = 4,
    boundary_threshold: float = 0.03,
    edge_band_radius: int = 3,
) -> Dict[str, float]:
    low, high = (float(value) for value in disparity_quantiles)
    if high <= low:
        raise ValueError("Disparity quantiles must be increasing")
    if gradient_scales <= 0 or boundary_threshold <= 0 or edge_band_radius < 0:
        raise ValueError("Detail metric settings must be positive")
    prediction_depth = (
        prediction.float().mul(high - low).add(low).clamp_min(1e-6).reciprocal()
    )
    target_edges = _depth_edges(radial_depth.float(), valid_mask.bool(), boundary_threshold)
    full_mae = float(masked_disparity_mae(prediction, target, valid_mask).item())
    gradient_error = float(
        multiscale_gradient_loss(
            prediction[None], target[None], valid_mask[None], scales=gradient_scales
        ).item()
    )
    if not bool(target_edges.any()):
        return {
            "full_mae": full_mae,
            "multiscale_gradient_error": gradient_error,
            "boundary_f1": None,
            "edge_band_mae": None,
            "edge_pixels": 0.0,
        }
    kernel_size = 2 * int(edge_band_radius) + 1
    edge_band = F.max_pool2d(
        target_edges.float()[None, None],
        kernel_size=kernel_size,
        stride=1,
        padding=edge_band_radius,
    )[0, 0].bool() & valid_mask.bool()
    return {
        "full_mae": full_mae,
        "multiscale_gradient_error": gradient_error,
        "boundary_f1": depth_boundary_f1(
            prediction_depth,
            radial_depth.float(),
            valid_mask.bool(),
            threshold=boundary_threshold,
        ),
        "edge_band_mae": float(
            masked_disparity_mae(prediction, target, edge_band).item()
        ),
        "edge_pixels": float(edge_band.sum().item()),
    }


@torch.no_grad()
def point_cloud_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    structure_mask: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """Return the four point-cloud metrics shown by the MoGe3 viewer."""

    def scope_metrics(mask: torch.Tensor) -> Dict[str, float]:
        valid = (
            mask.bool()
            & torch.isfinite(prediction).all(dim=-1)
            & torch.isfinite(target).all(dim=-1)
            & (target[..., 2] > 0)
        )
        if not bool(valid.any()):
            raise ValueError("Point-cloud metric scope contains no valid pixels")
        safe_target = torch.where(valid[..., None], target, torch.ones_like(target))
        denominator = safe_target[..., 2].clamp_min(1e-6)
        point_rel = (prediction - safe_target).norm(dim=-1) / denominator
        depth_rel = (prediction[..., 2] - safe_target[..., 2]).abs() / denominator
        ratio = torch.maximum(
            prediction[..., 2] / denominator,
            denominator / prediction[..., 2].clamp_min(1e-6),
        )
        return {
            "pixels": float(valid.sum().item()),
            "point_rel": float(point_rel[valid].mean().item()),
            "depth_rel": float(depth_rel[valid].mean().item()),
            "depth_delta_1.01": float((ratio[valid] < 1.01).float().mean().item()),
            "depth_delta_1.25": float((ratio[valid] < 1.25).float().mean().item()),
            "boundary_f1": depth_boundary_f1(
                prediction[..., 2], target[..., 2], valid
            ),
        }

    return {
        "full": scope_metrics(valid_mask),
        "structure": scope_metrics(structure_mask),
    }
