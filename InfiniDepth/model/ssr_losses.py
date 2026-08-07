"""Masked geometry losses and metrics for InfiniDepth SSR training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class GlobalAlignment:
    scale: torch.Tensor
    z_shift: torch.Tensor
    valid: torch.Tensor

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        shift = torch.zeros(
            (*self.z_shift.shape, 3), device=points.device, dtype=points.dtype
        )
        shift[..., 2] = self.z_shift
        return self.scale[..., None, None, None] * points + shift[..., None, None, :]

    def detached(self) -> "GlobalAlignment":
        return GlobalAlignment(self.scale.detach(), self.z_shift.detach(), self.valid.detach())


def _valid_points(gt_points: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    valid = torch.isfinite(gt_points).all(dim=-1) & (gt_points[..., 2] > 0)
    if valid_mask is not None:
        if valid_mask.shape != valid.shape:
            raise ValueError("valid_mask and point-map shapes do not match")
        valid &= valid_mask.bool()
    return valid


def solve_global_affine_alignment(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> GlobalAlignment:
    """Solve a positive global scale and camera-z translation per image."""
    if pred_points.ndim == 3:
        pred_points = pred_points.unsqueeze(0)
        gt_points = gt_points.unsqueeze(0)
        valid_mask = None if valid_mask is None else valid_mask.unsqueeze(0)
    solve_pred = pred_points.detach()
    solve_gt = gt_points.detach()
    valid = _valid_points(gt_points, valid_mask)
    scales, shifts, solved = [], [], []
    for batch_index in range(pred_points.shape[0]):
        mask = valid[batch_index]
        if int(mask.sum()) < 2:
            scales.append(pred_points.new_tensor(1.0))
            shifts.append(pred_points.new_tensor(0.0))
            solved.append(False)
            continue
        pred = solve_pred[batch_index][mask].float()
        gt = solve_gt[batch_index][mask].float()
        weight = gt[:, 2].clamp_min(1e-5).reciprocal()
        a11 = (weight[:, None] * pred.square()).sum()
        a12 = (weight * pred[:, 2]).sum()
        a22 = weight.sum()
        b1 = (weight[:, None] * pred * gt).sum()
        b2 = (weight * gt[:, 2]).sum()
        determinant = a11 * a22 - a12.square()
        if not bool(torch.isfinite(determinant)) or float(determinant.abs()) < 1e-12:
            scale = pred.new_tensor(1.0)
            shift = pred.new_tensor(0.0)
            ok = False
        else:
            scale = (b1 * a22 - b2 * a12) / determinant
            shift = (a11 * b2 - a12 * b1) / determinant
            ok = bool(torch.isfinite(scale) & torch.isfinite(shift) & (scale > 0))
            if not ok:
                scale = pred.new_tensor(1.0)
                shift = pred.new_tensor(0.0)
        scales.append(scale.to(pred_points.dtype))
        shifts.append(shift.to(pred_points.dtype))
        solved.append(ok)
    return GlobalAlignment(
        scale=torch.stack(scales),
        z_shift=torch.stack(shifts),
        valid=torch.tensor(solved, device=pred_points.device, dtype=torch.bool),
    )


def affine_invariant_global_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, GlobalAlignment]:
    alignment = solve_global_affine_alignment(pred_points, gt_points, valid_mask)
    aligned = alignment.detached().apply(pred_points)
    valid = _valid_points(gt_points, valid_mask) & alignment.valid[:, None, None]
    safe_gt = torch.where(valid[..., None], gt_points, torch.ones_like(gt_points))
    relative = (aligned - safe_gt).abs() / safe_gt[..., 2:3].clamp_min(1e-5)
    numerator = (relative * valid[..., None]).sum(dim=(-3, -2, -1))
    denominator = (valid.sum(dim=(-2, -1)) * 3).clamp_min(1)
    return numerator / denominator, alignment


def _weighted_group_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    output = values.new_zeros((group_count, values.shape[-1]))
    denominator = weights.new_zeros(group_count)
    output.index_add_(0, group_ids, values * weights[:, None])
    denominator.index_add_(0, group_ids, weights)
    return output / denominator.clamp_min(1e-8)[:, None]


def radial_partition_local_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    alignment: GlobalAlignment,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    scales: Sequence[int] = (4, 16, 64),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Multi-scale radial partition loss with local translation removal."""
    valid = _valid_points(gt_points, valid_mask)
    detached = alignment.detached()
    globally_aligned = detached.apply(pred_points)
    losses = pred_points.new_zeros(pred_points.shape[0])
    for batch_index in range(pred_points.shape[0]):
        mask = valid[batch_index]
        if not bool(mask.any()) or not bool(detached.valid[batch_index]):
            continue
        gt = gt_points[batch_index][mask]
        pred = pred_points[batch_index][mask]
        reference = gt
        weight = gt[:, 2].clamp_min(1e-5).reciprocal()
        for scale in scales:
            normalized = reference / reference.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            radius = reference.norm(dim=-1, keepdim=True).clamp_min(1e-6).log()
            keys = torch.floor(float(scale) * torch.cat((normalized, radius), dim=-1)).long()
            _, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
            group_count = int(inverse.max().item()) + 1
            residual = gt - detached.scale[batch_index] * pred
            translations = _weighted_group_mean(residual, weight, inverse, group_count)
            aligned = detached.scale[batch_index] * pred + translations[inverse]
            losses[batch_index] += (
                (aligned - gt).abs().mean(dim=-1) * weight
            ).sum() / weight.sum().clamp_min(1e-8)
    return losses


def edge_angle_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    valid = _valid_points(gt_points, valid_mask)
    safe_gt = torch.where(valid[..., None], gt_points, torch.zeros_like(gt_points))
    per_image = []
    for batch_index in range(pred_points.shape[0]):
        terms = []
        for dimension in (1, 2):
            if dimension == 1:
                pred_edge = pred_points[batch_index, 1:] - pred_points[batch_index, :-1]
                gt_edge = safe_gt[batch_index, 1:] - safe_gt[batch_index, :-1]
                edge_valid = valid[batch_index, 1:] & valid[batch_index, :-1]
            else:
                pred_edge = pred_points[batch_index, :, 1:] - pred_points[batch_index, :, :-1]
                gt_edge = safe_gt[batch_index, :, 1:] - safe_gt[batch_index, :, :-1]
                edge_valid = valid[batch_index, :, 1:] & valid[batch_index, :, :-1]
            cosine = torch.nn.functional.cosine_similarity(pred_edge, gt_edge, dim=-1, eps=1e-6)
            if bool(edge_valid.any()):
                terms.append((1.0 - cosine[edge_valid]).mean())
        per_image.append(torch.stack(terms).mean() if terms else pred_points.new_zeros(()))
    return torch.stack(per_image)


def geometry_loss_sequence(
    points_sequence: Iterable[torch.Tensor],
    gt_points: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    global_weight: float = 1.0,
    local_weight: float = 1.0,
    edge_weight: float = 1.0,
    local_scales: Sequence[int] = (4, 16, 64),
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    total = gt_points.new_zeros(())
    records: Dict[str, torch.Tensor] = {}
    for index, points in enumerate(points_sequence, start=1):
        global_loss, alignment = affine_invariant_global_loss(points, gt_points, valid_mask)
        local_loss = radial_partition_local_loss(
            points,
            gt_points,
            alignment,
            valid_mask,
            scales=local_scales,
            generator=generator,
        )
        edge_loss_value = edge_angle_loss(points, gt_points, valid_mask)
        terms = {
            "global": global_loss.mean(),
            "local": local_loss.mean(),
            "edge": edge_loss_value.mean(),
        }
        total = total + (
            global_weight * terms["global"]
            + local_weight * terms["local"]
            + edge_weight * terms["edge"]
        )
        for name, value in terms.items():
            records[f"k{index}/{name}"] = value.detach()
    records["total"] = total.detach()
    return total, records


@torch.no_grad()
def aligned_point_metrics(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    alignment = solve_global_affine_alignment(pred_points, gt_points, valid_mask)
    aligned = alignment.apply(pred_points)
    valid = _valid_points(gt_points, valid_mask) & alignment.valid[:, None, None]
    safe_gt = torch.where(valid[..., None], gt_points, torch.ones_like(gt_points))
    point_rel = (aligned - safe_gt).norm(dim=-1) / safe_gt[..., 2].clamp_min(1e-5)
    depth_rel = (aligned[..., 2] - safe_gt[..., 2]).abs() / safe_gt[..., 2].clamp_min(1e-5)
    return {
        "point_rel": float(point_rel[valid].mean()),
        "depth_rel": float(depth_rel[valid].mean()),
        "alignment_scale": float(alignment.scale.mean()),
        "alignment_z_shift": float(alignment.z_shift.mean()),
    }
