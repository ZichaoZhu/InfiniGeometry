from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from InfiniDepth.utils.warp_utils import WarpMedian
import training.disparity_refiner.train_lidar as train_lidar_module
from training.disparity_refiner.backup import verify_checkpoint_directory
from training.disparity_refiner.data import HypersimDisparityDataset, HypersimDisparitySample
from training.disparity_refiner.lidar import (
    coarse_fine_mask,
    load_segment_masks,
    local_point_metrics,
    make_lidar_prompt,
    save_segment_masks,
    select_fine_segments,
)
from training.disparity_refiner.train import ShuffledCycleSampler
from training.disparity_refiner.train_lidar import (
    CHECKPOINT_FORMAT,
    _archive_inputs,
    _mean_metrics,
    _require_complete_local_cache,
    _validate_config,
    restore_checkpoint,
    save_checkpoint,
)


def test_resume_archives_source_without_overwriting_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    output = tmp_path / "run"
    source_files = [
        "training/disparity_refiner/train_lidar.py",
        "training/disparity_refiner/backup.py",
        "training/disparity_refiner/lidar.py",
        "training/disparity_refiner/prepare_local_masks.py",
        "training/disparity_refiner/losses.py",
        "training/disparity_refiner/data.py",
        "training/disparity_refiner/train.py",
        "InfiniDepth/model/model.py",
        "InfiniDepth/model/disparity_refiner.py",
        "InfiniDepth/utils/warp_utils.py",
    ]
    for relative in source_files:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    monkeypatch.setattr(
        train_lidar_module,
        "__file__",
        str(project / "training/disparity_refiner/train_lidar.py"),
    )
    config_path = project / "config.json"
    manifest_path = project / "manifest.json"
    sample_ids_path = project / "sample_ids.json"
    mask_manifest = project / "mask_manifest.json"
    base_checkpoint = project / "base.pt"
    for path in (
        config_path,
        manifest_path,
        sample_ids_path,
        mask_manifest,
        base_checkpoint,
    ):
        path.write_text(path.name, encoding="utf-8")
    config = {
        "data": {"manifest": "manifest.json"},
        "evaluation": {"sample_ids_config": "sample_ids.json"},
    }

    original = _archive_inputs(
        output,
        config_path,
        config,
        project,
        base_checkpoint,
        mask_manifest,
        smoke=False,
    )
    (project / "training/disparity_refiner/train_lidar.py").write_text(
        "updated", encoding="utf-8"
    )
    resumed = _archive_inputs(
        output,
        config_path,
        config,
        project,
        base_checkpoint,
        mask_manifest,
        smoke=False,
        resume_step=2500,
    )

    assert original == output / "inputs/source_manifest.json"
    assert resumed == output / "inputs/resume_step_000002500/source_manifest.json"
    assert (
        output
        / "inputs/resume_step_000002500/source/training/disparity_refiner/train_lidar.py"
    ).read_text() == "updated"
    assert (output / "inputs/source/training/disparity_refiner/train_lidar.py").read_text() != "updated"


def _sample(height: int = 32, width: int = 48) -> HypersimDisparitySample:
    radial = torch.full((height, width), 2.0)
    valid = torch.ones((height, width), dtype=torch.bool)
    return HypersimDisparitySample(
        sample_id="synthetic",
        image=torch.zeros(3, height, width),
        target_disparity=torch.zeros(height, width),
        valid_mask=valid,
        radial_depth=radial,
        disparity_quantiles=(0.1, 1.0),
        structure_mask=None,
        metadata={"M_cam_from_uv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    )


def test_lidar_prompt_is_deterministic_and_uses_prompt_median() -> None:
    sample = _sample()
    settings = {
        "vertical_beams": 8,
        "horizontal_stride": 4,
        "dropout": 0.0,
        "vertical_margin_fraction": 0.02,
    }
    first = make_lidar_prompt(sample, settings, seed=7)
    second = make_lidar_prompt(sample, settings, seed=7)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    prompt, mask, target, scale = first
    assert int(mask.sum()) == 8 * 12
    assert torch.all(prompt[mask] == 0.5)
    assert scale.item() == pytest.approx(0.5)
    assert torch.all(target[sample.valid_mask] == 1.0)


def test_lidar_prompt_accepts_metric_disparity_below_point_zero_one() -> None:
    sample = _sample()
    sample.radial_depth.fill_(200.0)
    settings = {
        "vertical_beams": 8,
        "horizontal_stride": 4,
        "dropout": 0.0,
        "vertical_margin_fraction": 0.02,
    }
    _, mask, target, scale = make_lidar_prompt(sample, settings, seed=7)
    assert int(mask.sum()) == 8 * 12
    assert scale.item() == pytest.approx(0.005)
    assert torch.all(target[sample.valid_mask] == 1.0)


def test_depthsensor_warp_does_not_require_dense_ground_truth() -> None:
    prompt = torch.zeros(1, 1, 8, 8)
    mask = torch.zeros_like(prompt, dtype=torch.bool)
    prompt[..., 1:3, 1:5] = 0.5
    mask[..., 1:3, 1:5] = True
    normalized, normalized_mask, scale = WarpMedian().warp(
        prompt, prompt_depth=prompt, prompt_mask=mask
    )
    assert scale.item() == pytest.approx(0.5)
    assert torch.all(normalized[normalized_mask] == 1.0)

    metric_prompt = torch.full((1, 1, 8, 8), 0.005)
    metric_mask = torch.ones_like(metric_prompt, dtype=torch.bool)
    normalized, normalized_mask, scale = WarpMedian().warp(
        metric_prompt, prompt_depth=metric_prompt, prompt_mask=metric_mask
    )
    assert scale.item() == pytest.approx(0.005)
    assert torch.all(normalized[normalized_mask] == 1.0)


def test_moge3_local_points_use_global_scale_and_per_segment_translation() -> None:
    sample = _sample()
    target = torch.zeros((*sample.radial_depth.shape, 3))
    target[..., 2] = 2.0
    prediction = target * 2.0
    segments = torch.zeros((2, *sample.valid_mask.shape), dtype=torch.bool)
    segments[0, :8, :8] = True
    segments[1, -8:, -8:] = True
    prediction[segments[0]] += torch.tensor([0.2, -0.1, 0.3])
    prediction[segments[1]] += torch.tensor([-0.3, 0.1, -0.2])
    metrics = local_point_metrics(prediction, target, sample.valid_mask, segments)
    assert metrics["local_segment_count"] == 2
    assert metrics["local_point_rel"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["local_point_delta_0_01"] == pytest.approx(1.0)
    assert 0.0 < metrics["local_global_scale"] < 1.0


def test_local_mask_detection_filter_and_storage(tmp_path: Path) -> None:
    disparity = np.ones((32, 48), dtype=np.float32)
    disparity[8:24, 22:25] = 2.0
    valid = np.ones_like(disparity, dtype=bool)
    coarse = coarse_fine_mask(disparity, valid)
    assert coarse[:, 22:25].any()
    good = np.zeros_like(valid)
    good[8:24, 22:25] = True
    too_large = np.ones_like(valid)
    segments = select_fine_segments([good, too_large], coarse, valid)
    assert segments.shape == (1, 32, 48)
    path = tmp_path / "segments.npz"
    save_segment_masks(path, segments)
    assert np.array_equal(load_segment_masks(path), segments)


def test_local_metric_aggregation_is_segment_macro_average() -> None:
    per_image = {
        "one": {
            "k0": {
                "local_segment_count": 1.0,
                "_local_point_rel_sum": 0.1,
                "_local_point_delta_0_01_sum": 0.9,
            }
        },
        "two": {
            "k0": {
                "local_segment_count": 3.0,
                "_local_point_rel_sum": 0.9,
                "_local_point_delta_0_01_sum": 1.5,
            }
        },
    }
    result = _mean_metrics(per_image, (0,))["k0"]
    assert result["local_point_rel"] == pytest.approx(0.25)
    assert result["local_point_delta_0_01"] == pytest.approx(0.6)
    assert result["local_segment_count"] == 4


def test_local_metrics_skip_small_segments_and_allow_overlap() -> None:
    target = torch.zeros((4, 6, 3))
    target[..., 2] = 2.0
    valid = torch.ones((4, 6), dtype=torch.bool)
    segments = torch.zeros((3, 4, 6), dtype=torch.bool)
    segments[0, :2, :5] = True
    segments[1, 1:3, :5] = True
    segments[2, 0, :3] = True
    metrics = local_point_metrics(target, target, valid, segments, min_segment_pixels=10)
    assert metrics["local_segment_count"] == 2
    assert metrics["local_point_rel"] == 0.0
    empty = local_point_metrics(target, target, torch.zeros_like(valid), segments)
    assert empty["local_segment_count"] == 0
    assert empty["local_point_rel"] is None


def test_formal_cache_preflight_rejects_missing_local_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    rgb = source / "scene/rgb.jpg"
    depth = source / "scene/depth.hdf5"
    entries = [{"id": "one", "rgb": {"source": str(rgb)}, "depth": {"source": str(depth)}}]
    dataset = HypersimDisparityDataset(
        entries,
        source_root=source,
        height=8,
        width=8,
        structure_selections={},
        cache_root=cache,
    )
    with pytest.raises(FileNotFoundError, match="refuses NAS fallback"):
        _require_complete_local_cache(dataset)
    for path in (cache / rgb.relative_to(source), cache / depth.relative_to(source)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cached")
    _require_complete_local_cache(dataset)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.disparity_refiner = torch.nn.Linear(2, 1)


def test_refiner_checkpoint_is_immutable_and_resumable(tmp_path: Path) -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.disparity_refiner.parameters(), lr=1e-3)
    sampler = ShuffledCycleSampler(11)
    sampler.sample_indices(9, 2)
    expected_sampler = ShuffledCycleSampler(0)
    expected_sampler.setstate(sampler.getstate())
    expected_next_indices = expected_sampler.sample_indices(9, 3)
    expected = {
        name: value.detach().clone() for name, value in model.disparity_refiner.state_dict().items()
    }
    checkpoint_root = tmp_path / "checkpoints"
    directory = save_checkpoint(
        checkpoint_root,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        step=500,
        config_sha256="config",
        base_checkpoint_sha256="base",
        evaluation={"aggregate": {}},
        elapsed_seconds=3.0,
    )
    payload = torch.load(directory / "checkpoint.pt", weights_only=True)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["model_scope"] == "disparity_refiner"
    assert "model" not in payload
    assert len((directory / "SHA256SUMS").read_text(encoding="ascii").splitlines()) == 2
    verify_checkpoint_directory(directory)
    assert json.loads((tmp_path / "latest.json").read_text())["path"] == (
        "checkpoints/step_000000500"
    )
    with pytest.raises(FileExistsError):
        save_checkpoint(
            checkpoint_root,
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            step=500,
            config_sha256="config",
            base_checkpoint_sha256="base",
            evaluation={},
            elapsed_seconds=4.0,
        )
    with torch.no_grad():
        for parameter in model.disparity_refiner.parameters():
            parameter.zero_()
    restore_checkpoint(
        directory,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        config_sha256="config",
        base_checkpoint_sha256="base",
    )
    for name, value in model.disparity_refiner.state_dict().items():
        assert torch.equal(value, expected[name])
    assert sampler.sample_indices(9, 3) == expected_next_indices


def test_exp4_config_has_no_small_training_stage() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "experiment/exp4_infinidepth_lidar_refiner_hypersim_full/config.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config, smoke=False)
    assert config["data"].get("sample_ids") is None
    assert config["runs"][0]["sample_ids"] is None
    assert config["training"]["checkpoint_every"] == 500
    assert config["server"]["output_root"].startswith("/mnt/data/home/zhuzichao/")
    assert config["server"]["backup_root"].startswith("/nas1/home/zhuzichao/")
    assert config["evaluation"]["selection_metric"] == "k3_metric_disparity_mae_1_per_m"
    assert config["evaluation"]["local_points"]["delta_threshold"] == 0.01
    assert "stages" not in config["training"]
    assert len(config["model"]["checkpoint_sha256"]) == 64
