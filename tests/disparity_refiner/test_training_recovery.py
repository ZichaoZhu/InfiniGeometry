from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pytest
import torch

from experiment import monitor_exp2
from training.disparity_refiner import train as train_module
from training.disparity_refiner.train import (
    _accumulation_context,
    _checkpoint_stage_state,
    _configure_stage_learning_rates,
    _ddp_rewrap_required,
    _gradient_accumulation,
    _peek_global_sample_indices,
    _rank_sample_indices,
    _restore_checkpoint,
    _restore_rng_state,
    DistributedContext,
    ShuffledCycleSampler,
    _save_checkpoint,
    _stage_report,
)


def test_load_samples_excludes_configured_invalid_depth(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train_module, "load_manifest", lambda *args: {})
    monkeypatch.setattr(
        train_module,
        "select_manifest_entries",
        lambda *args, **kwargs: [{"id": "good"}, {"id": "invalid"}],
    )
    config = {
        "data": {
            "excluded_sample_ids": ["invalid"],
            "expected_train_count": 1,
            "lazy_loading": True,
            "local_cache": None,
            "manifest": "manifest.json",
            "manifest_sha256": "unused",
            "source_root": "/nas1/datasets/hypersim/raw",
        },
        "model": {"height": 384, "width": 512},
        "server": {"cache": str(tmp_path / "cache")},
        "structure_selections": [],
    }

    samples = train_module._load_samples(config, {"sample_ids": None}, tmp_path)

    assert samples.sample_ids == ["good"]


def evaluation(k0: float = 2.0, k3: float = 1.0) -> dict[str, object]:
    return {
        "aggregate": {
            "k0": {"full_mae": k0},
            "k1": {"full_mae": k3},
            "k3": {"full_mae": k3},
            "k5": {"full_mae": k3},
        },
        "k3_better_than_k0_count": 1,
        "per_image": {},
        "sample_count": 1,
    }


def test_resume_restores_rng_best_state_and_completed_stage_report(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = random.Random(17)
    random.seed(18)
    np.random.seed(19)
    torch.manual_seed(20)
    stage1 = _stage_report(
        "stage1",
        20_000,
        {
            "best_evaluation": evaluation(),
            "best_score": 1.0,
            "best_step": 17_500,
            "elapsed_seconds": 8.0,
            "final_evaluation": evaluation(2.1, 1.1),
        },
    )
    stage_state = {
        "best_evaluation": evaluation(1.9, 0.9),
        "best_score": 0.9,
        "best_step": 2_500,
        "elapsed_seconds": 9.0,
        "final_evaluation": evaluation(),
        "previous_full_score": 1.0,
        "stale_evaluations": 1,
    }
    path = tmp_path / "last.pt"
    _save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        stage="joint",
        stage_step=2_500,
        total_step=22_500,
        config_sha256="config",
        evaluation=evaluation(),
        include_optimizer=True,
        generator=generator,
        stage_state=stage_state,
        completed_stage_reports=[stage1],
    )
    expected = (
        generator.randrange(100),
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
    )
    generator.seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    restored = _restore_checkpoint(path, model, optimizer, "config")
    assert _restore_rng_state(restored, generator)
    actual = (
        generator.randrange(100),
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
    )
    assert actual == expected
    assert _checkpoint_stage_state(restored)["best_step"] == 2_500
    reports = restored["resume_state"]["completed_stage_reports"]
    assert reports[0]["stage"] == "stage1"


def test_shuffled_cycle_sampler_covers_dataset_and_restores_position() -> None:
    sampler = ShuffledCycleSampler(17)
    first = sampler.sample_indices(7, 4)
    state = sampler.getstate()
    expected = sampler.sample_indices(7, 6)
    restored = ShuffledCycleSampler(0)
    restored.setstate(state)
    assert restored.sample_indices(7, 6) == expected
    assert sorted(first + expected[:3]) == list(range(7))


def test_stage_boundary_and_legacy_checkpoint_state_are_reconstructable() -> None:
    state = _checkpoint_stage_state(
        {
            "evaluation": evaluation(),
            "stage_step": 20_000,
        }
    )
    report = _stage_report("stage1", 20_000, state)
    assert report["completed_steps"] == 20_000
    assert report["best_step"] == 20_000
    assert report["best_evaluation"] == evaluation()


def test_dino_freeze_and_warmup_schedule() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {"name": "ssr", "params": [], "lr": 0.0},
            {"name": "head", "params": [], "lr": 0.0},
            {"name": "dino", "params": [parameter], "lr": 0.0},
        ]
    )
    config = {
        "dino_freeze_steps": 1000,
        "dino_warmup_end": 2000,
        "learning_rates": {"ssr": 1e-5, "head": 1e-6, "dino": 5e-8},
    }
    assert _configure_stage_learning_rates(optimizer, [parameter], config, 1000)["dino"] == 0
    assert not parameter.requires_grad
    assert _configure_stage_learning_rates(optimizer, [parameter], config, 1500)["dino"] == pytest.approx(2.5e-8)
    assert parameter.requires_grad
    assert _configure_stage_learning_rates(optimizer, [parameter], config, 2000)["dino"] == pytest.approx(5e-8)


@pytest.mark.parametrize(
    ("world_size", "accumulation"), [(1, 8), (2, 4), (4, 2)]
)
def test_global_batch_formula(world_size: int, accumulation: int) -> None:
    assert _gradient_accumulation(8, 1, world_size) == accumulation


def test_global_batch_rejects_non_divisible_world_size() -> None:
    with pytest.raises(ValueError, match=r"microbatch_size \* world_size"):
        _gradient_accumulation(8, 3, 2)


@pytest.mark.parametrize("world_size", [2, 4])
def test_global_sample_batch_is_sharded_without_overlap(world_size: int) -> None:
    sampler = ShuffledCycleSampler(17)
    accumulation = _gradient_accumulation(8, 1, world_size)
    merged = []
    expected = []
    for _ in range(accumulation):
        global_indices = sampler.sample_indices(100, world_size)
        expected.extend(global_indices)
        rank_values = [
            _rank_sample_indices(global_indices, rank, 1, world_size)
            for rank in range(world_size)
        ]
        assert len({value for values in rank_values for value in values}) == world_size
        merged.extend(value for values in rank_values for value in values)
    assert merged == expected


def test_sampler_resume_preserves_next_global_batch() -> None:
    sampler = ShuffledCycleSampler(17)
    sampler.sample_indices(100, 8)
    state = sampler.getstate()
    expected = sampler.sample_indices(100, 8)
    restored = ShuffledCycleSampler(0)
    restored.setstate(state)
    assert restored.sample_indices(100, 8) == expected


def test_peek_global_sample_indices_does_not_advance_sampler() -> None:
    sampler = ShuffledCycleSampler(17)
    expected = _peek_global_sample_indices(sampler, 100, 2, 4)
    assert sampler.sample_indices(100, 8) == expected


def test_dino_rewrap_boundary_and_no_sync_count() -> None:
    assert not _ddp_rewrap_required(False, False)
    assert _ddp_rewrap_required(False, True)

    class Wrapper:
        calls = 0

        def no_sync(self):
            self.calls += 1
            return nullcontext()

    wrapper = Wrapper()
    distributed = DistributedContext(0, 0, 2, torch.device("cpu"), "gloo")
    for step in range(4):
        with _accumulation_context(wrapper, distributed, step, 4):
            pass
    assert wrapper.calls == 3


def test_distributed_checkpoint_requires_same_world_size(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "last.pt"
    _save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        stage="stage1",
        stage_step=1,
        total_step=1,
        config_sha256="config",
        evaluation=evaluation(),
        include_optimizer=True,
        generator=random.Random(17),
        stage_state={"best_score": 1.0},
        distributed_state={"world_size": 2},
    )
    with pytest.raises(ValueError, match="cannot resume"):
        _restore_checkpoint(path, model, optimizer, "config", world_size=4)
    assert _restore_checkpoint(path, model, optimizer, "config", world_size=2)


def test_legacy_checkpoint_remains_single_gpu_compatible(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "legacy.pt"
    _save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        stage="stage1",
        stage_step=1,
        total_step=1,
        config_sha256="config",
        evaluation=evaluation(),
        include_optimizer=True,
        generator=random.Random(17),
        stage_state={"best_score": 1.0},
    )
    assert _restore_checkpoint(path, model, optimizer, "config", world_size=1)
    with pytest.raises(ValueError, match="Legacy single-GPU"):
        _restore_checkpoint(path, model, optimizer, "config", world_size=2)


def make_monitor_experiment(tmp_path: Path, record: dict[str, object]) -> Path:
    experiment = tmp_path / monitor_exp2.EXP2_NAME
    run = experiment / "runs" / "main"
    (run / "metrics").mkdir(parents=True)
    (experiment / "config.json").write_text(
        json.dumps(
            {
                "training": {
                    "stages": {
                        "stage1": {"max_steps": 20_000},
                        "joint": {"max_steps": 10_000},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics" / "history.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    return experiment


def monitor_record(*, k0: float = 2.0, k3: float = 1.0, loss: float = 0.5) -> dict[str, object]:
    return {
        "elapsed_seconds": 100.0,
        "evaluation": evaluation(k0, k3),
        "gradient_norms": {"dino": 0.1, "head": 0.2, "ssr": 0.3},
        "scope": "full",
        "stage": "stage1",
        "stage_step": 500,
        "total_step": 500,
        "training": {
            "loss": loss,
            "k1_bounded_residual_max": 0.05,
            "k1_bounded_residual_min": -0.05,
        },
    }


def test_monitor_detects_health_stale_nonfinite_identity_exit_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = make_monitor_experiment(tmp_path, monitor_record())
    monkeypatch.setattr(monitor_exp2, "process_is_alive", lambda pid: True)
    monkeypatch.setattr(monitor_exp2, "process_identity_error", lambda pid, path: None)
    status = monitor_exp2.check_once(experiment, 123, 7200)
    assert status["health"] == "healthy"
    assert status["progress_fraction"] == pytest.approx(1 / 60)

    history = experiment / "runs" / "main" / "metrics" / "history.jsonl"
    os.utime(history, (1, 1))
    status = monitor_exp2.check_once(experiment, 123, 100, now=1000)
    assert status["health"] == "alert"
    assert json.loads((experiment / "runs" / "main" / "monitor" / "alert.json").read_text())["kind"] == "evaluation_stale"

    history.write_text(json.dumps(monitor_record(loss=float("nan"))) + "\n", encoding="utf-8")
    status = monitor_exp2.check_once(experiment, 123, 7200, now=time.time())
    assert status["health"] == "alert"
    assert json.loads((experiment / "runs" / "main" / "monitor" / "alert.json").read_text())["kind"] == "non_finite_metrics"

    history.write_text(json.dumps(monitor_record()) + "\n", encoding="utf-8")
    monkeypatch.setattr(monitor_exp2, "process_identity_error", lambda pid, path: "wrong command")
    monitor_exp2.check_once(experiment, 123, 7200)
    assert json.loads((experiment / "runs" / "main" / "monitor" / "alert.json").read_text())["kind"] == "process_identity_mismatch"

    monkeypatch.setattr(monitor_exp2, "process_is_alive", lambda pid: False)
    monitor_exp2.check_once(experiment, 123, 7200)
    assert json.loads((experiment / "runs" / "main" / "monitor" / "alert.json").read_text())["kind"] == "training_exited"

    report = experiment / "runs" / "main" / "metrics" / "report.json"
    report.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    status = monitor_exp2.check_once(experiment, 123, 7200)
    assert status["health"] == "completed"
    assert json.loads((experiment / "runs" / "main" / "monitor" / "alert.json").read_text())["kind"] == "training_completed"


def test_monitor_warns_after_two_full_evaluations_without_ssr_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = make_monitor_experiment(tmp_path, monitor_record(k0=1.0, k3=1.1))
    history = experiment / "runs" / "main" / "metrics" / "history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(monitor_record(k0=1.0, k3=1.2)) + "\n")
    monkeypatch.setattr(monitor_exp2, "process_is_alive", lambda pid: True)
    monkeypatch.setattr(monitor_exp2, "process_identity_error", lambda pid, path: None)
    monitor_exp2.check_once(experiment, 123, 7200)
    alert = json.loads((experiment / "runs" / "main" / "monitor" / "alert.json").read_text())
    assert alert["kind"] == "ssr_no_gain_two_full_evals"


def test_monitor_keeps_startup_stale_anchor_and_deduplicates_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = make_monitor_experiment(tmp_path, monitor_record())
    (experiment / "runs" / "main" / "metrics" / "history.jsonl").unlink()
    monkeypatch.setattr(monitor_exp2, "process_is_alive", lambda pid: True)
    monkeypatch.setattr(monitor_exp2, "process_identity_error", lambda pid, path: None)
    first = monitor_exp2.check_once(experiment, 123, 100, now=1000)
    assert first["health"] == "starting"
    monitor_exp2.check_once(experiment, 123, 100, now=1200)
    alert_path = experiment / "runs" / "main" / "monitor" / "alert.json"
    first_alert = json.loads(alert_path.read_text())
    monitor_exp2.check_once(experiment, 123, 100, now=1300)
    second_alert = json.loads(alert_path.read_text())
    assert first_alert["event_id"] == second_alert["event_id"]
    assert first_alert["created_at_unix"] == second_alert["created_at_unix"]
