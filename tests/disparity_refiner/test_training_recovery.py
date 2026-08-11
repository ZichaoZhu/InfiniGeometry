from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pytest
import torch

from experiment import monitor_exp2
from training.disparity_refiner.train import (
    _checkpoint_stage_state,
    _configure_stage_learning_rates,
    _restore_checkpoint,
    _restore_rng_state,
    _save_checkpoint,
    _stage_report,
)


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
