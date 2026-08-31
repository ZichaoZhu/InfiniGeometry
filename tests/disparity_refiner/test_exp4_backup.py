import json
from pathlib import Path

import pytest

import training.disparity_refiner.backup as backup_module
from training.disparity_refiner.backup import (
    BackupError,
    RunBackup,
    atomic_json,
    sha256,
    verify_checkpoint_directory,
)
from training.disparity_refiner.train_lidar import _backup_on_exit


def _checkpoint(run: Path, step: int) -> Path:
    path = run / "checkpoints" / f"step_{step:09d}"
    path.mkdir(parents=True)
    (path / "checkpoint.pt").write_bytes(f"checkpoint-{step}".encode())
    atomic_json(path / "metadata.json", {"step": step})
    (path / "SHA256SUMS").write_text(
        f"{sha256(path / 'checkpoint.pt')}  checkpoint.pt\n"
        f"{sha256(path / 'metadata.json')}  metadata.json\n",
        encoding="ascii",
    )
    atomic_json(run / "latest.json", {"step": step, "path": f"checkpoints/{path.name}"})
    return path


def test_exit_backup_publishes_verified_checkpoint_and_final_report(tmp_path: Path) -> None:
    primary = tmp_path / "mnt" / "run"
    backup = tmp_path / "nas" / "run"
    checkpoint = _checkpoint(primary, 500)
    atomic_json(primary / "metrics" / "report.json", {"status": "running"})
    (primary / "logs").mkdir()
    (primary / "logs" / "events.jsonl").write_text("{}\n")
    worker = RunBackup(primary, backup, retry_delays=(0, 0, 0))
    worker.backup_checkpoint(checkpoint, step=500)
    report = {"status": "completed", "steps": 500}
    worker.finish(
        primary / "metrics" / "report.json", report, step=500, status="completed"
    )
    copied = backup / "checkpoints" / checkpoint.name
    verify_checkpoint_directory(copied, require_backup_marker=True)
    assert json.loads((backup / "metrics" / "report.json").read_text()) == report
    assert json.loads((primary / "backup_status.json").read_text())["status"] == "completed"


def test_exit_backup_retries_then_preserves_local_checkpoint(tmp_path: Path) -> None:
    primary = tmp_path / "mnt" / "run"
    checkpoint = _checkpoint(primary, 500)
    worker = RunBackup(primary, tmp_path / "nas" / "run", retry_delays=(0, 0, 0))
    calls = 0

    def fail(_: Path, __: int) -> None:
        nonlocal calls
        calls += 1
        raise OSError("NAS unavailable")

    worker._copy_checkpoint_once = fail  # type: ignore[method-assign]
    with pytest.raises(BackupError, match="NAS unavailable"):
        worker.backup_checkpoint(checkpoint, step=500)
    assert calls == 4
    verify_checkpoint_directory(checkpoint)
    status = json.loads((primary / "backup_status.json").read_text())
    assert status["status"] == "backup_failed"


def test_corrupt_checkpoint_is_never_published(tmp_path: Path) -> None:
    primary = tmp_path / "mnt" / "run"
    backup = tmp_path / "nas" / "run"
    checkpoint = _checkpoint(primary, 500)
    (checkpoint / "checkpoint.pt").write_bytes(b"corrupt")
    worker = RunBackup(primary, backup, retry_delays=(0, 0, 0))
    with pytest.raises(BackupError, match="SHA-256 mismatch"):
        worker.backup_checkpoint(checkpoint, step=500)
    assert not (backup / "checkpoints" / checkpoint.name).exists()


def test_checkpoint_verification_rejects_unexpected_member(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "run", 500)
    (checkpoint / "unexpected.bin").write_bytes(b"not in SHA256SUMS")
    with pytest.raises(ValueError, match="file set mismatch"):
        verify_checkpoint_directory(checkpoint)


def test_existing_immutable_input_is_not_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = tmp_path / "mnt" / "run"
    backup = tmp_path / "nas" / "run"
    source = primary / "inputs" / "base_checkpoint.pt"
    destination = backup / "inputs" / "base_checkpoint.pt"
    source.parent.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"base")
    destination.write_bytes(b"base")
    real_sha256 = backup_module.sha256

    def reject_input_hash(path: Path) -> str:
        if path.name == "base_checkpoint.pt":
            raise AssertionError("existing immutable input was rehashed")
        return real_sha256(path)

    monkeypatch.setattr(backup_module, "sha256", reject_input_hash)
    RunBackup(primary, backup)._copy_metadata_once()


def test_exit_only_backs_up_latest_local_checkpoint(tmp_path: Path) -> None:
    primary = tmp_path / "mnt" / "run"
    backup = tmp_path / "nas" / "run"
    first = _checkpoint(primary, 500)
    second = _checkpoint(primary, 1000)
    report_path = primary / "metrics" / "report.json"
    report = {"status": "interrupted", "last_completed_step": 1001}

    _backup_on_exit(
        RunBackup(primary, backup, retry_delays=(0, 0, 0)),
        primary,
        report_path,
        report,
        step=1001,
        status="exit_backup_completed",
    )

    assert first.is_dir()
    assert second.is_dir()
    assert not (backup / "checkpoints" / first.name).exists()
    verify_checkpoint_directory(
        backup / "checkpoints" / second.name, require_backup_marker=True
    )
