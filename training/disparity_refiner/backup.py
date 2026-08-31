from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping, Optional, Sequence


class BackupError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_file_sizes(path: Path, *, include_backup_marker: bool = False) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"Checkpoint member may not be a symlink: {candidate}")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path).as_posix()
        if relative == "BACKUP_COMPLETE.json" and not include_backup_marker:
            continue
        sizes[relative] = candidate.stat().st_size
    return sizes


def verify_checkpoint_directory(path: Path, *, require_backup_marker: bool = False) -> None:
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError(path)
    sums = path / "SHA256SUMS"
    if not sums.is_file() or sums.is_symlink():
        raise ValueError(f"Missing checkpoint SHA256SUMS: {path}")
    expected_members: set[str] = set()
    for line in sums.read_text(encoding="ascii").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or not expected or not relative:
            raise ValueError(f"Invalid checkpoint SHA256SUMS line: {line!r}")
        if relative in expected_members:
            raise ValueError(f"Duplicate checkpoint SHA256SUMS member: {relative}")
        candidate = (path / relative).resolve()
        if path.resolve() not in candidate.parents or not candidate.is_file():
            raise ValueError(f"Unsafe or missing checkpoint member: {relative}")
        if sha256(candidate) != expected:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {candidate}")
        expected_members.add(relative)
    if not expected_members:
        raise ValueError(f"Empty checkpoint SHA256SUMS: {path}")
    marker = path / "BACKUP_COMPLETE.json"
    if require_backup_marker and not marker.is_file():
        raise ValueError(f"NAS checkpoint is not marked complete: {path}")
    expected_files = expected_members | {"SHA256SUMS"}
    if require_backup_marker:
        expected_files.add("BACKUP_COMPLETE.json")
    actual_files = set(_checkpoint_file_sizes(path, include_backup_marker=True))
    if actual_files != expected_files:
        raise ValueError(
            f"Checkpoint file set mismatch: expected {sorted(expected_files)}, "
            f"found {sorted(actual_files)}"
        )


def atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = sha256(source)
    source_bytes = source.stat().st_size
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    shutil.copyfile(source, temporary)
    if temporary.stat().st_size != source_bytes or sha256(temporary) != source_digest:
        temporary.unlink(missing_ok=True)
        raise IOError(f"Incomplete backup copy: {source} -> {destination}")
    temporary.replace(destination)


class RunBackup:
    """Synchronously publish the latest local checkpoint when a run exits."""

    def __init__(
        self,
        source_run: Path,
        backup_run: Path,
        *,
        retry_delays: Sequence[float] = (5.0, 15.0, 45.0),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.source_run = source_run.resolve()
        self.backup_run = backup_run.resolve()
        if self.source_run == self.backup_run:
            raise ValueError("Primary and backup run directories must differ")
        self.retry_delays = tuple(float(value) for value in retry_delays)
        self.sleep = sleep
        self.copied_immutable: set[Path] = set()

    @property
    def status_path(self) -> Path:
        return self.source_run / "backup_status.json"

    def _status(self, status: str, *, step: Optional[int] = None, error: str = "") -> None:
        value: dict[str, object] = {
            "format": "infinidepth-exp4-backup-v1",
            "status": status,
            "primary_run": str(self.source_run),
            "backup_run": str(self.backup_run),
            "updated_at_unix": time.time(),
        }
        if step is not None:
            value["step"] = int(step)
        if error:
            value["error"] = error
        atomic_json(self.status_path, value)

    def _copy_metadata_once(self) -> None:
        directory_names = {"inputs", "metrics", "logs", "artifacts"}
        for source in self.source_run.rglob("*"):
            if not source.is_file() or source.is_symlink():
                continue
            relative = source.relative_to(self.source_run)
            if relative.parts[0] == "checkpoints" or source == self.status_path:
                continue
            if len(relative.parts) > 1 and relative.parts[0] not in directory_names:
                continue
            destination = self.backup_run / relative
            if relative.parts[0] == "inputs":
                if relative in self.copied_immutable:
                    continue
                if destination.exists():
                    # Inputs are immutable and atomically verified on first publication.
                    if source.stat().st_size != destination.stat().st_size:
                        raise FileExistsError(f"Archived input differs on NAS: {destination}")
                else:
                    atomic_copy(source, destination)
                self.copied_immutable.add(relative)
            else:
                atomic_copy(source, destination)

    def _copy_checkpoint_once(self, checkpoint: Path, step: int) -> None:
        verify_checkpoint_directory(checkpoint)
        source_sizes = _checkpoint_file_sizes(checkpoint)
        destination = self.backup_run / "checkpoints" / checkpoint.name
        if destination.exists():
            verify_checkpoint_directory(destination, require_backup_marker=True)
            destination_sizes = _checkpoint_file_sizes(destination)
            if destination_sizes != source_sizes:
                raise IOError(f"NAS checkpoint file count or size mismatch: {destination}")
            marker = json.loads(
                (destination / "BACKUP_COMPLETE.json").read_text(encoding="utf-8")
            )
            if marker.get("source_sha256sums") != sha256(checkpoint / "SHA256SUMS"):
                raise FileExistsError(f"Existing NAS checkpoint differs: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            try:
                shutil.copytree(checkpoint, temporary)
                verify_checkpoint_directory(temporary)
                if _checkpoint_file_sizes(temporary) != source_sizes:
                    raise IOError(
                        f"NAS checkpoint file count or size mismatch: {temporary}"
                    )
                atomic_json(
                    temporary / "BACKUP_COMPLETE.json",
                    {
                        "format": "infinidepth-exp4-checkpoint-backup-v1",
                        "step": int(step),
                        "source": str(checkpoint),
                        "source_sha256sums": sha256(checkpoint / "SHA256SUMS"),
                        "completed_at_unix": time.time(),
                    },
                )
                temporary.replace(destination)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

    def backup_checkpoint(self, checkpoint: Path, *, step: int) -> None:
        checkpoint = checkpoint.resolve()
        self._status("copying", step=step)
        delays = (0.0, *self.retry_delays)
        last_error: Optional[Exception] = None
        for delay in delays:
            if delay:
                self.sleep(delay)
            try:
                self._copy_checkpoint_once(checkpoint, step)
                self._status("checkpoint_verified", step=step)
                atomic_copy(self.status_path, self.backup_run / self.status_path.name)
                return
            except Exception as exc:  # noqa: BLE001 - preserve the local checkpoint on any I/O failure
                last_error = exc
        assert last_error is not None
        self._status("backup_failed", step=step, error=str(last_error))
        raise BackupError(f"Checkpoint backup failed at step {step}: {last_error}") from last_error

    def finish(
        self,
        report_path: Path,
        report: Mapping[str, object],
        *,
        step: int,
        status: str,
    ) -> None:
        last_error: Optional[Exception] = None
        for delay in (0.0, *self.retry_delays):
            if delay:
                self.sleep(delay)
            try:
                self._copy_metadata_once()
                atomic_json(self.backup_run / report_path.relative_to(self.source_run), report)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - final NAS publication is all-or-nothing
                last_error = exc
        if last_error is not None:
            self._status("backup_failed", step=step, error=str(last_error))
            raise BackupError(f"Final backup publication failed: {last_error}") from last_error
        try:
            self._status(status, step=step)
            atomic_copy(self.status_path, self.backup_run / self.status_path.name)
        except Exception as exc:  # noqa: BLE001 - completion requires a published NAS status
            self._status("backup_failed", step=step, error=str(exc))
            raise BackupError(f"Final backup status publication failed: {exc}") from exc
        atomic_json(report_path, report)
