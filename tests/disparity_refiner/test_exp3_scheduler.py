from __future__ import annotations

import json

from experiment.prepare_exp3_cache import file_list
from experiment.schedule_exp3 import (
    build_train_command,
    Scheduler,
    choose_gpus,
    external_compute_pids,
    GpuState,
    is_strictly_idle,
    update_idle_counts,
)


def gpu(
    index: int,
    *,
    free: int = 24_000,
    utilization: int = 0,
    pids: tuple[int, ...] = (),
) -> GpuState:
    return GpuState(index, f"gpu-{index}", "RTX 4090", free, utilization, pids)


def test_idle_requires_memory_utilization_and_zero_compute_processes() -> None:
    assert is_strictly_idle(gpu(0), 22_528, 10)
    assert not is_strictly_idle(gpu(0, free=22_000), 22_528, 10)
    assert not is_strictly_idle(gpu(0, utilization=11), 22_528, 10)
    assert not is_strictly_idle(gpu(0, pids=(123,)), 22_528, 10)


def test_idle_counts_reset_and_select_four_before_two() -> None:
    states = [gpu(index) for index in range(4)]
    counts: dict[str, int] = {}
    for _ in range(3):
        counts = update_idle_counts(
            states,
            counts,
            minimum_free_memory_mib=22_528,
            maximum_utilization_percent=10,
        )
    assert [value.index for value in choose_gpus(
        states,
        counts,
        required_idle_checks=3,
        minimum_gpus=2,
        maximum_gpus=4,
    )] == [0, 1, 2, 3]
    states[2] = gpu(2, pids=(123,))
    counts = update_idle_counts(
        states,
        counts,
        minimum_free_memory_mib=22_528,
        maximum_utilization_percent=10,
    )
    assert counts["gpu-2"] == 0
    assert [value.index for value in choose_gpus(
        states,
        counts,
        required_idle_checks=3,
        minimum_gpus=2,
        maximum_gpus=4,
    )] == [0, 1]


def test_v06_two_gpu_selection_requires_both_cards() -> None:
    states = [gpu(0), gpu(1)]
    counts = {"gpu-0": 3, "gpu-1": 2}
    assert not choose_gpus(
        states,
        counts,
        required_idle_checks=3,
        minimum_gpus=2,
        maximum_gpus=2,
    )
    counts["gpu-1"] = 3
    assert [value.index for value in choose_gpus(
        states,
        counts,
        required_idle_checks=3,
        minimum_gpus=2,
        maximum_gpus=2,
    )] == [0, 1]


def test_external_process_detection_ignores_only_own_process_tree(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiment.schedule_exp3.child_processes", lambda pid: {pid, pid + 1}
    )
    states = [gpu(0, pids=(100, 101, 999)), gpu(1, pids=(101,))]
    assert external_compute_pids(states, 100) == [999]


def test_scheduler_host_can_be_overridden_without_changing_config(tmp_path) -> None:
    experiment = tmp_path / "experiment" / "exp3"
    experiment.mkdir(parents=True)
    config = {
        "scheduler": {
            "host": "ZJU3DV-V06",
            "poll_interval_seconds": 300,
            "required_idle_checks": 3,
            "minimum_gpus": 2,
            "maximum_gpus": 2,
            "minimum_free_memory_mib": 22_528,
            "maximum_utilization_percent": 10,
        }
    }
    path = experiment / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    scheduler = Scheduler(experiment, host="ZJU3DV-4090")

    assert scheduler.host == "ZJU3DV-4090"
    assert json.loads(path.read_text(encoding="utf-8"))["scheduler"]["host"] == "ZJU3DV-V06"


def test_single_gpu_revision_keeps_config_and_uses_single_process(tmp_path) -> None:
    experiment = tmp_path / "experiment" / "exp3"
    experiment.mkdir(parents=True)
    config = {
        "scheduler": {
            "host": "ZJU3DV-V06",
            "poll_interval_seconds": 300,
            "required_idle_checks": 3,
            "minimum_gpus": 2,
            "maximum_gpus": 2,
            "minimum_free_memory_mib": 22_528,
            "maximum_utilization_percent": 10,
        }
    }
    path = experiment / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    scheduler = Scheduler(experiment, world_size=1)

    command = build_train_command(
        world_size=1,
        config=path,
        output=experiment / "runs/main",
    )

    assert scheduler.minimum_gpus == scheduler.maximum_gpus == 1
    assert "torchrun" not in command[0]
    assert "--resume" not in command


def test_cache_list_contains_only_train_and_selected_validation(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    index = source / "ml-hypersim/evermotion_dataset/analysis"
    index.mkdir(parents=True)
    (index / "metadata_images_split_scene_v1.csv").write_text(
        "scene_name,camera_name,frame_id,split_partition_name\n"
        "train_scene,cam_00,1,train\n"
        "kept_val,cam_01,2,val\n"
        "skipped_val,cam_00,3,val\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("experiment.prepare_exp3_cache.SOURCE_ROOT", source)
    config = {
        "data": {
            "excluded_sample_ids": ["train_scene_cam_00_frame.0001"],
            "expected_train_count": 0,
        },
        "evaluation": {"full_sample_ids": ["kept_val_cam_01_frame.0002"]},
    }

    paths = file_list(config)

    assert len(paths) == 4
    assert all("skipped_val" not in path for path in paths)
