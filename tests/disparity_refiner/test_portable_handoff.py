"""Standard-library checks: no torch, GPU, dataset scan or training subprocess."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiment.exp6_4_official_ssr_disparity_adapter.automate import ROOT, prepare_config
from training.disparity_refiner import runtime_paths as paths

EXPERIMENT = ROOT / "experiment/exp6_4_official_ssr_disparity_adapter"


class PortableHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.machine = dict(safe_root=str(self.root / "work"), data_root=str(self.root / "data"),
                            cache_root=str(self.root / "work/cache"), checkpoint=str(self.root / "work/base.pt"))
        self.config = prepare_config(EXPERIMENT, "official_flex", paths=self.machine, write=False)

    def test_science_and_historical_template_unchanged(self):
        historical = prepare_config(EXPERIMENT, "official_flex", write=False)
        original = json.loads((EXPERIMENT / "config.json").read_text())
        self.assertEqual(original["model"]["refiner_backend"], "spconv")
        for key in ("training", "evaluation", "runs", "seed"):
            self.assertEqual(self.config[key], historical[key])
        self.assertEqual(self.config["model"], dict(historical["model"], checkpoint=self.machine["checkpoint"]))
        self.assertEqual(paths.source_root(self.config), self.root / "data")
        self.assertEqual(paths.source_root(historical), paths.LEGACY_SOURCE_ROOT)
        historical["data"]["source_root"] = str(self.root / "data")
        with self.assertRaises(PermissionError): paths.source_root(historical)

    def test_machine_options_cannot_override_science_or_roots(self):
        for changes in ({"learning_rate": 1}, {"data_root": "/"}, {"checkpoint": "relative.pt"},
                        {"cache_root": str(self.root / "data/cache")}, {"checkpoint": str(self.root / "other.pt")}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, PermissionError)):
                prepare_config(EXPERIMENT, "official_flex", paths=dict(self.machine, **changes), write=False)

    def test_output_boundaries_and_symlinks(self):
        self.assertEqual(paths.protected_output(self.config, self.root / "work/runs/new"), self.root / "work/runs/new")
        for target in [self.root / "work", self.root / "outside", self.root / "work/cache/run", ROOT / "runs"]:
            with self.subTest(target=target), self.assertRaises(PermissionError):
                paths.protected_output(self.config, target)
        (self.root / "work").mkdir()
        (self.root / "work/link").symlink_to(self.root / "data", target_is_directory=True)
        with self.assertRaises(PermissionError): paths.protected_output(self.config, self.root / "work/link/run")

    def test_smoke_and_immutable_configuration(self):
        destination = self.root / "config.json"
        with patch("experiment.exp6_4_official_ssr_disparity_adapter.automate.ROOT", ROOT):
            prepare_config(EXPERIMENT, "official_flex", smoke=True, paths=self.machine, destination=destination)
            before = destination.read_bytes()
            prepare_config(EXPERIMENT, "official_flex", smoke=True, paths=self.machine, destination=destination)
            self.assertEqual(destination.read_bytes(), before)
            with self.assertRaises(RuntimeError):
                prepare_config(EXPERIMENT, "official_flex", smoke=False, paths=self.machine, destination=destination)
        smoke = json.loads(before)
        self.assertEqual(smoke["training"]["stages"]["stage1"]["max_steps"], 56)
        self.assertEqual(smoke["data"]["expected_train_count"], 8)
        self.assertEqual(smoke["training"]["global_batch_size"], 8)

    def test_fixed_evaluation_hash_checks(self):
        protocol, moge = self.root / "protocol", self.root / "moge"
        source = protocol / "code/run_common_eval.py"
        masks = protocol / "local_detail/masks/hypersim_val100/moge3_v2_sam2_1_small_v1/manifest.json"
        for path in (source, masks, moge / "moge/test/metrics.py"):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fixture")
        cfg = dict(evaluation=dict(protocol_root=str(protocol), official_moge_root=str(moge)))
        digest = hashlib.sha256(b"fixture").hexdigest()
        with patch.object(paths, "PROTOCOL_SHA256", digest), patch.object(paths, "MASK_SHA256", digest):
            self.assertEqual(paths.evaluation_paths(cfg), (source, masks.parent, moge))
            source.write_text("changed")
            with self.assertRaises(RuntimeError): paths.evaluation_paths(cfg)

    def test_cli_check_only_no_writes_and_no_training(self):
        machine = self.root / "paths.json"; machine.write_text(json.dumps(self.machine))
        dest = EXPERIMENT / "local/stdlib_test_never_written.json"
        self.assertFalse(dest.exists())
        command = [sys.executable, str(EXPERIMENT / "prepare_official.py"), "--paths", str(machine),
                   "--output-config", str(dest), "--output", str(self.root / "work/run"), "--check-only"]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        record = json.loads(result.stdout)
        self.assertEqual(record["mode"], "check_only")
        self.assertEqual(record["gpu_validation"], "not_run")
        self.assertFalse(dest.exists()); self.assertFalse((self.root / "work").exists())
        command[command.index("--output-config") + 1] = str(EXPERIMENT / "config.json")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_terminal_reports_only_use_validated_main_rank_output(self):
        # Execute the actual small report helper without importing torch.
        tree = ast.parse((ROOT / "training/disparity_refiner/train.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_write_terminal_report")
        calls = []
        namespace = {"Mapping": dict, "os": type("Env", (), {"environ": {}}),
                     "_REPORT_OUTPUT": None, "_atomic_json": lambda *args: calls.append(args)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "report-helper", "exec"), namespace)
        namespace["_write_terminal_report"]({"status": "failed"})
        self.assertEqual(calls, [])
        namespace["_REPORT_OUTPUT"] = self.root / "validated"
        namespace["os"].environ["RANK"] = "1"
        namespace["_write_terminal_report"]({"status": "failed"})
        self.assertEqual(calls, [])
        namespace["os"].environ["RANK"] = "0"
        namespace["_write_terminal_report"]({"status": "paused"})
        self.assertEqual(calls, [(self.root / "validated/metrics/report.json", {"status": "paused"})])


if __name__ == "__main__":
    unittest.main()
