import json
import os
import shutil
import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_experiments as runner
from factory_supervisor import FactorySupervisor


class WebotsLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = ROOT / ".test_tmp_safe" / (
            "webots_launcher_" + uuid.uuid4().hex)
        self.temp_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_result(self, *, seed, runtime_mode="webots", suffix="run"):
        path = self.temp_dir / (
            f"experiment_A_Hungarian_20260101_000000_{suffix}.json")
        path.write_text(json.dumps({
            "experiment_info": {
                "scenario": "A",
                "scheduler": "Hungarian",
                "seed": seed,
                "runtime_mode": runtime_mode,
            },
            "summary_metrics": {"total_tasks_completed": 1},
        }), encoding="utf-8")
        return path

    def test_external_world_changes_only_controller_ownership(self):
        target = self.temp_dir / "smart_factory_extern.wbt"
        runner._make_external_world(str(target))
        source = Path(runner.WORLD_FILE).read_text(encoding="utf-8")
        external = target.read_text(encoding="utf-8")
        self.assertEqual(8, source.count('controller "robot_controller"'))
        self.assertEqual(1, source.count('controller "factory_supervisor"'))
        self.assertEqual(9, external.count('controller "<extern>"'))
        restored = external.replace(
            'controller "<extern>"', 'controller "robot_controller"', 8)
        restored = restored.replace(
            'controller "<extern>"', 'controller "factory_supervisor"', 1)
        self.assertEqual(source, restored)

    def test_load_latest_results_requires_seed_runtime_and_freshness(self):
        wrong_seed = self._write_result(seed=10, suffix="wrong_seed")
        stale = self._write_result(seed=11, suffix="stale")
        fresh = self._write_result(seed=11, suffix="fresh")
        old_time = time.time() - 100
        os.utime(stale, (old_time, old_time))
        now = time.time()
        os.utime(wrong_seed, (now + 1, now + 1))
        os.utime(fresh, (now, now))
        with mock.patch.object(runner, "RESULTS_DIR", str(self.temp_dir)):
            result = runner.load_latest_results(
                "A", "Hungarian", seed=11, runtime_mode="webots",
                min_mtime=now - 1)
        self.assertEqual(11, result["experiment_info"]["seed"])

    def test_auto_mode_uses_external_tcp_when_qprocess_is_blocked(self):
        def fake_external(_path, _scenario, _env, _show, _timeout):
            self._write_result(seed=8204, suffix="external")
            return True

        preflight = mock.Mock(
            returncode=0, stdout="Webots version: R2023b",
            stderr="Critical: QProcess: CreateFile failed. (access denied)")
        with mock.patch.object(runner, "RESULTS_DIR", str(self.temp_dir)), \
                mock.patch.object(runner.subprocess, "run",
                                  return_value=preflight), \
                mock.patch.object(
                    runner, "_run_webots_with_external_controllers",
                    side_effect=fake_external) as external:
            result = runner.run_single_experiment(
                "A", "Hungarian", 8204, webots_path="webots.exe",
                controller_mode="auto")
        self.assertEqual("webots", result["experiment_info"]["runtime_mode"])
        external.assert_called_once()

    def test_short_webots_run_has_bounded_failure_timeout(self):
        self.assertEqual(140.0, runner.webots_wall_timeout_seconds(10.0))
        self.assertEqual(3720.0, runner.webots_wall_timeout_seconds(1800.0))

    def test_long_batch_can_override_wall_timeout(self):
        with mock.patch.dict(
                os.environ,
                {"SMART_FACTORY_WEBOTS_WALL_TIMEOUT_SECONDS": "7200"}):
            self.assertEqual(
                7200.0, runner.webots_wall_timeout_seconds(1800.0))
        with mock.patch.dict(
                os.environ,
                {"SMART_FACTORY_WEBOTS_WALL_TIMEOUT_SECONDS": "invalid"}):
            self.assertEqual(
                3720.0, runner.webots_wall_timeout_seconds(1800.0))

    def test_experiment_matrix_requires_every_result(self):
        valid = {"C": {"Hungarian": [{"summary_metrics": {}}]}}
        missing = {"C": {"Hungarian": [None]}}
        empty = {"C": {"Hungarian": []}}
        self.assertTrue(runner.experiment_matrix_complete(valid))
        self.assertFalse(runner.experiment_matrix_complete(missing))
        self.assertFalse(runner.experiment_matrix_complete(empty))
        self.assertFalse(runner.experiment_matrix_complete({}))

    def test_parallel_runner_preserves_seed_result_order(self):
        def fake_run(_scenario, scheduler, seed, _webots_path, **_kwargs):
            return {"summary_metrics": {}, "scheduler": scheduler,
                    "seed": seed}

        with mock.patch.object(
                runner, "run_single_experiment", side_effect=fake_run), \
                mock.patch.object(runner, "generate_comparison_report"):
            results = runner.run_all_experiments(
                scenarios=["A"], schedulers=["Hungarian", "Greedy"],
                seeds=[9, 3, 7], max_parallel=3)
        for scheduler in ("Hungarian", "Greedy"):
            self.assertEqual(
                [9, 3, 7],
                [item["seed"] for item in results["A"][scheduler]])
        self.assertTrue(runner.experiment_matrix_complete(results))

    def test_parallel_runner_rejects_invalid_worker_count(self):
        with self.assertRaisesRegex(ValueError, "max_parallel"):
            runner.run_all_experiments(
                scenarios=["A"], schedulers=["Hungarian"], seeds=[1],
                max_parallel=0)

    def test_fallback_debug_log_failure_does_not_stop_simulation(self):
        factory = FactorySupervisor.__new__(FactorySupervisor)
        factory.sim_time = 10.0
        factory.metrics = mock.Mock(output_dir=str(self.temp_dir))
        with mock.patch("builtins.open", side_effect=PermissionError("denied")):
            plans, offsets, partial = factory._build_joint_fallback_plans({})
        self.assertEqual(({}, {}, {}), (plans, offsets, partial))
        self.assertEqual("", factory._fallback_debug_path)


if __name__ == "__main__":
    unittest.main()
