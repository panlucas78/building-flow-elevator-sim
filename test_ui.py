"""Functional Streamlit checks, including settings, CSV ingestion and empty traffic."""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent


def by_label(elements, label):
    return next(element for element in elements if element.label == label)


class InterfaceTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_file(str(ROOT / "eos.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_settings_recompute_shared_input_and_replay_step(self):
        app = self.make_app()
        original = app.session_state["experiment"].fingerprint
        by_label(app.number_input, "随机种子").set_value(43)
        by_label(app.button, "运行公平对比").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertNotEqual(original, app.session_state["experiment"].fingerprint)
        by_label(app.button, "下一步").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(app.session_state["replay_time"], 0)
        by_label(app.button, "播放").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state["replay_playing"])
        by_label(app.button, "暂停").click().run(timeout=30)
        self.assertFalse(app.session_state["replay_playing"])
        by_label(app.button, "回到起点").click().run(timeout=30)
        self.assertEqual(app.session_state["replay_time"], 0)
        by_label(app.select_slider, "仿真时间（秒）").set_value(3600.0).run(timeout=30)
        by_label(app.button, "播放").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(app.session_state["replay_playing"])

    def test_zero_demand_renders_without_division_errors(self):
        app = self.make_app()
        by_label(app.slider, "基础到达率（人 / 秒）").set_value(0.0)
        by_label(app.slider, "高峰额外到达率（人 / 秒）").set_value(0.0)
        by_label(app.button, "运行公平对比").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["experiment"].requests), 0)
        self.assertEqual(app.session_state["experiment"].results["adaptive"].metrics["total_distance"], 0)

    def test_csv_is_used_instead_of_generated_traffic(self):
        # File bytes enter through the same UI ingestion boundary without a GUI upload.
        script = "\n".join([
            "from pathlib import Path",
            "from types import SimpleNamespace",
            "from unittest.mock import patch",
            "from eos import main",
            "data = Path('sample_requests.csv').read_bytes()",
            "with patch('streamlit.file_uploader', return_value=SimpleNamespace(getvalue=lambda: data)):",
            "    main()",
        ])
        app = AppTest.from_string(script).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        experiment = app.session_state["experiment"]
        self.assertEqual(experiment.input_source, "导入请求")
        self.assertEqual(len(experiment.requests), 12)
        self.assertTrue(all(result.metrics["total"] == 12 for result in experiment.results.values()))

    def test_oversized_csv_field_has_friendly_error(self):
        script = "\n".join([
            "from types import SimpleNamespace",
            "from unittest.mock import patch",
            "from eos import main",
            "data = b'id,arrival,origin,destination\\n1,0,1,' + b'2' * 140000 + b'\\n'",
            "with patch('streamlit.file_uploader', return_value=SimpleNamespace(getvalue=lambda: data)):",
            "    main()",
        ])
        app = AppTest.from_string(script).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("CSV" in error.value for error in app.error))

    def test_zip_is_reused_until_experiment_changes(self):
        script = "\n".join([
            "import streamlit as st",
            "from unittest.mock import patch",
            "from elevator_sim import export_experiment as original_export",
            "from eos import main",
            "def tracked_export(experiment):",
            "    st.session_state['zip_builds'] = st.session_state.get('zip_builds', 0) + 1",
            "    return original_export(experiment)",
            "with patch('eos.export_experiment', side_effect=tracked_export):",
            "    main()",
        ])
        app = AppTest.from_string(script).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        fingerprint = app.session_state["experiment"].fingerprint
        original_zip = app.session_state["experiment_zip"]
        # 模拟升级前会话中的旧预测记录，新界面仍须能读取而不崩溃。
        for result in app.session_state["experiment"].results.values():
            for row in result.forecasts:
                row.pop("predicted_next_horizon", None)
                row.pop("horizon_seconds", None)
        by_label(app.radio, "乘客状态").set_value("已送达").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["zip_builds"], 1)
        self.assertEqual(app.session_state["experiment_zip"], original_zip)
        # 客流未变但行驶参数变化，必须重新导出；仅用请求指纹缓存会漏掉这一情况。
        by_label(app.slider, "每跨一层行驶时间（秒）").set_value(4.0)
        by_label(app.button, "运行公平对比").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["experiment"].fingerprint, fingerprint)
        self.assertEqual(app.session_state["zip_builds"], 2)
        self.assertNotEqual(app.session_state["experiment_zip"], original_zip)

    def test_invalid_cli_duration_is_an_argument_error(self):
        process = subprocess.run(
            [sys.executable, str(ROOT / "eos.py"), "--batch", "--duration", "0", "--seeds", "1"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=15,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("duration", process.stderr)
        self.assertNotIn("Traceback", process.stderr)

    def test_batch_form_and_empty_policy_feedback(self):
        app = self.make_app()
        by_label(app.select_slider, "仿真时长（分钟）").set_value(10)
        by_label(app.button, "运行公平对比").click().run(timeout=30)
        by_label(app.slider, "重复次数").set_value(2)
        by_label(app.button, "运行批量验证").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["batch_result"]["runs"]), 2)
        self.assertEqual(len(app.session_state["batch_result"]["summary"]), 5)
        previous = app.session_state["batch_result"]
        with patch("elevator_sim.benchmark", side_effect=ValueError("测试数据无效")):
            by_label(app.button, "运行批量验证").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["batch_result"], previous)
        self.assertTrue(any("上一次成功" in error.value for error in app.error))
        by_label(app.multiselect, "对比策略").set_value([])
        by_label(app.button, "运行公平对比").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("至少选择" in error.value for error in app.error))


if __name__ == "__main__":
    unittest.main(verbosity=2)
