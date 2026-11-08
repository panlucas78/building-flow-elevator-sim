"""Regression tests for physical timing, fairness, capacity and request lifecycle."""
import io
import json
import math
import unittest
import zipfile
from dataclasses import asdict, replace

from elevator_sim import (
    POLICIES, DemandPredictor, Request, SimulationConfig, Simulator, benchmark,
    csv_bytes, export_experiment, generate_requests, requests_from_csv, run_experiment,
)


def config(**changes):
    base = SimulationConfig(floors=8, elevators=1, duration=120, peak_center=40,
                            travel_seconds=1, door_open_seconds=0, door_close_seconds=0,
                            boarding_seconds=0, alighting_seconds=0, sample_interval=1)
    return replace(base, **changes)


class SimulationTests(unittest.TestCase):
    def test_hand_calculated_single_passenger(self):
        cfg = config(floors=5, duration=40, travel_seconds=3, door_open_seconds=2,
                     door_close_seconds=2, boarding_seconds=0.6, alighting_seconds=0.4)
        result = Simulator(cfg, [Request(0, 0, 3, 5)], "nearest").run()
        passenger = result.passengers[0]
        self.assertAlmostEqual(passenger["boarded_at"], 8.6)
        self.assertAlmostEqual(passenger["delivered_at"], 19.0)
        self.assertAlmostEqual(result.metrics["total_distance"], 4.0)
        self.assertAlmostEqual(result.metrics["empty_distance"], 2.0)
        self.assertEqual(result.metrics["door_cycles"], 2)

    def test_call_completes_once_without_reopening_forever(self):
        result = Simulator(config(), [Request(0, 0, 2, 5)], "nearest").run()
        self.assertEqual(result.metrics["delivered"], 1)
        self.assertEqual(result.metrics["waiting"], 0)
        self.assertEqual(result.metrics["door_cycles"], 2)
        self.assertEqual(sum(e["event"] == "分配" for e in result.events), 1)
        self.assertEqual(sum(e["event"] == "上车" for e in result.events), 1)
        self.assertEqual(sum(e["event"] == "送达" for e in result.events), 1)

    def test_car_between_targets_keeps_moving(self):
        requests = [Request(0, 0, 1, 8), Request(1, 4, 2, 1), Request(2, 4, 7, 8)]
        result = Simulator(config(), requests, "nearest").run()
        self.assertEqual(result.metrics["delivered"], 3)
        self.assertEqual(result.metrics["waiting"], 0)
        self.assertGreater(result.metrics["total_distance"], 7)

    def test_capacity_and_same_floor_multiple_passengers(self):
        requests = [Request(index, 0, 1, 3) for index in range(6)]
        result = Simulator(config(capacity=1), requests, "eta").run()
        self.assertEqual(result.metrics["delivered"], 6)
        self.assertTrue(all(car["load"] <= 1 for snap in result.snapshots for car in snap["elevators"]))
        self.assertEqual(len([event for event in result.events if event["event"] == "上车"]), 6)

    def test_opposite_hall_directions_both_get_service(self):
        result = Simulator(config(), [Request(0, 0, 3, 7), Request(1, 0, 3, 1)], "eta").run()
        self.assertEqual(result.metrics["delivered"], 2)
        self.assertGreaterEqual(result.passengers[1]["boarded_at"], result.passengers[0]["delivered_at"])

    def test_unserved_people_remain_in_metrics(self):
        requests = [Request(0, 9, 8, 1), Request(1, 9, 7, 1)]
        result = Simulator(config(duration=10, peak_center=5, travel_seconds=3), requests, "adaptive").run()
        self.assertEqual(result.metrics["total"], 2)
        self.assertEqual(result.metrics["waiting"], 2)
        self.assertEqual(result.metrics["completion_rate"], 0)
        self.assertIsNone(result.metrics["avg_wait"])
        self.assertEqual(result.metrics["observed_wait"], 1)
        self.assertAlmostEqual(result.metrics["avg_queue"], 0.2)

    def test_partial_empty_movement_at_horizon_is_counted(self):
        result = Simulator(config(duration=2, peak_center=1, travel_seconds=3),
                           [Request(0, 0, 8, 1)], "nearest").run()
        self.assertAlmostEqual(result.metrics["empty_distance"], 2 / 3)
        self.assertAlmostEqual(result.metrics["total_distance"], 2 / 3)

    def test_empty_request_stream(self):
        experiment = run_experiment(config(), requests=[])
        for result in experiment.results.values():
            self.assertEqual(result.metrics["total"], 0)
            self.assertEqual(result.metrics["total_distance"], 0)
            self.assertIsNone(result.metrics["avg_wait"])
        json.dumps(experiment.summary(), allow_nan=False)

    def test_requests_are_seeded_and_legal_for_every_scenario(self):
        for scenario in ("up_peak", "down_peak", "mixed", "class_change"):
            cfg = config(elevators=3, scenario=scenario, base_rate=0.5)
            first, second = generate_requests(cfg), generate_requests(cfg)
            self.assertEqual(first, second)
            self.assertGreater(len(first), 0)
            self.assertTrue(all(req.origin != req.destination and 0 <= req.arrival < cfg.duration for req in first))
            self.assertEqual(len(first), len({req.id for req in first}))
        self.assertTrue(any(req.origin == 1 for req in generate_requests(config(base_rate=1))))

    def test_all_policies_share_requests_and_conserve_people(self):
        cfg = config(elevators=3, capacity=2, base_rate=1.0, scenario="mixed")
        experiment = run_experiment(cfg, POLICIES)
        for result in experiment.results.values():
            metrics = result.metrics
            self.assertEqual(metrics["total"], len(experiment.requests))
            self.assertEqual(metrics["total"], metrics["waiting"] + metrics["riding"] + metrics["delivered"])
            self.assertEqual([row["id"] for row in result.passengers], [req.id for req in experiment.requests])
            assignments = [event["passenger"] for event in result.events if event["event"] == "分配"]
            self.assertEqual(len(assignments), len(set(assignments)))
            for snap in result.snapshots:
                for car in snap["elevators"]:
                    self.assertLessEqual(car["load"], cfg.capacity)
                    self.assertTrue(1 <= car["position"] <= cfg.floors)
            for row in result.passengers:
                if row["boarded_at"] is not None:
                    self.assertGreaterEqual(row["boarded_at"], row["arrival"])
                if row["delivered_at"] is not None:
                    self.assertGreaterEqual(row["delivered_at"], row["boarded_at"])

    def test_deterministic_replay_and_monitoring_has_no_effect(self):
        cfg = config(elevators=2, scenario="mixed", base_rate=0.4)
        first, second = run_experiment(cfg), run_experiment(cfg)
        self.assertEqual(first.fingerprint, second.fingerprint)
        for policy in first.results:
            self.assertEqual(first.results[policy].metrics, second.results[policy].metrics)
            self.assertEqual(first.results[policy].snapshots, second.results[policy].snapshots)
        silent = run_experiment(cfg, record=False)
        self.assertEqual(first.summary(), silent.summary())

    def test_future_arrivals_cannot_change_earlier_behavior(self):
        cfg = config(elevators=2)
        prefix = [Request(0, 0, 4, 1), Request(1, 1, 4, 1), Request(2, 2, 4, 1)]
        a = Simulator(cfg, prefix + [Request(3, 90, 2, 8)], "adaptive").run()
        b = Simulator(cfg, prefix + [Request(3, 90, 7, 1)], "adaptive").run()
        self.assertEqual([snap for snap in a.snapshots if snap["time"] < 90],
                         [snap for snap in b.snapshots if snap["time"] < 90])

    def test_predictor_is_causal_and_finite(self):
        predictor = DemandPredictor(config())
        self.assertEqual(sum(predictor.forecast(0)), 0)
        predictor.observe(Request(0, 10, 3, 1))
        predictor.observe(Request(1, 100, 7, 1))
        forecast = predictor.forecast(20)
        self.assertGreater(forecast[3], 0)
        self.assertEqual(forecast[7], 0)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in forecast))

    def test_csv_roundtrip_and_validation(self):
        cfg = config()
        requests = (Request(0, 0.5, 1, 8), Request(1, 2.5, 8, 1))
        data = csv_bytes([asdict(req) for req in requests])
        self.assertEqual(requests_from_csv(data, cfg), requests)
        for invalid in (b"wrong,columns\n1,2\n", b"id,arrival,origin,destination\n0,nan,1,2\n",
                        b"id,arrival,origin,destination\n0,1,1,1\n", b"id,arrival,origin,destination\n0,1,1,2\n0,2,1,3\n"):
            with self.assertRaises(ValueError):
                requests_from_csv(invalid, cfg)

    def test_invalid_configuration_and_horizon(self):
        for changes in ({"floors": 1}, {"capacity": 0}, {"travel_seconds": 0},
                        {"base_rate": float("nan")}, {"peak_center": 999}, {"scenario": "unknown"},
                        {"seed": -1}, {"seed": True}, {"seed": 1.5}):
            with self.assertRaises(ValueError):
                config(**changes)
        with self.assertRaises(ValueError):
            Simulator(config(), [Request(0, 120, 1, 2)], "eta")

    def test_csv_rejects_silent_data_loss_and_parser_errors(self):
        cfg = config()
        malformed = (
            b"id,arrival,origin,destination,destination\n0,0,1,8,2\n",
            b"id,arrival,origin,destination\n0,0,1,2,8\n",
            b"id,arrival,origin,destination,\n0,0,1,2,extra\n",
            b"id,arrival,origin,destination\n0,0,1\n",
            b'id,arrival,origin,destination\n0,0,1,"2\n',
            b"id,arrival,origin,destination\n0,0,1," + b"9" * 140000,
            b" " * (5 * 1024 * 1024 + 1),
        )
        for data in malformed:
            with self.subTest(length=len(data), prefix=data[:70]):
                with self.assertRaisesRegex(ValueError, "CSV"):
                    requests_from_csv(data, cfg)
        # 合法的附加列与带空白的表头仍可导入，不影响实际起终点。
        data = b" id ,arrival,origin,destination,note\n0,0,1,8,metadata\n"
        self.assertEqual(requests_from_csv(data, cfg), (Request(0, 0.0, 1, 8),))

    def test_forecast_records_configured_horizon(self):
        requests = [Request(i, float(i + 1), 2, 1) for i in range(3)]
        short = run_experiment(config(forecast_horizon=60), ("eta",), requests)
        long = run_experiment(config(forecast_horizon=120), ("eta",), requests)
        a = next(row for row in short.results["eta"].forecasts if row["time"] >= 20)
        b = next(row for row in long.results["eta"].forecasts if row["time"] >= 20)
        self.assertEqual(b["horizon_seconds"], 120)
        self.assertGreater(a["predicted_next_horizon"], 0)
        self.assertAlmostEqual(b["predicted_next_horizon"], 2 * a["predicted_next_horizon"])
        self.assertEqual(a["predicted_next_60"], a["predicted_next_horizon"])
        self.assertNotIn("predicted_next_60", b)
        self.assertIn("未来120秒", long.summary()["methodology"]["forecast"])

    def test_benchmark_requires_distinct_seeds_and_reports_progress(self):
        cfg = config(base_rate=0.2)
        for seeds in ([42, 42], [True, 42], [-1, 42], [1.5, 42]):
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    benchmark(cfg, seeds)
        progress = []
        report = benchmark(cfg, [41, 42], on_progress=lambda done, total, seed: progress.append((done, total, seed)))
        self.assertEqual(progress, [(1, 2, 41), (2, 2, 42)])
        self.assertEqual(report, benchmark(cfg, [41, 42]))

    def test_export_and_paired_benchmark(self):
        cfg = config(base_rate=0.2)
        experiment = run_experiment(cfg)
        with zipfile.ZipFile(io.BytesIO(export_experiment(experiment))) as archive:
            self.assertIsNone(archive.testzip())
            metadata = json.loads(archive.read("experiment.json"))
            self.assertEqual(metadata["request_sha256"], experiment.fingerprint)
            restored = requests_from_csv(archive.read("requests.csv"), cfg)
            self.assertEqual(restored, experiment.requests)
        report = benchmark(cfg, [41, 42, 43])
        self.assertEqual(len(report["runs"]), 3)
        self.assertEqual(report["paired_wait_comparison"]["n"], 3)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
