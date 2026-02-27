"""Unit tests for the closed-loop thermal management pipeline."""

import os
import sys
import tempfile
import unittest

import numpy as np

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pipeline.config import PipelineConfig
from pipeline.accelsim_trace_parser import AccelSimTraceParser
from pipeline.hbm_power_model import HBMPowerModel
from pipeline.hotspot_interface import HotSpotInterface
from pipeline.dtm_policies import (
    DTMAction,
    ThrottlingPolicy,
    DVFSPolicy,
    LowPowerModePolicy,
    CompositeDTMPolicy,
)
from pipeline.pipeline_runner import PipelineRunner


# ======================================================================
# Config
# ======================================================================
class TestPipelineConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        self.assertEqual(cfg.NUM_BANKS, 128)
        self.assertEqual(cfg.NUM_CHANNELS, 8)
        self.assertEqual(cfg.NUM_PSEUDO_CHANNELS, 16)
        self.assertEqual(cfg.BANKS_IN_Z, 8)
        self.assertEqual(cfg.banks_per_layer, 16)

    def test_bank_to_channel(self):
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        self.assertEqual(cfg.bank_to_channel(0), 0)
        self.assertEqual(cfg.bank_to_channel(7), 0)
        self.assertEqual(cfg.bank_to_channel(8), 1)
        self.assertEqual(cfg.bank_to_channel(64), 0)  # second half wraps

    def test_channel_to_banks(self):
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        banks = cfg.channel_to_banks(0)
        self.assertEqual(len(banks), 16)
        self.assertTrue(all(0 <= b < 128 for b in banks))

    def test_pseudochannel_to_banks(self):
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        banks = cfg.pseudochannel_to_banks(0)
        self.assertEqual(len(banks), 8)


# ======================================================================
# AccelSim Trace Parser
# ======================================================================
class TestAccelSimTraceParser(unittest.TestCase):

    def setUp(self):
        self.cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        self.parser = AccelSimTraceParser(self.cfg)

    def test_parse_raw_access_file(self):
        path = os.path.join(self.cfg.output_dir, "test_access.txt")
        with open(path, "w") as f:
            f.write(" ".join(["100"] * 128) + "\n")
            f.write(" ".join(["50"] * 128) + "\n")
        reads, writes = self.parser.parse_raw_access_file(path)
        self.assertEqual(len(reads), 128)
        self.assertEqual(reads[0], 100)
        self.assertEqual(writes[0], 50)

    def test_parse_raw_access_file_missing(self):
        reads, writes = self.parser.parse_raw_access_file("/nonexistent")
        self.assertEqual(sum(reads), 0)
        self.assertEqual(sum(writes), 0)

    def test_parse_accelsim_output_bulk(self):
        text = (
            "some header\n"
            "dram_reads_per_bank = " + ",".join(["200"] * 128) + "\n"
            "dram_writes_per_bank = " + ",".join(["100"] * 128) + "\n"
            "gpu_sim_cycle = 50000\n"
        )
        reads, writes, cycles = self.parser.parse_accelsim_output(text)
        self.assertEqual(reads[0], 200)
        self.assertEqual(writes[0], 100)
        self.assertEqual(cycles, 50000)

    def test_parse_accelsim_output_per_bank(self):
        text = (
            "dram[0]: bk[0]: rd = 10\n"
            "dram[0]: bk[0]: wr = 5\n"
            "dram[0]: bk[1]: rd = 20\n"
        )
        reads, writes, _ = self.parser.parse_accelsim_output(text)
        self.assertEqual(reads[0], 10)
        self.assertEqual(writes[0], 5)
        self.assertEqual(reads[1], 20)

    def test_parse_trace_file(self):
        path = os.path.join(self.cfg.output_dir, "sim_out.txt")
        with open(path, "w") as f:
            f.write("dram_reads_per_bank = " + ",".join(["42"] * 128) + "\n")
            f.write("gpu_sim_cycle = 99999\n")
        reads, writes, cycles = self.parser.parse_trace_file(path)
        self.assertEqual(reads[0], 42)
        self.assertEqual(cycles, 99999)


# ======================================================================
# HBM Power Model
# ======================================================================
class TestHBMPowerModel(unittest.TestCase):

    def setUp(self):
        self.cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        self.model = HBMPowerModel(self.cfg)

    def test_idle_power(self):
        reads = [0] * 128
        writes = [0] * 128
        active = [1] * 128
        bank_power, logic_power = self.model.compute_power_trace(
            reads, writes, active
        )
        self.assertEqual(len(bank_power), 128)
        # With zero accesses, power should just be refresh + static
        for p in bank_power:
            self.assertGreaterEqual(p, 0.0)

    def test_active_vs_lpm_power(self):
        reads = [1000] * 128
        writes = [1000] * 128
        active = [1] * 128
        bp_active, _ = self.model.compute_power_trace(reads, writes, active)

        active_lpm = [0] * 128
        bp_lpm, _ = self.model.compute_power_trace(reads, writes, active_lpm)

        # LPM banks should have less power than active banks
        self.assertGreater(bp_active[0], bp_lpm[0])

    def test_dvfs_scaling(self):
        reads = [5000] * 128
        writes = [5000] * 128
        active = [1] * 128

        bp_nom, _ = self.model.compute_power_trace(
            reads, writes, active, freq_scale=1.0, voltage_scale=1.0
        )
        bp_scaled, _ = self.model.compute_power_trace(
            reads, writes, active, freq_scale=0.5, voltage_scale=0.85
        )
        # Scaled power should be lower
        self.assertGreater(bp_nom[0], bp_scaled[0])

    def test_ptrace_header_format(self):
        header = self.model.format_ptrace_header()
        parts = header.split("\t")
        # 16 logic cores + 128 banks
        self.assertEqual(len(parts), 16 + 128)
        self.assertTrue(parts[0].startswith("LC_"))
        self.assertTrue(parts[16].startswith("B_"))

    def test_write_power_trace(self):
        reads = [100] * 128
        writes = [50] * 128
        active = [1] * 128
        bp, lp = self.model.compute_power_trace(reads, writes, active)
        path = os.path.join(self.cfg.output_dir, "test_power.trace")
        self.model.write_power_trace(bp, lp, path)
        with open(path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)  # header + one data line


# ======================================================================
# DTM Policies
# ======================================================================
class TestThrottlingPolicy(unittest.TestCase):

    def setUp(self):
        self.cfg = PipelineConfig(output_dir=tempfile.mkdtemp())

    def test_no_throttle_below_threshold(self):
        policy = ThrottlingPolicy(self.cfg, throttle_temp=80, critical_temp=85)
        temps = np.full(128, 70.0)
        action = policy.evaluate(temps)
        self.assertTrue(all(f == 1.0 for f in action.throttle_factors))

    def test_full_throttle_above_critical(self):
        policy = ThrottlingPolicy(self.cfg, throttle_temp=80, critical_temp=85)
        temps = np.full(128, 90.0)
        action = policy.evaluate(temps)
        self.assertTrue(all(f == 0.0 for f in action.throttle_factors))

    def test_partial_throttle(self):
        policy = ThrottlingPolicy(self.cfg, throttle_temp=80, critical_temp=85)
        temps = np.full(128, 70.0)
        temps[0] = 82.5  # midway between 80 and 85 → 50% throttle
        action = policy.evaluate(temps)
        self.assertAlmostEqual(action.throttle_factors[0], 0.5, places=2)
        self.assertEqual(action.throttle_factors[1], 1.0)


class TestDVFSPolicy(unittest.TestCase):

    def setUp(self):
        self.cfg = PipelineConfig(output_dir=tempfile.mkdtemp())

    def test_cool_no_scaling(self):
        policy = DVFSPolicy(self.cfg)
        temps = np.full(128, 60.0)
        action = policy.evaluate(temps)
        self.assertEqual(action.freq_scale, 1.0)
        self.assertEqual(action.voltage_scale, 1.0)

    def test_hot_triggers_scaling(self):
        policy = DVFSPolicy(self.cfg)
        temps = np.full(128, 83.0)  # above highest threshold (82)
        action = policy.evaluate(temps)
        self.assertLess(action.freq_scale, 1.0)
        self.assertLess(action.voltage_scale, 1.0)


class TestLowPowerModePolicy(unittest.TestCase):

    def setUp(self):
        self.cfg = PipelineConfig(output_dir=tempfile.mkdtemp())

    def test_lpm_triggered(self):
        policy = LowPowerModePolicy(self.cfg, critical_temp=85, cooldown_temp=75)
        temps = np.full(128, 90.0)
        action = policy.evaluate(temps)
        self.assertEqual(len(action.lpm_channels), 8)  # all channels
        self.assertTrue(all(not a for a in action.bank_active))

    def test_cool_no_lpm(self):
        policy = LowPowerModePolicy(self.cfg, critical_temp=85, cooldown_temp=75)
        temps = np.full(128, 60.0)
        action = policy.evaluate(temps)
        self.assertEqual(len(action.lpm_channels), 0)

    def test_cooldown_restores(self):
        policy = LowPowerModePolicy(self.cfg, critical_temp=85, cooldown_temp=75)
        # First: trigger LPM
        temps_hot = np.full(128, 90.0)
        action = policy.evaluate(temps_hot)
        self.assertGreater(len(action.lpm_channels), 0)
        # Second: temperatures drop below cooldown
        temps_cool = np.full(128, 70.0)
        action = policy.evaluate(temps_cool, current_action=action)
        self.assertEqual(len(action.lpm_channels), 0)


class TestCompositeDTMPolicy(unittest.TestCase):

    def setUp(self):
        self.cfg = PipelineConfig(output_dir=tempfile.mkdtemp())

    def test_composite_chains_policies(self):
        policy = CompositeDTMPolicy(self.cfg, [
            ThrottlingPolicy(self.cfg),
            DVFSPolicy(self.cfg),
            LowPowerModePolicy(self.cfg),
        ])
        temps = np.full(128, 90.0)
        action = policy.evaluate(temps)
        # Throttle should be applied
        self.assertTrue(all(f == 0.0 for f in action.throttle_factors))
        # DVFS should be active
        self.assertLess(action.freq_scale, 1.0)
        # LPM should be active
        self.assertGreater(len(action.lpm_channels), 0)


# ======================================================================
# Pipeline Runner (integration, no real HotSpot binary)
# ======================================================================
class TestPipelineRunner(unittest.TestCase):

    def test_run_with_static_trace(self):
        outdir = tempfile.mkdtemp()
        cfg = PipelineConfig(output_dir=outdir)
        runner = PipelineRunner(cfg)

        # Create a simple trace file
        trace_path = os.path.join(outdir, "trace.txt")
        with open(trace_path, "w") as f:
            f.write(" ".join(["500"] * 128) + "\n")
            f.write(" ".join(["300"] * 128) + "\n")

        # Run for 3 iterations (HotSpot binary won't be present, so
        # temperatures won't be computed, but the power-trace files should
        # still be written)
        history = runner.run(iterations=3, trace_source=trace_path)
        self.assertEqual(len(history["bank_power"]), 3)
        self.assertTrue(os.path.isfile(cfg.full_power_trace_file))

    def test_run_with_per_interval_dir(self):
        outdir = tempfile.mkdtemp()
        cfg = PipelineConfig(output_dir=outdir)
        runner = PipelineRunner(cfg)

        trace_dir = os.path.join(outdir, "intervals")
        os.makedirs(trace_dir)
        for i in range(5):
            with open(os.path.join(trace_dir, "interval_%d.txt" % i), "w") as f:
                val = 100 * (i + 1)
                f.write(" ".join([str(val)] * 128) + "\n")
                f.write(" ".join([str(val // 2)] * 128) + "\n")

        history = runner.run(iterations=5, per_interval_trace_dir=trace_dir)
        self.assertEqual(len(history["bank_power"]), 5)

    def test_idle_run(self):
        """With no trace source the pipeline runs with zero accesses."""
        outdir = tempfile.mkdtemp()
        cfg = PipelineConfig(output_dir=outdir)
        runner = PipelineRunner(cfg)
        history = runner.run(iterations=2)
        self.assertEqual(len(history["bank_power"]), 2)
        # All power should be just refresh (very small)
        for p in history["bank_power"][0]:
            self.assertGreaterEqual(p, 0.0)


# ======================================================================
# HotSpot Interface (unit-level, no binary)
# ======================================================================
class TestHotSpotInterface(unittest.TestCase):

    def test_setup_creates_output_files(self):
        outdir = tempfile.mkdtemp()
        cfg = PipelineConfig(output_dir=outdir)
        hs = HotSpotInterface(cfg)
        hs.setup()
        self.assertTrue(os.path.isfile(cfg.full_temperature_trace_file))
        self.assertTrue(os.path.isfile(cfg.full_power_trace_file))
        self.assertTrue(os.path.isfile(cfg.full_totalpower_trace_file))

    def test_read_temps_returns_zeros_when_no_file(self):
        outdir = tempfile.mkdtemp()
        cfg = PipelineConfig(output_dir=outdir)
        hs = HotSpotInterface(cfg)
        temps = hs.get_bank_temperatures()
        self.assertEqual(len(temps), 128)
        self.assertEqual(sum(temps), 0.0)


if __name__ == "__main__":
    unittest.main()
