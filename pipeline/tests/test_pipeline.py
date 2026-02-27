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


# ======================================================================
# Real config file integration tests
# ======================================================================
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CONFIG_DIR = os.path.join(REPO_ROOT, "config", "hotspot", "3Dmem_16core")


@unittest.skipUnless(
    os.path.isdir(CONFIG_DIR),
    "HBM config directory not present",
)
class TestRealConfigFiles(unittest.TestCase):
    """Validate pipeline integration with the real HBM config files."""

    def test_config_files_exist(self):
        """All expected config files should be present."""
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        self.assertTrue(os.path.isfile(cfg.hotspot_config_file),
                        "mem_hotspot.config missing")
        self.assertTrue(os.path.isfile(cfg.hotspot_layer_file),
                        "mem.lcf missing")
        self.assertTrue(os.path.isfile(cfg.init_file_external),
                        "mem.init missing")

    def test_floorplan_files_exist(self):
        """All .flp files referenced by mem.lcf should exist."""
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        for name in [
            "mem_ctrl.flp", "mem_tim.flp",
            "mem_bank_1.flp", "mem_bank_2.flp", "mem_bank_3.flp",
            "mem_bank_4.flp", "mem_bank_5.flp", "mem_bank_6.flp",
            "mem_bank_7.flp", "mem_bank_8.flp",
        ]:
            path = os.path.join(cfg.hotspot_floorplan_folder, name)
            self.assertTrue(os.path.isfile(path), "%s missing" % name)

    def test_lcf_has_18_layers(self):
        """The LCF file should define exactly 18 layers (0..17)."""
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        layer_numbers = []
        with open(cfg.hotspot_layer_file) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.isdigit():
                    layer_numbers.append(int(stripped))
        self.assertEqual(len(layer_numbers), cfg.NUM_HOTSPOT_LAYERS)
        self.assertEqual(layer_numbers, list(range(18)))

    def test_lcf_no_hardcoded_absolute_paths(self):
        """The LCF should not contain hardcoded absolute paths."""
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        with open(cfg.hotspot_layer_file) as fh:
            for lineno, line in enumerate(fh, 1):
                if line.strip().endswith(".flp"):
                    self.assertFalse(
                        os.path.isabs(line.strip()),
                        "Line %d has absolute path: %s" % (lineno, line.strip()),
                    )

    def test_prepare_lcf_creates_absolute_paths(self):
        """prepare_lcf() should resolve relative paths to absolute."""
        outdir = tempfile.mkdtemp()
        cfg = PipelineConfig(output_dir=outdir)
        cfg.prepare_lcf()
        self.assertTrue(os.path.isfile(cfg.runtime_layer_file))

        with open(cfg.runtime_layer_file) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.endswith(".flp"):
                    self.assertTrue(
                        os.path.isabs(stripped),
                        "Runtime LCF should have absolute path: %s" % stripped,
                    )
                    self.assertTrue(
                        os.path.isfile(stripped),
                        "Resolved floorplan not found: %s" % stripped,
                    )

    def test_bank_floorplan_has_16_banks(self):
        """Each mem_bank_*.flp should define 16 banks (B_0..B_15)."""
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        flp_path = os.path.join(cfg.hotspot_floorplan_folder, "mem_bank_1.flp")
        bank_names = []
        with open(flp_path) as fh:
            for line in fh:
                if line.startswith("B_"):
                    bank_names.append(line.split()[0])
        self.assertEqual(len(bank_names), cfg.banks_per_layer)

    def test_mem_ctrl_has_16_logic_cores(self):
        """mem_ctrl.flp should define 16 logic cores (LC_0..LC_15)."""
        cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        flp_path = os.path.join(cfg.hotspot_floorplan_folder, "mem_ctrl.flp")
        lc_names = []
        with open(flp_path) as fh:
            for line in fh:
                if line.startswith("LC_"):
                    lc_names.append(line.split()[0])
        self.assertEqual(len(lc_names), cfg.NUM_LOGIC_CORES)

    def test_setup_copies_init_file(self):
        """HotSpotInterface.setup() should copy mem.init to output dir."""
        outdir = tempfile.mkdtemp()
        cfg = PipelineConfig(output_dir=outdir)
        hs = HotSpotInterface(cfg)
        hs.setup()
        self.assertTrue(os.path.isfile(cfg.init_file))
        # init file should be non-empty
        self.assertGreater(os.path.getsize(cfg.init_file), 0)

    def test_num_hotspot_layers_constant(self):
        """NUM_HOTSPOT_LAYERS should be 18."""
        self.assertEqual(PipelineConfig.NUM_HOTSPOT_LAYERS, 18)


# ======================================================================
# Realistic Power Model
# ======================================================================
class TestRealisticPowerModel(unittest.TestCase):
    """Tests for the temperature-dependent leakage power model."""

    def setUp(self):
        self.cfg = PipelineConfig(output_dir=tempfile.mkdtemp())
        self.model = HBMPowerModel(self.cfg)

    def test_idle_power_is_nonzero(self):
        """Even with zero accesses, static+refresh power should be > 0."""
        reads = [0] * 128
        writes = [0] * 128
        active = [1] * 128
        bp, lp = self.model.compute_power_trace(reads, writes, active)
        self.assertGreater(bp[0], 0.0, "Idle bank power should be > 0")
        self.assertGreater(lp[0], 0.0, "Logic core power should be > 0")

    def test_leakage_increases_with_temperature(self):
        """Hotter banks should have higher leakage power."""
        reads = [0] * 128
        writes = [0] * 128
        active = [1] * 128

        self.model.set_bank_temperatures([45.0] * 128)
        bp_cool, _ = self.model.compute_power_trace(reads, writes, active)

        self.model.set_bank_temperatures([85.0] * 128)
        bp_hot, _ = self.model.compute_power_trace(reads, writes, active)

        self.assertGreater(bp_hot[0], bp_cool[0],
                           "Power at 85°C should exceed power at 45°C")

    def test_leakage_doubles_per_12c(self):
        """Leakage should roughly double every ~12°C."""
        import math
        cfg = self.cfg
        p45 = cfg.BANK_STATIC_POWER_W_REF * math.exp(cfg.LEAKAGE_TEMP_COEFF * 0)
        p57 = cfg.BANK_STATIC_POWER_W_REF * math.exp(cfg.LEAKAGE_TEMP_COEFF * 12)
        ratio = p57 / p45
        self.assertAlmostEqual(ratio, 2.054, places=1)

    def test_lpm_power_is_fraction_of_active(self):
        """LPM power should be LOW_POWER_LEAKAGE_FRACTION of leakage."""
        reads = [1000] * 128
        writes = [1000] * 128
        active_on = [1] * 128
        active_off = [0] * 128

        bp_on, _ = self.model.compute_power_trace(reads, writes, active_on)
        bp_off, _ = self.model.compute_power_trace(reads, writes, active_off)

        # LPM should be small fraction of leakage, definitely < active power
        self.assertLess(bp_off[0], bp_on[0] * 0.2)

    def test_logic_core_power_nonzero(self):
        """Logic cores should have static + dynamic power."""
        reads = [0] * 128
        writes = [0] * 128
        active = [1] * 128
        _, lp = self.model.compute_power_trace(reads, writes, active)
        self.assertEqual(len(lp), 16)
        expected = self.cfg.LOGIC_CORE_STATIC_POWER_W + self.cfg.LOGIC_CORE_DYNAMIC_POWER_W
        self.assertAlmostEqual(lp[0], expected, places=4)

    def test_activation_precharge_energy(self):
        """Dynamic power should include act+pre energy beyond just read/write."""
        reads = [10000] * 128
        writes = [0] * 128
        active = [1] * 128

        bp, _ = self.model.compute_power_trace(reads, writes, active)
        # Power with activation+precharge should be higher than just read energy
        # Read-only power: 10000 * 20.55 nJ / 1e6 ns = 0.2055 W
        # Act+Pre adds: 10000 * (3.2+1.1) nJ / 1e6 ns = 0.043 W
        # Plus leakage (~20mW) and refresh
        self.assertGreater(bp[0], 0.24)


# ======================================================================
# Benchmarks
# ======================================================================
class TestBenchmarks(unittest.TestCase):

    def setUp(self):
        self.cfg = PipelineConfig(output_dir=tempfile.mkdtemp())

    def test_list_benchmarks(self):
        from pipeline.benchmarks import list_benchmarks
        names = list_benchmarks()
        self.assertIn("stream", names)
        self.assertIn("sgemm", names)
        self.assertIn("resnet50", names)
        self.assertIn("random", names)
        self.assertIn("hotspot_stress", names)

    def test_stream_uniform(self):
        from pipeline.benchmarks import get_benchmark
        fn = get_benchmark("stream")
        reads, writes = fn(self.cfg, step=0)
        self.assertEqual(len(reads), 128)
        self.assertEqual(len(writes), 128)
        # STREAM is uniform — all banks should have same counts
        self.assertEqual(len(set(reads)), 1)
        self.assertEqual(len(set(writes)), 1)
        # Read:write ≈ 2:1
        self.assertGreater(reads[0], writes[0])

    def test_sgemm_non_uniform(self):
        from pipeline.benchmarks import get_benchmark
        fn = get_benchmark("sgemm")
        reads, writes = fn(self.cfg, step=0)
        # SGEMM should have non-uniform access
        self.assertGreater(len(set(reads)), 1)

    def test_resnet50_burst_idle(self):
        from pipeline.benchmarks import get_benchmark
        fn = get_benchmark("resnet50")
        reads_conv, _ = fn(self.cfg, step=0)  # convolution phase
        reads_bn, _ = fn(self.cfg, step=3)    # batch-norm phase
        # Convolution should have much higher traffic than batch-norm
        self.assertGreater(reads_conv[0], reads_bn[0] * 2)

    def test_benchmark_counts_are_realistic(self):
        """Access counts should be in a realistic DRAM range."""
        from pipeline.benchmarks import get_benchmark
        fn = get_benchmark("stream")
        reads, writes = fn(self.cfg, step=0)
        # STREAM at 70% of 31,250 peak → ~21,875 total per bank
        total = reads[0] + writes[0]
        self.assertGreater(total, 10_000)
        self.assertLess(total, 50_000)

    def test_benchmark_run_via_pipeline(self):
        """Pipeline runner should accept benchmark parameter."""
        from pipeline.pipeline_runner import PipelineRunner
        outdir = tempfile.mkdtemp()
        cfg = PipelineConfig(output_dir=outdir)
        runner = PipelineRunner(cfg)
        history = runner.run(iterations=2, benchmark="stream")
        self.assertEqual(len(history["bank_power"]), 2)
        # Power should be non-trivial (not all zeros)
        self.assertGreater(sum(history["bank_power"][0]), 0)


if __name__ == "__main__":
    unittest.main()
