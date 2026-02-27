"""
Closed-loop pipeline runner.

Orchestrates the four stages of the thermal management loop:

  1. **Accel-Sim** — obtain HBM access traces (from file or live run)
  2. **Power model** — convert access traces → power trace
  3. **HotSpot** — compute temperatures from power trace
  4. **DTM policy** — evaluate temperatures and produce control actions

The loop repeats for a configurable number of iterations or until a
convergence / termination criterion is met.
"""

import logging
import os
import subprocess
import time

import numpy as np

from .config import PipelineConfig
from .accelsim_trace_parser import AccelSimTraceParser
from .hbm_power_model import HBMPowerModel
from .hotspot_interface import HotSpotInterface
from .dtm_policies import DTMAction, CompositeDTMPolicy

log = logging.getLogger(__name__)


class PipelineRunner:
    """
    Top-level orchestrator for the closed-loop pipeline.

    Usage::

        cfg = PipelineConfig(...)
        runner = PipelineRunner(cfg)
        runner.set_dtm_policy(my_policy)
        runner.run(iterations=200, trace_file="mem_trace.txt")
    """

    def __init__(self, config=None):
        self.cfg = config or PipelineConfig()
        self.parser = AccelSimTraceParser(self.cfg)
        self.power_model = HBMPowerModel(self.cfg)
        self.hotspot = HotSpotInterface(self.cfg)
        self.dtm_policy = CompositeDTMPolicy(self.cfg)

        # State carried across iterations
        self.bank_active = [1] * self.cfg.NUM_BANKS
        self.freq_scale = 1.0
        self.voltage_scale = 1.0
        self.throttle_factors = [1.0] * self.cfg.NUM_BANKS

        # History for analysis
        self.history = {
            "bank_temps": [],
            "bank_power": [],
            "freq_scale": [],
            "voltage_scale": [],
            "lpm_channels": [],
            "throttle_factors": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_dtm_policy(self, policy):
        """Replace the current DTM policy (may be a ``CompositeDTMPolicy``)."""
        self.dtm_policy = policy

    def run(
        self,
        iterations=100,
        trace_source=None,
        accelsim_cmd=None,
        per_interval_trace_dir=None,
        benchmark=None,
    ):
        """
        Execute the closed-loop pipeline.

        Parameters
        ----------
        iterations : int
            Number of sampling intervals to simulate.
        trace_source : str | None
            Path to a file with per-bank access counts (read then write).
            If *None*, falls back to ``accelsim_cmd`` or synthetic zeros.
        accelsim_cmd : list[str] | None
            Shell command to run Accel-Sim for one interval.  Its stdout
            is parsed for per-bank DRAM access stats.
        per_interval_trace_dir : str | None
            Directory containing per-interval trace files named
            ``interval_<N>.txt`` (read / write counts per bank).
        benchmark : str | callable | None
            Either a registered benchmark name (e.g. ``"stream"``) or a
            callable ``fn(cfg, step) -> (reads, writes)``.
        """
        log.info("Pipeline starting for %d iterations", iterations)
        self.hotspot.setup()

        for step in range(iterations):
            t0 = time.time()

            # ---- Stage 1: obtain access trace ----------------------------
            reads, writes = self._get_access_trace(
                step, trace_source, accelsim_cmd, per_interval_trace_dir,
                benchmark,
            )

            # Apply throttle factors from previous DTM action
            reads = self._apply_throttle(reads)
            writes = self._apply_throttle(writes)

            # ---- Stage 2: compute power trace ----------------------------
            bank_power, logic_power = self.power_model.compute_power_trace(
                reads, writes, self.bank_active,
                freq_scale=self.freq_scale,
                voltage_scale=self.voltage_scale,
            )
            self.power_model.write_power_trace(
                bank_power, logic_power, self.cfg.power_trace_file
            )
            self.power_model.append_power_line(
                bank_power, logic_power, self.cfg.full_power_trace_file
            )

            # ---- Stage 3: run HotSpot ------------------------------------
            ok = self.hotspot.run()
            if not ok:
                log.warning("HotSpot invocation failed at step %d; skipping DTM", step)
                self._record_history(step, bank_power, None)
                continue

            bank_temps = self.hotspot.get_bank_temperatures()

            # Feed temperatures back to power model for leakage loop
            self.power_model.set_bank_temperatures(bank_temps)

            # ---- Stage 4: DTM policy -------------------------------------
            action = self.dtm_policy.evaluate(bank_temps)
            self._apply_action(action)

            self._record_history(step, bank_power, bank_temps)
            self._write_bank_state()

            elapsed = time.time() - t0
            if step % 10 == 0:
                log.info(
                    "Step %d/%d  peak_temp=%.2f°C  freq=%.2f  lpm_ch=%s  (%.2fs)",
                    step, iterations,
                    float(np.max(bank_temps)) if bank_temps is not None else 0,
                    self.freq_scale,
                    sorted(action.lpm_channels),
                    elapsed,
                )

        log.info("Pipeline finished after %d iterations", iterations)
        return self.history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_access_trace(self, step, trace_source, accelsim_cmd, per_interval_dir,
                          benchmark=None):
        """Return (reads, writes) for the current interval."""
        # Option 0: benchmark workload
        if benchmark is not None:
            if callable(benchmark):
                return benchmark(self.cfg, step)
            from .benchmarks import get_benchmark
            fn = get_benchmark(benchmark)
            return fn(self.cfg, step)

        # Option A: per-interval files
        if per_interval_dir:
            path = os.path.join(per_interval_dir, "interval_%d.txt" % step)
            if os.path.isfile(path):
                return self.parser.parse_raw_access_file(path)

        # Option B: single static trace file
        if trace_source and os.path.isfile(trace_source):
            return self.parser.parse_raw_access_file(trace_source)

        # Option C: live Accel-Sim run
        if accelsim_cmd:
            return self._run_accelsim(accelsim_cmd)

        # Fallback: idle (no accesses)
        return [0] * self.cfg.NUM_BANKS, [0] * self.cfg.NUM_BANKS

    def _run_accelsim(self, cmd):
        """Run Accel-Sim as a subprocess and parse its stdout."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            reads, writes, _ = self.parser.parse_accelsim_output(result.stdout)
            return reads, writes
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log.error("Accel-Sim execution failed: %s", exc)
            return [0] * self.cfg.NUM_BANKS, [0] * self.cfg.NUM_BANKS

    def _apply_throttle(self, counts):
        """Reduce access counts by the per-bank throttle factor."""
        return [int(c * f) for c, f in zip(counts, self.throttle_factors)]

    def _apply_action(self, action):
        """Translate a ``DTMAction`` into pipeline state."""
        self.throttle_factors = list(action.throttle_factors)
        self.freq_scale = action.freq_scale
        self.voltage_scale = action.voltage_scale
        self.bank_active = [1 if a else 0 for a in action.bank_active]

    def _write_bank_state(self):
        """Write the current bank-active state to ``BankStateData.txt``."""
        bpl = self.cfg.banks_per_layer
        with open(self.cfg.bank_state_file, "w") as fh:
            for i in range(0, self.cfg.NUM_BANKS, bpl):
                line = " ".join(str(x) for x in self.bank_active[i : i + bpl])
                fh.write(line + "\n")

    def _record_history(self, step, bank_power, bank_temps):
        self.history["bank_power"].append(list(bank_power))
        self.history["freq_scale"].append(self.freq_scale)
        self.history["voltage_scale"].append(self.voltage_scale)
        self.history["lpm_channels"].append(set(
            ch for ch in range(self.cfg.NUM_CHANNELS)
            if not self.bank_active[self.cfg.channel_to_banks(ch)[0]]
        ))
        self.history["throttle_factors"].append(list(self.throttle_factors))
        if bank_temps is not None:
            self.history["bank_temps"].append(bank_temps.tolist())
        else:
            self.history["bank_temps"].append([0.0] * self.cfg.NUM_BANKS)
