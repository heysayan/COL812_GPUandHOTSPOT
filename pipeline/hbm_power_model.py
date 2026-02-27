"""
Realistic HBM power model.

Converts per-bank read / write access counts into a power trace suitable for
HotSpot.  Supports:
  * Dynamic read / write energy
  * Refresh energy
  * Static (leakage) power
  * DVFS-aware power scaling  (P ∝ C·V²·f)
  * Per-bank active / low-power-mode gating
"""

import numpy as np


class HBMPowerModel:
    """Compute per-bank power from access counts and DVFS state."""

    def __init__(self, config):
        self.cfg = config

    def compute_power_trace(
        self,
        reads_per_bank,
        writes_per_bank,
        bank_active,
        freq_scale=1.0,
        voltage_scale=1.0,
    ):
        """
        Return ``(bank_power, logic_power)`` where each is a list of floats.

        Parameters
        ----------
        reads_per_bank  : list[int]   — read accesses per bank this interval
        writes_per_bank : list[int]   — write accesses per bank this interval
        bank_active     : list[int]   — 1 = active, 0 = low-power-mode
        freq_scale      : float       — frequency scale factor (1.0 = nominal)
        voltage_scale   : float       — voltage scale factor  (1.0 = nominal)
        """
        cfg = self.cfg
        num_banks = cfg.NUM_BANKS
        timestep_us = cfg.TIMESTEP_US

        # DVFS power scaling: dynamic power ∝ V² · f
        dvfs_dynamic_factor = (voltage_scale ** 2) * freq_scale
        # Leakage scales roughly linearly with voltage
        dvfs_static_factor = voltage_scale

        # Refresh energy per timestep (constant background)
        avg_refresh_intervals = timestep_us / cfg.T_REFI_US
        avg_refresh_rows = avg_refresh_intervals * cfg.ROWS_REFRESHED_PER_INTERVAL
        refresh_energy = avg_refresh_rows * cfg.ENERGY_PER_REFRESH_PJ
        avg_refresh_power = refresh_energy / (timestep_us * 1000)  # nJ / ns → W

        bank_power = [0.0] * num_banks
        for b in range(num_banks):
            if not bank_active[b]:
                # In low-power mode only leakage applies (at reduced rate)
                bank_power[b] = round(cfg.BANK_STATIC_POWER_W * dvfs_static_factor * 0.1, 6)
                continue

            rd_energy = reads_per_bank[b] * cfg.ENERGY_PER_READ_PJ
            wr_energy = writes_per_bank[b] * cfg.ENERGY_PER_WRITE_PJ
            dynamic_power = (rd_energy + wr_energy) / (timestep_us * 1000)
            dynamic_power *= dvfs_dynamic_factor

            static_power = cfg.BANK_STATIC_POWER_W * dvfs_static_factor
            bank_power[b] = round(dynamic_power + static_power + avg_refresh_power, 6)

        # Logic-core power (for 3Dmem stack)
        logic_power = []
        if cfg.TYPE_OF_STACK == "3Dmem":
            lp = cfg.LOGIC_CORE_POWER_W * dvfs_dynamic_factor
            logic_power = [round(lp, 6)] * cfg.NUM_LOGIC_CORES

        return bank_power, logic_power

    # ------------------------------------------------------------------
    # Trace-file formatting helpers
    # ------------------------------------------------------------------
    def format_ptrace_header(self):
        """Return the HotSpot ptrace header string."""
        parts = []
        if self.cfg.TYPE_OF_STACK == "3Dmem":
            for i in range(self.cfg.NUM_LOGIC_CORES):
                parts.append("LC_%d" % i)
        for z in range(self.cfg.BANKS_IN_Z):
            for x in range(self.cfg.BANKS_IN_X):
                for y in range(self.cfg.BANKS_IN_Y):
                    bank = z * self.cfg.BANKS_IN_X * self.cfg.BANKS_IN_Y + x * self.cfg.BANKS_IN_Y + y
                    parts.append("B_%d" % bank)
        return "\t".join(parts)

    def format_power_line(self, bank_power, logic_power):
        """Return a single power-trace data line."""
        parts = [str(p) for p in logic_power] + [str(p) for p in bank_power]
        return "\t".join(parts)

    def write_power_trace(self, bank_power, logic_power, filepath):
        """Overwrite *filepath* with header + single power line."""
        header = self.format_ptrace_header()
        line = self.format_power_line(bank_power, logic_power)
        with open(filepath, "w") as fh:
            fh.write(header + "\n")
            fh.write(line + "\n")

    def append_power_line(self, bank_power, logic_power, filepath):
        """Append one power line to *filepath*."""
        line = self.format_power_line(bank_power, logic_power)
        with open(filepath, "a") as fh:
            fh.write(line + "\n")
