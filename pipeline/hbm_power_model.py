"""
Realistic HBM2/HBM2E power model.

Converts per-bank read / write access counts into a power trace suitable for
HotSpot.  Power components:

1. **Dynamic read/write** — energy per access × count / timestep, scaled by
   V²·f for DVFS.
2. **Activation/precharge** — each read/write implies a row activation and
   precharge; energy from CACTI3DD.
3. **Refresh** — periodic row-refresh commands; constant background.
4. **Static (leakage)** — temperature-dependent: P_leak(T) = P_ref ·
   exp(α·(T − T_ref)).  Doubles roughly every 10–12 °C, matching MOSFET
   sub-threshold leakage behaviour at 20 nm.
5. **Logic-core power** — I/O PHY, DLL, command decoder, ECC engine; split
   into a static + dynamic component.

Supports DVFS-aware scaling and per-bank low-power-mode gating.
"""

import math

import numpy as np


class HBMPowerModel:
    """Compute per-bank power from access counts and DVFS state."""

    def __init__(self, config):
        self.cfg = config
        # Cache temperature state for leakage feedback loop.
        # Initialised to ambient; updated externally after each HotSpot run.
        self._bank_temps_c = np.full(config.NUM_BANKS, 45.0)

    def set_bank_temperatures(self, temps_c):
        """Update the per-bank temperatures used for leakage calculation.

        Parameters
        ----------
        temps_c : array-like of float
            Per-bank temperatures in °C (length ``NUM_BANKS``).
        """
        self._bank_temps_c = np.asarray(temps_c, dtype=float)

    # ------------------------------------------------------------------
    # Core power computation
    # ------------------------------------------------------------------
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
        timestep_ns = timestep_us * 1000  # convert µs → ns

        # DVFS power scaling: dynamic power ∝ V² · f
        dvfs_dynamic = (voltage_scale ** 2) * freq_scale
        # Leakage scales roughly linearly with voltage
        dvfs_static = voltage_scale

        # ---- Refresh background power (constant per bank) ---------------
        avg_refresh_intervals = timestep_us / cfg.T_REFI_US
        avg_refresh_rows = avg_refresh_intervals * cfg.ROWS_REFRESHED_PER_INTERVAL
        refresh_energy_nj = avg_refresh_rows * cfg.ENERGY_PER_REFRESH_NJ
        refresh_power_w = refresh_energy_nj / timestep_ns  # nJ / ns = W

        # ---- Per-bank power ---------------------------------------------
        bank_power = [0.0] * num_banks
        for b in range(num_banks):
            # Temperature-dependent leakage
            temp_c = self._bank_temps_c[b] if b < len(self._bank_temps_c) else 45.0
            leakage_w = self._leakage_power(temp_c) * dvfs_static

            if not bank_active[b]:
                # Low-power mode: only reduced leakage
                bank_power[b] = round(
                    leakage_w * cfg.LOW_POWER_LEAKAGE_FRACTION, 6
                )
                continue

            rd = reads_per_bank[b]
            wr = writes_per_bank[b]
            total_accesses = rd + wr

            # Dynamic: column read/write energy
            rw_energy_nj = (
                rd * cfg.ENERGY_PER_READ_NJ + wr * cfg.ENERGY_PER_WRITE_NJ
            )
            # Activation + precharge: each access opens and closes a row
            act_pre_energy_nj = total_accesses * (
                cfg.ENERGY_PER_ACT_NJ + cfg.ENERGY_PER_PRE_NJ
            )
            dynamic_power_w = (rw_energy_nj + act_pre_energy_nj) / timestep_ns
            dynamic_power_w *= dvfs_dynamic

            bank_power[b] = round(
                dynamic_power_w + leakage_w + refresh_power_w, 6
            )

        # ---- Logic-core power (for 3Dmem stack) -------------------------
        logic_power = []
        if cfg.TYPE_OF_STACK == "3Dmem":
            # Each logic core: static + dynamic (data routing / ECC / I/O)
            lc_static = cfg.LOGIC_CORE_STATIC_POWER_W * dvfs_static
            lc_dynamic = cfg.LOGIC_CORE_DYNAMIC_POWER_W * dvfs_dynamic
            logic_power = [round(lc_static + lc_dynamic, 6)] * cfg.NUM_LOGIC_CORES

        return bank_power, logic_power

    # ------------------------------------------------------------------
    # Leakage model
    # ------------------------------------------------------------------
    def _leakage_power(self, temp_c):
        """Temperature-dependent leakage: P(T) = P_ref · exp(α·(T − T_ref)).

        This models sub-threshold leakage in 20 nm MOSFET technology.
        With α ≈ 0.06 the power roughly doubles every ~11.5 °C, consistent
        with published HBM2 characterisation data.
        """
        cfg = self.cfg
        delta = temp_c - cfg.BANK_STATIC_REF_TEMP_C
        # Clamp to avoid overflow in extreme scenarios
        exponent = cfg.LEAKAGE_TEMP_COEFF * min(delta, 200.0)
        return cfg.BANK_STATIC_POWER_W_REF * math.exp(exponent)

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
                    bank = (
                        z * self.cfg.BANKS_IN_X * self.cfg.BANKS_IN_Y
                        + x * self.cfg.BANKS_IN_Y
                        + y
                    )
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
