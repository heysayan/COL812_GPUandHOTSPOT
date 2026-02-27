"""
HotSpot tool interface.

Builds the HotSpot command line, invokes the tool, and parses the resulting
temperature trace files so that the DTM policy can make decisions.
"""

import os
import shutil
import subprocess

import numpy as np


class HotSpotInterface:
    """Invoke HotSpot and read back temperature results."""

    def __init__(self, config):
        self.cfg = config
        self._iteration = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def setup(self):
        """Prepare output files and copy the initial temperature file."""
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

        if os.path.isfile(cfg.init_file_external):
            shutil.copy2(cfg.init_file_external, cfg.init_file)

        # Generate runtime LCF with absolute floorplan paths
        cfg.prepare_lcf()

        header = self._ptrace_header()
        for path in (
            cfg.full_temperature_trace_file,
            cfg.full_power_trace_file,
            cfg.full_totalpower_trace_file,
        ):
            with open(path, "w") as fh:
                fh.write(header + "\n")

        # Create initial BankStateData.txt (all banks active) — HotSpot
        # reads this from the current working directory.
        bpl = cfg.NUM_BANKS // cfg.BANKS_IN_Z
        with open(cfg.bank_state_file, "w") as fh:
            for i in range(0, cfg.NUM_BANKS, bpl):
                fh.write(" ".join(["1"] * bpl) + "\n")

        self._iteration = 0

    # ------------------------------------------------------------------
    # Run HotSpot
    # ------------------------------------------------------------------
    def run(self, voltage_per_layer=None):
        """
        Invoke HotSpot on the current power trace and produce a temperature
        trace.  Returns ``True`` on success.

        Parameters
        ----------
        voltage_per_layer : list[float] | None
            Supply voltage for each of the ``NUM_HOTSPOT_LAYERS`` layers
            (3Dmem_16core has 18: mem_ctrl + TIM + 8×(bank + TIM)).
            Defaults to all ``DEFAULT_VOLTAGE_V``.
        """
        cfg = self.cfg
        num_layers = cfg.NUM_HOTSPOT_LAYERS

        if voltage_per_layer is None:
            voltage_per_layer = [cfg.DEFAULT_VOLTAGE_V] * num_layers

        voltage_str = ",".join("%.2f" % v for v in voltage_per_layer)
        layer_flags = ",".join(["1"] * num_layers) + ","

        # Use the runtime LCF (with absolute paths) if it was prepared
        layer_file = cfg.runtime_layer_file
        if not os.path.isfile(layer_file):
            layer_file = cfg.hotspot_layer_file

        cmd_parts = [
            cfg.hotspot_executable,
            "-c", cfg.hotspot_config_file,
            "-p", cfg.power_trace_file,
            "-pTot", cfg.power_trace_total_file,
            "-o", cfg.temperature_trace_file,
            "-model_secondary", "1",
            "-model_type", "grid",
            "-steady_file", cfg.steady_temp_file,
            "-all_transient_file", cfg.all_transient_file,
            "-grid_steady_file", cfg.grid_steady_file,
            "-steady_state_print_disable", "1",
            "-l", layer_flags,
            "-type", cfg.TYPE_OF_STACK,
            "-sampling_intvl", str(cfg.interval_sec),
            "-grid_layer_file", layer_file,
            "-detailed_3D", "on",
            "-v", voltage_str,
        ]

        # Use init file from second iteration onward, or if external init exists
        if os.path.isfile(cfg.init_file):
            cmd_parts += ["-init_file", cfg.init_file]

        try:
            subprocess.run(cmd_parts, check=True, capture_output=True, text=True,
                           cwd=self.cfg.output_dir)
        except FileNotFoundError:
            # HotSpot binary not built — allow graceful degradation for testing
            return False
        except subprocess.CalledProcessError:
            return False

        # Propagate transient state for next iteration
        if os.path.isfile(cfg.all_transient_file):
            shutil.copy2(cfg.all_transient_file, cfg.init_file)

        # Append latest data to full trace files
        self._append_last_line(cfg.temperature_trace_file, cfg.full_temperature_trace_file)
        self._append_last_line(cfg.power_trace_file, cfg.full_power_trace_file)
        self._append_last_line(cfg.power_trace_total_file, cfg.full_totalpower_trace_file)

        self._iteration += 1
        return True

    # ------------------------------------------------------------------
    # Temperature readers
    # ------------------------------------------------------------------
    def get_bank_temperatures(self):
        """Return an ``np.ndarray`` of per-bank temperatures (°C)."""
        return self._read_temps(self.cfg.temperature_trace_file)

    def get_channel_temperatures(self):
        """Return an array of per-channel peak temperatures (°C)."""
        temps = self.get_bank_temperatures()
        ch_temps = np.zeros(self.cfg.NUM_CHANNELS)
        for c in range(self.cfg.NUM_CHANNELS):
            banks = self.cfg.channel_to_banks(c)
            ch_temps[c] = np.max(temps[banks])
        return ch_temps

    def get_pseudochannel_temperatures(self):
        """Return an array of per-pseudo-channel peak temperatures (°C)."""
        temps = self.get_bank_temperatures()
        pc_temps = np.zeros(self.cfg.NUM_PSEUDO_CHANNELS)
        for pc in range(self.cfg.NUM_PSEUDO_CHANNELS):
            banks = self.cfg.pseudochannel_to_banks(pc)
            pc_temps[pc] = np.max(temps[banks])
        return pc_temps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _ptrace_header(self):
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

    def _read_temps(self, filepath):
        """Read bank temperatures from the temperature trace file.

        The file has a header row (with LC_* and B_* columns) followed by a
        data row.  Bank values start after the logic-core columns.
        """
        temps = np.zeros(self.cfg.NUM_BANKS)
        if not os.path.isfile(filepath):
            return temps
        with open(filepath, "r") as fh:
            fh.readline()  # header
            data_line = fh.readline().strip()
        if not data_line:
            return temps
        values = data_line.split()
        offset = self.cfg.NUM_LOGIC_CORES if self.cfg.TYPE_OF_STACK == "3Dmem" else 0
        for i in range(min(self.cfg.NUM_BANKS, len(values) - offset)):
            temps[i] = float(values[offset + i])
        return temps

    @staticmethod
    def _append_last_line(src, dst):
        """Append the last line of *src* to *dst*."""
        if not os.path.isfile(src):
            return
        with open(src, "r") as fh:
            lines = fh.readlines()
        if lines:
            with open(dst, "a") as fh:
                fh.write(lines[-1] if lines[-1].endswith("\n") else lines[-1] + "\n")
