"""
Accel-Sim trace parser for extracting HBM memory access statistics.

Reads memory-access trace files produced by Accel-Sim's trace-driven
simulation and returns per-bank read / write counts for a given sampling
interval.
"""

import os
import re


class AccelSimTraceParser:
    """Parse Accel-Sim output to extract per-bank HBM read/write counts."""

    # Regex for a GPGPU-Sim style memory-access stat line:
    #   gpu_tot_ipc = ...
    #   L2_cache_stats_breakdown[GLOBAL_ACC_R][HIT] = 12345
    # We look for lines referencing DRAM access counts per bank.
    _DRAM_RD_RE = re.compile(
        r"dram\[(\d+)\]:\s*bk\[(\d+)\]:\s*rd\s*=\s*(\d+)", re.IGNORECASE
    )
    _DRAM_WR_RE = re.compile(
        r"dram\[(\d+)\]:\s*bk\[(\d+)\]:\s*wr\s*=\s*(\d+)", re.IGNORECASE
    )

    # Simpler Accel-Sim stats format: tab-separated per-bank values
    #   "dram_reads_per_bank = 100,200,300,..."
    _BULK_RD_RE = re.compile(
        r"dram_reads_per_bank\s*=\s*(.+)", re.IGNORECASE
    )
    _BULK_WR_RE = re.compile(
        r"dram_writes_per_bank\s*=\s*(.+)", re.IGNORECASE
    )

    # GPU cycle counter
    _CYCLE_RE = re.compile(r"gpu_sim_cycle\s*=\s*(\d+)", re.IGNORECASE)
    _TOT_CYCLE_RE = re.compile(r"gpu_tot_sim_cycle\s*=\s*(\d+)", re.IGNORECASE)

    def __init__(self, config):
        self.cfg = config
        self.num_banks = config.NUM_BANKS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse_trace_file(self, filepath):
        """
        Parse a single Accel-Sim stats / trace file and return
        ``(reads_per_bank, writes_per_bank, total_cycles)``.

        Both ``reads_per_bank`` and ``writes_per_bank`` are lists of length
        ``NUM_BANKS``.  ``total_cycles`` is the latest ``gpu_sim_cycle`` found.
        """
        reads = [0] * self.num_banks
        writes = [0] * self.num_banks
        total_cycles = 0

        if not os.path.isfile(filepath):
            return reads, writes, total_cycles

        with open(filepath, "r") as fh:
            for line in fh:
                self._parse_line(line, reads, writes)
                m = self._CYCLE_RE.search(line) or self._TOT_CYCLE_RE.search(line)
                if m:
                    total_cycles = max(total_cycles, int(m.group(1)))

        return reads, writes, total_cycles

    def parse_accelsim_output(self, output_text):
        """
        Parse Accel-Sim textual *stdout / stderr* output (as a string)
        and return ``(reads_per_bank, writes_per_bank, total_cycles)``.
        """
        reads = [0] * self.num_banks
        writes = [0] * self.num_banks
        total_cycles = 0

        for line in output_text.splitlines():
            self._parse_line(line, reads, writes)
            m = self._CYCLE_RE.search(line) or self._TOT_CYCLE_RE.search(line)
            if m:
                total_cycles = max(total_cycles, int(m.group(1)))

        return reads, writes, total_cycles

    def parse_raw_access_file(self, filepath):
        """
        Parse a simple two-line (reads then writes) space-separated file that
        lists per-bank access counts directly — the format already used by the
        legacy ``sample_script.py``.
        """
        reads = [0] * self.num_banks
        writes = [0] * self.num_banks

        if not os.path.isfile(filepath):
            return reads, writes

        with open(filepath, "r") as fh:
            lines = [l.strip() for l in fh if l.strip()]

        if len(lines) >= 1:
            vals = lines[0].split()
            for i, v in enumerate(vals[: self.num_banks]):
                reads[i] = int(v)
        if len(lines) >= 2:
            vals = lines[1].split()
            for i, v in enumerate(vals[: self.num_banks]):
                writes[i] = int(v)

        return reads, writes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _parse_line(self, line, reads, writes):
        # Per-bank style: dram[0]: bk[3]: rd = 42
        m = self._DRAM_RD_RE.search(line)
        if m:
            bank_idx = self._global_bank_index(int(m.group(1)), int(m.group(2)))
            if 0 <= bank_idx < self.num_banks:
                reads[bank_idx] += int(m.group(3))
            return

        m = self._DRAM_WR_RE.search(line)
        if m:
            bank_idx = self._global_bank_index(int(m.group(1)), int(m.group(2)))
            if 0 <= bank_idx < self.num_banks:
                writes[bank_idx] += int(m.group(3))
            return

        # Bulk style: dram_reads_per_bank = 100,200,...
        m = self._BULK_RD_RE.search(line)
        if m:
            vals = re.split(r"[,\s]+", m.group(1).strip())
            for i, v in enumerate(vals[: self.num_banks]):
                reads[i] += int(v)
            return

        m = self._BULK_WR_RE.search(line)
        if m:
            vals = re.split(r"[,\s]+", m.group(1).strip())
            for i, v in enumerate(vals[: self.num_banks]):
                writes[i] += int(v)

    @staticmethod
    def _global_bank_index(dram_chip, local_bank):
        """Map (dram_chip, local_bank) → global bank index 0..127."""
        return dram_chip * 16 + local_bank
