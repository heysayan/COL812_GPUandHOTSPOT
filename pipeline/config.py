"""
Centralized configuration for the closed-loop thermal management pipeline.

All HBM geometry, energy parameters, frequency settings, file paths, and
thermal thresholds are kept here so that every other module imports a single
``PipelineConfig`` instance.
"""

import os


class PipelineConfig:
    """Holds every tuneable knob of the pipeline."""

    # ------------------------------------------------------------------
    # HBM bank / geometry
    # ------------------------------------------------------------------
    BANK_SIZE_MB = 64
    NUM_COLUMNS_K = 1
    BITS_PER_COLUMN = 8
    ROWS_PER_BANK_K = BANK_SIZE_MB / NUM_COLUMNS_K / BITS_PER_COLUMN  # in Kilo

    BANKS_IN_X = 4
    BANKS_IN_Y = 4
    BANKS_IN_Z = 8
    NUM_BANKS = BANKS_IN_X * BANKS_IN_Y * BANKS_IN_Z  # 128

    LOGIC_CORES_IN_X = BANKS_IN_X
    LOGIC_CORES_IN_Y = BANKS_IN_Y
    NUM_LOGIC_CORES = LOGIC_CORES_IN_X * LOGIC_CORES_IN_Y  # 16

    # ------------------------------------------------------------------
    # HBM channel / pseudo-channel layout
    # ------------------------------------------------------------------
    CHANNEL_BUS_WIDTH = 128     # bits
    PSEUDO_CHANNEL_BUS_WIDTH = 64  # bits
    NUM_CHANNELS = 8
    NUM_PSEUDO_CHANNELS = 16

    # ------------------------------------------------------------------
    # Energy per access  (pJ → converted to nJ inside power model)
    # ------------------------------------------------------------------
    ENERGY_PER_READ_PJ = 20.55
    ENERGY_PER_WRITE_PJ = 20.55
    ENERGY_PER_REFRESH_PJ = 3.55

    # ------------------------------------------------------------------
    # Frequency / timing
    # ------------------------------------------------------------------
    DEFAULT_CLOCK_GHZ = 2.0          # nominal HBM clock
    MAX_CLOCK_GHZ = 3.2
    MIN_CLOCK_GHZ = 0.5
    DEFAULT_VOLTAGE_V = 1.1          # nominal supply voltage
    MIN_VOLTAGE_V = 0.8
    MAX_VOLTAGE_V = 1.2

    SAMPLING_INTERVAL_NS = 1_000_000  # 1 ms
    TIMESTEP_US = SAMPLING_INTERVAL_NS / 1000
    T_REFI_US = 7.8
    NUM_REFRESH_COMMANDS_PER_TREFW = 8
    ROWS_REFRESHED_PER_INTERVAL = ROWS_PER_BANK_K / NUM_REFRESH_COMMANDS_PER_TREFW

    BANK_STATIC_POWER_W = 0.0
    LOGIC_CORE_POWER_W = 0.0

    # ------------------------------------------------------------------
    # Thermal thresholds  (°C)
    # ------------------------------------------------------------------
    THROTTLE_TEMP_C = 80.0
    CRITICAL_TEMP_C = 85.0
    COOLDOWN_TEMP_C = 75.0

    DVFS_TEMP_THRESHOLDS_C = [76.0, 78.0, 80.0, 82.0]
    DVFS_FREQ_SCALE_FACTORS = [1.0, 0.85, 0.70, 0.50]
    DVFS_VOLTAGE_SCALE_FACTORS = [1.0, 0.95, 0.90, 0.85]

    # ------------------------------------------------------------------
    # Type of 3-D stack
    # ------------------------------------------------------------------
    TYPE_OF_STACK = "3Dmem"

    # 3Dmem_16core layer structure (from mem.lcf):
    #   layer 0  = mem_ctrl  (logic cores, power dissipation)
    #   layer 1  = mem_tim   (TIM, no power)
    #   layer 2  = mem_bank_1 (power dissipation)
    #   layer 3  = mem_tim
    #   ...
    #   layer 16 = mem_bank_8
    #   layer 17 = mem_tim
    # Total: 18 layers  (1 logic + 8 bank layers, each followed by TIM)
    NUM_HOTSPOT_LAYERS = 18

    # ------------------------------------------------------------------
    # Paths  (overridable via constructor)
    # ------------------------------------------------------------------
    def __init__(
        self,
        hotspot_dir=None,
        repo_root=None,
        board_config=None,
        accelsim_trace_dir=None,
        output_dir=None,
    ):
        # repo_root is the top-level directory of the repository
        if repo_root is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.repo_root = repo_root

        if hotspot_dir is None:
            hotspot_dir = os.path.join(repo_root, "hotspot_tool")
        self.hotspot_dir = hotspot_dir
        self.hotspot_executable = os.path.join(self.hotspot_dir, "hotspot")
        self.board_config = board_config or "3Dmem_16core"

        self.gpu_simulator_dir = os.path.join(repo_root, "gpu-simulator")

        config_root = os.path.join(repo_root, "config", "hotspot", self.board_config)
        self.init_file_external = os.path.join(config_root, "mem.init")
        self.hotspot_config_file = os.path.join(config_root, "mem_hotspot.config")
        self.hotspot_floorplan_folder = config_root
        self.hotspot_layer_file = os.path.join(config_root, "mem.lcf")

        self.accelsim_trace_dir = accelsim_trace_dir or os.path.join(repo_root, "traces")

        self.output_dir = output_dir or os.path.join(repo_root, "pipeline_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Output file names (inside output_dir)
        self.power_trace_file = os.path.join(self.output_dir, "power_mem.trace")
        self.power_trace_total_file = os.path.join(self.output_dir, "tmmpFile_power2")
        self.full_power_trace_file = os.path.join(self.output_dir, "full_power_mem.trace")
        self.full_totalpower_trace_file = os.path.join(self.output_dir, "full_totalpower_mem.trace")
        self.temperature_trace_file = os.path.join(self.output_dir, "temperature_mem.trace")
        self.full_temperature_trace_file = os.path.join(self.output_dir, "full_temperature_mem.trace")
        self.steady_temp_file = os.path.join(self.output_dir, "steady_temperature_mem.log")
        self.grid_steady_file = os.path.join(self.output_dir, "grid_steady_mem.log")
        self.all_transient_file = os.path.join(self.output_dir, "all_transient_mem.init")
        self.init_file = os.path.join(self.output_dir, "temperature_mem.init")
        self.bank_state_file = os.path.join(self.output_dir, "BankStateData.txt")
        self.dtm_log_file = os.path.join(self.output_dir, "dtm_actions.log")
        # Runtime LCF with absolute paths (generated by prepare_lcf)
        self.runtime_layer_file = os.path.join(self.output_dir, "mem.lcf")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def interval_sec(self):
        return self.SAMPLING_INTERVAL_NS * 1e-9

    @property
    def banks_per_layer(self):
        return self.NUM_BANKS // self.BANKS_IN_Z

    def bank_to_channel(self, bank):
        b = bank % 64
        return b // 8

    def bank_to_pseudochannel(self, bank):
        b = bank % 64
        return b // 4

    def pseudochannel_to_banks(self, pchannel):
        start = pchannel * 4
        return list(range(start, start + 4)) + list(range(start + 64, start + 68))

    def channel_to_banks(self, channel):
        start = channel * 8
        return list(range(start, start + 8)) + list(range(start + 64, start + 72))

    def prepare_lcf(self):
        """Generate a runtime ``.lcf`` with absolute floorplan paths.

        The shipped ``mem.lcf`` stores floorplan paths relative to the repo
        root (e.g. ``config/hotspot/3Dmem_16core/mem_ctrl.flp``).  HotSpot
        requires absolute paths, so this method reads the template LCF,
        resolves every ``.flp`` reference against ``self.repo_root``, and
        writes the result to ``self.runtime_layer_file``.
        """
        if not os.path.isfile(self.hotspot_layer_file):
            return

        with open(self.hotspot_layer_file, "r") as fh:
            lines = fh.readlines()

        with open(self.runtime_layer_file, "w") as fh:
            for line in lines:
                stripped = line.strip()
                if stripped.endswith(".flp"):
                    # Resolve relative path against repo root
                    if not os.path.isabs(stripped):
                        stripped = os.path.join(self.repo_root, stripped)
                    fh.write(stripped + "\n")
                else:
                    fh.write(line)

