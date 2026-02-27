"""
Dynamic Thermal Management (DTM) policies.

Each policy reads current temperatures and returns *actions* that the pipeline
runner applies before the next simulation interval.  Supported actions:

* **Throttle** — reduce read/write access rates for hot banks/channels.
* **DVFS** — scale frequency and voltage globally or per-channel.
* **Low-Power Mode** — gate entire banks / pseudo-channels / channels.
* **Composite** — chain several policies together.
"""

from abc import ABC, abstractmethod
import logging

import numpy as np

log = logging.getLogger(__name__)


# ======================================================================
# DTM action dataclass
# ======================================================================
class DTMAction:
    """Container for the adjustments a DTM policy requests."""

    __slots__ = (
        "throttle_factors",
        "freq_scale",
        "voltage_scale",
        "bank_active",
        "lpm_channels",
        "lpm_pseudochannels",
        "migration_pairs",
    )

    def __init__(self, num_banks):
        # Per-bank multiplicative throttle factor (1.0 = no throttle, 0.0 = full block)
        self.throttle_factors = [1.0] * num_banks
        # Global DVFS knobs
        self.freq_scale = 1.0
        self.voltage_scale = 1.0
        # Per-bank active flag (True = active)
        self.bank_active = [True] * num_banks
        # Sets of channels / pseudo-channels in low-power mode
        self.lpm_channels = set()
        self.lpm_pseudochannels = set()
        # List of (src_channel, dst_channel) migration pairs
        self.migration_pairs = []


# ======================================================================
# Base policy
# ======================================================================
class DTMPolicy(ABC):
    """Abstract base for all DTM policies."""

    def __init__(self, config):
        self.cfg = config

    @abstractmethod
    def evaluate(self, bank_temps, current_action=None):
        """
        Given per-bank temperatures (np.ndarray, °C) and the current
        action state, return an updated ``DTMAction``.
        """
        ...


# ======================================================================
# Throttling policy
# ======================================================================
class ThrottlingPolicy(DTMPolicy):
    """
    Linearly throttle read/write accesses for banks whose temperature
    exceeds ``throttle_temp``, reaching zero at ``critical_temp``.
    """

    def __init__(self, config, throttle_temp=None, critical_temp=None):
        super().__init__(config)
        self.throttle_temp = throttle_temp or config.THROTTLE_TEMP_C
        self.critical_temp = critical_temp or config.CRITICAL_TEMP_C

    def evaluate(self, bank_temps, current_action=None):
        action = current_action or DTMAction(self.cfg.NUM_BANKS)
        span = self.critical_temp - self.throttle_temp
        if span <= 0:
            return action

        for b in range(self.cfg.NUM_BANKS):
            t = bank_temps[b]
            if t >= self.critical_temp:
                action.throttle_factors[b] = 0.0
                log.info("Bank %d throttled to 0%% (T=%.1f°C)", b, t)
            elif t >= self.throttle_temp:
                factor = 1.0 - (t - self.throttle_temp) / span
                action.throttle_factors[b] = round(max(factor, 0.0), 4)
                log.debug("Bank %d throttled to %.0f%% (T=%.1f°C)", b, factor * 100, t)
            else:
                action.throttle_factors[b] = 1.0

        return action


# ======================================================================
# DVFS policy
# ======================================================================
class DVFSPolicy(DTMPolicy):
    """
    Step-wise global Dynamic Voltage & Frequency Scaling.

    Temperature thresholds and corresponding scale factors are taken from
    ``PipelineConfig.DVFS_TEMP_THRESHOLDS_C`` and
    ``PipelineConfig.DVFS_FREQ_SCALE_FACTORS``.
    """

    def evaluate(self, bank_temps, current_action=None):
        action = current_action or DTMAction(self.cfg.NUM_BANKS)
        peak_temp = float(np.max(bank_temps))

        thresholds = self.cfg.DVFS_TEMP_THRESHOLDS_C
        freq_factors = self.cfg.DVFS_FREQ_SCALE_FACTORS
        volt_factors = self.cfg.DVFS_VOLTAGE_SCALE_FACTORS

        # Walk thresholds from hottest to coolest
        action.freq_scale = freq_factors[0]
        action.voltage_scale = volt_factors[0]
        for i in range(len(thresholds) - 1, -1, -1):
            if peak_temp >= thresholds[i]:
                action.freq_scale = freq_factors[i]
                action.voltage_scale = volt_factors[i]
                log.info(
                    "DVFS: peak=%.1f°C → freq_scale=%.2f, voltage_scale=%.2f",
                    peak_temp, action.freq_scale, action.voltage_scale,
                )
                break

        return action


# ======================================================================
# Low-Power Mode policy
# ======================================================================
class LowPowerModePolicy(DTMPolicy):
    """
    Put entire channels into low-power mode when their peak temperature
    exceeds ``critical_temp`` and bring them back when below ``cooldown_temp``.
    """

    def __init__(self, config, critical_temp=None, cooldown_temp=None):
        super().__init__(config)
        self.critical_temp = critical_temp or config.CRITICAL_TEMP_C
        self.cooldown_temp = cooldown_temp or config.COOLDOWN_TEMP_C

    def evaluate(self, bank_temps, current_action=None):
        action = current_action or DTMAction(self.cfg.NUM_BANKS)

        for ch in range(self.cfg.NUM_CHANNELS):
            banks = self.cfg.channel_to_banks(ch)
            ch_peak = float(np.max(bank_temps[banks]))

            if ch_peak >= self.critical_temp:
                action.lpm_channels.add(ch)
                for b in banks:
                    action.bank_active[b] = False
                log.info("Channel %d → LPM (peak=%.1f°C)", ch, ch_peak)
            elif ch in action.lpm_channels and ch_peak < self.cooldown_temp:
                action.lpm_channels.discard(ch)
                for b in banks:
                    action.bank_active[b] = True
                log.info("Channel %d ← LPM restored (peak=%.1f°C)", ch, ch_peak)

        return action


# ======================================================================
# Composite policy
# ======================================================================
class CompositeDTMPolicy(DTMPolicy):
    """Chain multiple DTM policies; each one refines the action of the previous."""

    def __init__(self, config, policies=None):
        super().__init__(config)
        self.policies = policies or []

    def add_policy(self, policy):
        self.policies.append(policy)

    def evaluate(self, bank_temps, current_action=None):
        action = current_action or DTMAction(self.cfg.NUM_BANKS)
        for policy in self.policies:
            action = policy.evaluate(bank_temps, action)
        return action
