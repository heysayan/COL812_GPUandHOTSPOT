"""
Closed-Loop Thermal Management Pipeline for HBM on GPU (AccelSim + HotSpot).

Pipeline stages:
  1. Accel-Sim simulates benchmark runs → produces HBM access traces
  2. HBM Power Model converts access traces → power traces
  3. HotSpot computes temperatures from power traces
  4. DTM Policy reads temperatures and adjusts simulation parameters
"""

from .config import PipelineConfig
from .accelsim_trace_parser import AccelSimTraceParser
from .hbm_power_model import HBMPowerModel
from .hotspot_interface import HotSpotInterface
from .dtm_policies import (
    DTMPolicy,
    ThrottlingPolicy,
    DVFSPolicy,
    LowPowerModePolicy,
    CompositeDTMPolicy,
)
from .pipeline_runner import PipelineRunner

__all__ = [
    "PipelineConfig",
    "AccelSimTraceParser",
    "HBMPowerModel",
    "HotSpotInterface",
    "DTMPolicy",
    "ThrottlingPolicy",
    "DVFSPolicy",
    "LowPowerModePolicy",
    "CompositeDTMPolicy",
    "PipelineRunner",
]
