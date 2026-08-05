"""GPU-accelerated real-time sweep renderer for multichannel timeseries."""

from .__version__ import __version__ as __version__
from .channel_plot import ChannelPlotWidget
from .controls import ChannelPlotControlsWidget
from .overlays import ChannelLabelOverlay, ScaleBarOverlay
from .scatter_widget import ScatterConfig, ScatterWidget
from .spectrum_widget import SpectrumConfig, SpectrumWidget
from .sweep_buffer import SweepEvent
from .sweep_widget import SweepConfig, SweepWidget

__all__ = [
    "ChannelLabelOverlay",
    "ChannelPlotControlsWidget",
    "ChannelPlotWidget",
    "ScaleBarOverlay",
    "ScatterConfig",
    "ScatterWidget",
    "SpectrumConfig",
    "SpectrumWidget",
    "SweepConfig",
    "SweepEvent",
    "SweepWidget",
]
