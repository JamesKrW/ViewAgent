"""Compatibility import for the VAGEN-SLIME RL wrapper.

New code imports :class:`graphrl.slime.wrapper.SlimeWrapper`.  The old class
name remains available so external pipeline extensions do not break merely
because the backend implementation changed.
"""

from graphrl.slime.wrapper import SlimeWrapper


class VagenWrapper(SlimeWrapper):
    """Deprecated spelling of :class:`SlimeWrapper`."""


__all__ = ["VagenWrapper"]
