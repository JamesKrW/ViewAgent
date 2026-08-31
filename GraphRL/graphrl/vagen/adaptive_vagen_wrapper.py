"""Compatibility import for the adaptive VAGEN-SLIME wrapper."""

from graphrl.slime.adaptive_wrapper import AdaptiveSlimeWrapper


class AdaptiveVagenWrapper(AdaptiveSlimeWrapper):
    """Deprecated spelling of :class:`AdaptiveSlimeWrapper`."""


__all__ = ["AdaptiveVagenWrapper"]
