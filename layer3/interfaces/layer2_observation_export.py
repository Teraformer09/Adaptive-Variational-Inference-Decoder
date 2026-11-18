"""Layer 3 -> Layer 2 Observation Export Adapter (`layer2_observation_export.py`)."""

from __future__ import annotations

from typing import Any, Dict
from layer3.interfaces.posterior_packet import PosteriorPacket


def export_posterior_for_layer2(packet: PosteriorPacket) -> Dict[str, Any]:
    """Export a validated `PosteriorPacket` into Layer 2's syndrome-only observation payload."""
    if not isinstance(packet, PosteriorPacket):
        raise TypeError(f"Expected PosteriorPacket, got {type(packet)}")
    return packet.to_layer2_dict()
