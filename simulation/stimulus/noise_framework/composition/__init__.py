"""Physical noise composition modules."""

from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.composition.pauli_composition import (
    compose_mixed_pauli,
    compose_pauli_masks_xor,
)
from simulation.stimulus.noise_framework.composition.probability_composition import (
    compose_independent_bernoulli,
    compose_xor_parity_bernoulli,
    place_into_21_mechanism_vector,
)

__all__ = [
    "NoiseComposer",
    "compose_independent_bernoulli",
    "compose_mixed_pauli",
    "compose_pauli_masks_xor",
    "compose_xor_parity_bernoulli",
    "place_into_21_mechanism_vector",
]
