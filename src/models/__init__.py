"""Neural receiver models package for nrx-bounded.

Exports:
- CNNNeuralReceiver / NeuralReceiver_CNN / ResidualBlock (Baseline DeepRx)
- ProposedNeuralReceiver / NeuralReceiver_proposed / FactorizedAxialAttention / PositionalEncoding2D / LMMSEFeatureExtractor
"""

from src.models.cnn import (
    CNNNeuralReceiver,
    NeuralReceiver,
    NeuralReceiver_CNN,
    ResidualBlock,
)
from src.models.axial import (
    AxialAttentionBlock,
    FactorizedAxialAttention,
    LMMSEFeatureExtractor,
    PositionalEncoding2D,
    ProposedNeuralReceiver,
    NeuralReceiver_proposed,
)
from src.models.gate import (
    GateMLP,
    GatedNeuralReceiver,
    combine_llr,
    physics_ramp,
)

__all__ = [
    "CNNNeuralReceiver",
    "NeuralReceiver",
    "NeuralReceiver_CNN",
    "ResidualBlock",
    "AxialAttentionBlock",
    "FactorizedAxialAttention",
    "LMMSEFeatureExtractor",
    "PositionalEncoding2D",
    "ProposedNeuralReceiver",
    "NeuralReceiver_proposed",
    "GateMLP",
    "GatedNeuralReceiver",
    "combine_llr",
    "physics_ramp",
]
