from abc import ABC
from typing import Union, Sequence

import torch
from torch.nn.modules.utils import _pair


class LearningRule(ABC):
    r"""
    Abstract object for Learning rule
    """

    def __init__(self):
        r"""
        Initialize learning rule object
        """

    def __call__(self, network):
        r"""
        Update weight connections

        Args:
            network ([FastSpike.Network]): Network's object
        """
        self.update_eligibility(network)

    def update_eligibility(self, network):
        # Decay eligibility trace
        network.eligibility *= network.neurons.trace_decay
        if network.neurons.traces_additive:
            network.eligibility += network.neurons.trace_scale * network.spikes
        else:
            network.eligibility.masked_fill_(
                network.spikes, network.neurons.trace_scale
            )


class STDP(LearningRule):
    r"""
    Spike-timing-dependent plasticity (STDP) learning rule.
    """

    def __init__(
        self,
        nu: Union[float, Sequence[float]],
    ) -> None:
        """
        Initialize STD learning rule

        Args:
            nu (Union[float, Sequence[float]]): Learning rate for Post-Pre(LTD) and Pre-Post(LTP) events.
        """
        super().__init__()
        self.nu = _pair(nu)

    def __call__(self, network):
        r"""
        Update weight connections

        Args:
            network ([FastSpike.Network]): Network's object
        """
        super().__call__(network)
        weight_update = torch.zeros_like(network.weight)
        # Pre-Post activities
        weight_update += self.nu[1] * torch.einsum(
            "bix,bxj->ij",
            network.eligibility.unsqueeze(-1),
            network.spikes.float().unsqueeze(-2),
        )
        # Post-Pre activities
        weight_update += self.nu[0] * torch.einsum(
            "bix,bxj->ij",
            network.spikes.float().unsqueeze(-1),
            network.eligibility.unsqueeze(-2),
        )

        network.weight += weight_update * network.adjacency
