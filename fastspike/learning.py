from abc import ABC
from typing import Union, Tuple, Optional, Sequence

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
        batch_integration: callable = torch.mean,
    ) -> None:
        """
        Initialize STD learning rule

        Args:
            nu (Union[float, Sequence[float]]): Learning rate for Post-Pre(LTD) and Pre-Post(LTP) events.
            batch_integration (callable): Method used for integrating weight changes along the batches axis.
        """
        super().__init__()
        self.nu = _pair(nu)
        self.batch_integration = batch_integration

    def __call__(self, network):
        r"""
        Update weight connections

        Args:
            network ([FastSpike.Network]): Network's object
        """
        super().__call__(network)
        # Post-Pre activities
        if self.nu[0]:
            network.weight -= self.batch_integration(
                (
                    network.spikes.unsqueeze(-2)
                    * network.eligibility.unsqueeze(-1)
                    * network.weight
                ),
                dim=0,
            )
        # Pre-Post activities.
        if self.nu[1]:
            network.weight += self.batch_integration(
                (
                    network.spikes.unsqueeze(-1)
                    * network.eligibility.unsqueeze(-2)
                    * network.weight
                ),
                dim=0,
            )
