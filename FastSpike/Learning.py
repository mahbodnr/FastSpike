from abc import ABC
from typing import Union, Tuple, Optional, Sequence

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