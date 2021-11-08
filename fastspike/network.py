from typing import Tuple, Optional

import torch
from torch.nn import Parameter

from fastspike.neurons import NeuronGroup
from fastspike.neurons import NeuronType
from fastspike.learning import LearningRule


class Network(torch.nn.Module):
    r"""
    FastSpike principal Network object.
    """

    def __init__(
        self,
        neurons_type: NeuronType,
        learning_rule: Optional[LearningRule] = None,
        batch_size: Optional[int] = 1,
    ) -> None:
        r"""
        Initializes network object.

        Args:
            neurons_type (NeuronType): The model of the neurons in the network
            learning_rule (LearningRule, optional): Applied learning rule to the network. Defaults to None.
            batch_size (int, optional): Size of each mini-batch.
        """
        super().__init__()

        self.neurons = neurons_type
        self.dt = self.neurons.dt
        self.learning_rule = learning_rule
        self.batch_size = batch_size

        self.weight = Parameter(torch.Tensor(0, 0), requires_grad=False)
        self.adjacency = Parameter(torch.Tensor(0, 0), requires_grad=False)
        self.register_buffer(
            "voltage", self.neurons.v_rest * torch.ones(self.batch_size, 0)
        )
        self.register_buffer(
            "spikes", torch.zeros(self.batch_size, 0, dtype=torch.bool)
        )
        self.register_buffer("refractory", torch.zeros(self.batch_size, 0))
        if self.learning_rule is not None:
            self.register_buffer("eligibility", torch.zeros_like(self.spikes.float()))

    def group(self, N: int, name: Optional[str] = None) -> None:
        r"""
        Add a neuron group to the network

        Args:
            N (int): Size of the group (number of neurons)
            name(str, optional): If not None, set the neuron group to a class attribute with this name.

        Returns:
            NeuronGroup instance.
        """
        neuron_group = NeuronGroup(N, slice(len(self.weight), len(self.weight) + N))
        if name is not None:
            assert not hasattr(self, name), f"Attribute name {name} already exists."
            setattr(self, name, neuron_group)

        self.weight.data = torch.nn.functional.pad(self.weight, (0, N, 0, N))
        self.adjacency.data = torch.nn.functional.pad(self.adjacency, (0, N, 0, N))
        self.spikes.data = torch.nn.functional.pad(self.spikes, (0, N))
        self.voltage.data = torch.nn.functional.pad(self.voltage, (0, N))
        self.refractory.data = torch.nn.functional.pad(self.refractory, (0, N))
        if self.learning_rule is not None:
            self.eligibility.data = torch.nn.functional.pad(self.eligibility, (0, N))

        return neuron_group

    def connect(
        self,
        source: NeuronGroup,
        target: NeuronGroup,
        weight: torch.Tensor,
        adjacency: torch.Tensor = None,
    ) -> None:
        """
        Add a connection between two neuron groups in the network

        Args:
            source (NeuronGroup): Source neuron group
            target (NeuronGroup): Target neuron group
            weight (torch.Tensor): Weights matrix as a tensor of shape: [source group neurons, target group neurons].
            adjacency (torch.Tensor, optional): Adjacency matrix as a tensor of shape like weight. If None all neurons
            in the connection will be considered connected. Defaults to None.
        """
        assert weight.shape == torch.Size(
            [source.n, target.n]
        ), f"Weight must be of shape {[source.n, target.n]}. got {weight.shape} instead."
        self.weight[source.idx, target.idx] = weight

        if adjacency is not None:
            assert adjacency.shape == torch.Size(
                [source.n, target.n]
            ), f"Adjacency must be of shape {[source.n, target.n]}. got {weight.shape} instead."
            self.adjacency[source.idx, target.idx] = adjacency
        else:
            self.adjacency[source.idx, target.idx] = torch.ones(
                source.n, target.n, dtype=torch.bool
            )

    def forward(
        self,
        input_spikes: torch.Tensor = None,
        input_voltages: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the network simulation for one timestep

        Args:
            input_spikes (torch.Tensor, optional): External spikes injected to the network. Size must be equal to total
            number of neurons. Defaults to None.
            input_voltages (torch.Tensor, optional): External voltagesinjected to the network. Size must be equal to
            total number of neurons. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of spikes and voltages tensors
        """
        # Decay voltages.
        if self.neurons.voltage_decay_factor is not None:
            self.voltage = (
                self.neurons.voltage_decay_factor * (self.voltage - self.neurons.v_rest)
                + self.neurons.v_rest
            )
        # External voltage:
        if input_voltages is not None:
            input_voltages.masked_fill_(self.refractory > 0, 0.0)
            self.voltage += input_voltages
        # Evoke spikes
        self.spikes = self.voltage >= self.neurons.v_thresh
        # External spikes
        if input_spikes is not None:
            self.spikes.logical_or_(input_spikes.bool())
        # update voltages
        self.voltage += self.spikes.float() @ self.weight  # + self.bias
        self.voltage.masked_fill_(
            self.refractory > 0, self.neurons.v_rest
        )  # reset the voltage of the neurons in the refractory period
        self.voltage.masked_fill_(
            self.spikes, self.neurons.v_reset
        )  # change the voltage of spiked neurons to v_reset
        # Update refractory timepoints
        self.refractory -= self.dt
        self.refractory.masked_fill_(self.spikes, self.neurons.refractory_period)
        # Learning process
        if self.training and self.learning_rule is not None:
            # Apply the learning rule and update weights
            self.learning_rule(self)

        return self.spikes, self.voltage

    def reset(self):
        self.voltage.zero_()
        self.spikes.zero_()
        self.refractory.zero_()
        self.eligibility.zero_()
