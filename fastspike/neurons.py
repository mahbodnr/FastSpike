from abc import ABC, abstractmethod
from typing import Union

import torch


class NeuronGroup(torch.nn.Module):
    def __init__(self, n, idx):
        super().__init__()
        self.n = n
        self.idx = idx


class NeuronType(ABC, torch.nn.Module):
    r"""
    Abstract neuron type class.
    """

    def __init__(
        self,
        dt: int,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        traces_additive: bool = False,
    ) -> None:
        """
        Initializes neuron type object.

        Args:
            dt (int): Timestep. Indicates amount of time passed in each step of simulation.
            tc_trace (Union[float, torch.Tensor], optional): Neurons trace time constant in ms. used to update
            eligibility trace every timepoint .Defaults to 20.0.
            trace_scale (Union[float, torch.Tensor], optional): If traces_additive is True, this value will be added to
            neurons' eligibility trace after each spike. If traces_additive is False, neurons' eligibility trace will be
            set to this this value. Defaults to 1.0.
            traces_additive (bool, optional): Whether to add trace_scale to neurons' entitlement trace after spikes
            occur or set them to the trace_scale value. Defaults to False.
        """
        super().__init__()
        self.dt = dt

        self.traces_additive = traces_additive

        self.register_buffer("voltage_decay_factor", None)
        self.register_buffer("tc_trace", torch.tensor(tc_trace))
        self.register_buffer("trace_scale", torch.tensor(trace_scale))

    def update(self, spikes: torch.Tensor) -> None:
        """
        Update features of neuron group based on the network's activity.

        Args:
            spikes (torch.Tensor): Spikes evoked in this timepoint
        """
        # Decay and set spike traces.
        self.eligibility *= self.trace_decay

        if self.traces_additive:
            self.eligibility += self.trace_scale * spikes
        else:
            self.eligibility.masked_fill_(spikes, self.trace_scale)

    def compute_decay_factors(self) -> None:
        r"""
        Compute decay factors based on time constants.
        """
        self.trace_decay = torch.exp(
            -self.dt / self.tc_trace
        )  # Spike trace decay (per timestep).


class LIF(NeuronType):
    r"""
    leaky integrate-and-fire (LIF) neurons
    """

    def __init__(
        self,
        dt: int,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        traces_additive: bool = False,
        v_thresh: Union[float, torch.Tensor] = -52.0,
        v_rest: Union[float, torch.Tensor] = -65.0,
        v_reset: Union[float, torch.Tensor] = -65.0,
        refractory_period: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
    ) -> None:
        r"""
        Initializes LIF neurons object.

        Args:
            dt (int): Timestep. Indicates amount of time passed in each step of simulation.
            tc_trace (Union[float, torch.Tensor], optional): Neurons trace time constant in ms. used to update
            eligibility trace every timepoint .Defaults to 20.0.
            trace_scale (Union[float, torch.Tensor], optional): If traces_additive is True, this value will be added to
            neurons' eligibility trace after each spike. If traces_additive is False, neurons' eligibility trace will be
            set to this this value. Defaults to 1.0.
            traces_additive (bool, optional): Whether to add trace_scale to neurons' entitlement trace after spikes
            occur or set them to the trace_scale value. Defaults to False.
            v_thresh (Union[float, torch.Tensor], optional): Neurons' threshold voltage in mV. Defaults to -52.0.
            v_rest (Union[float, torch.Tensor], optional): Neurons' rest voltage in mV. Defaults to -65.0.
            v_reset (Union[float, torch.Tensor], optional): Neurons' reset voltages in mV. Defaults to -65.0.
            refractory_period (Union[int, torch.Tensor], optional): Neurons' refractory period in ms . Defaults to 5.
            tc_decay (Union[float, torch.Tensor], optional): Voltage decay time constant in ms. Defaults to 100.0.
        """

        super().__init__(
            dt=dt,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
        )
        self.register_buffer(
            "v_rest", torch.tensor(v_rest, dtype=torch.float)
        )  # Rest voltage.
        self.register_buffer(
            "v_reset", torch.tensor(v_reset, dtype=torch.float)
        )  # Post-spike reset voltage.
        self.register_buffer(
            "v_thresh", torch.tensor(v_thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer(
            "refractory_period", torch.tensor(refractory_period, dtype=torch.int32)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay, dtype=torch.float)
        )  # Time constant of neuron voltage decay.

        self.compute_decay_factors()

    def init_group_params(self, batch_size):
        r"""
        Initializes LIF neuron group parameters.
        """
        self.register_buffer(
            "refractory", torch.zeros(batch_size, 0)
        )  # Refractory period counter for each neuron.

    def compute_decay_factors(self) -> None:
        r"""
        Compute decay factors based on time constants.
        """
        super().compute_decay_factors()
        self.register_buffer(
            "voltage_decay_factor",
            torch.exp(-self.dt / self.tc_decay),
        )  # Neuron voltage decay (per timestep).

    def _add_group(self, N: int) -> None:
        r"""
        Add new neurons to the group

        Args:
            N (int): Number of neurons
        """
        self.refractory.data = torch.nn.functional.pad(self.refractory, (0, N))

    def reset(self):
        self.refractory.zero_()

    def forward(self, weight, spikes, voltage, input_spikes, input_voltages) -> None:
        r"""
        Update voltage and spikes of LIF neurons.
        """
        # Decay voltages.
        if self.voltage_decay_factor is not None:
            new_voltage = (
                self.voltage_decay_factor * (voltage - self.v_rest) + self.v_rest
            )
        # External voltage:
        if input_voltages is not None:
            input_voltages.masked_fill_(self.refractory > 0, 0.0)
            new_voltage += input_voltages
        # Evoke spikes
        new_spikes = new_voltage >= self.v_thresh
        # External spikes
        if input_spikes is not None:
            new_spikes.logical_or_(input_spikes.bool())
        # update voltages
        new_voltage += new_spikes.float() @ weight  # + self.bias
        new_voltage.masked_fill_(
            self.refractory > 0, self.v_rest
        )  # reset the voltage of the neurons in the refractory period
        new_voltage.masked_fill_(
            new_spikes, self.v_reset
        )  # change the voltage of spiked neurons to v_reset
        # Update refractory timepoints
        self.refractory -= self.dt
        self.refractory.masked_fill_(new_spikes, self.refractory_period)

        return new_spikes, new_voltage
