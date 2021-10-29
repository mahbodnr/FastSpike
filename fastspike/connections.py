from typing import Union, Tuple, Optional, Sequence

import torch
from torch.nn.modules.utils import _pair


def RandomConnection(
    n_source: int,
    n_target: int,
    connection_probability: float,
    w_min: Optional[float] = 0.0,
    w_max: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Connect every two neurons in a connection with a determined probability and a random weight.

    Args:
        n_source (int): Number of neurons in the source group.
        n_target (int): Number of neurons in the source group.
        connection_probability (float): The probability of each two neurons geting connected.
        w_min (float, optional): minimum weight value. Defaults to 0.
        w_max (float, optional): maximum weight value. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of weight and adjacency tensors
    """
    adjacency = (
        (torch.rand(n_source, n_target) + connection_probability - 0.5).round().bool()
    )
    weight = (torch.rand(n_source, n_target) * (w_max - w_min) + w_min).masked_fill_(
        adjacency.logical_not(), 0
    )
    return weight, adjacency


def LocallyConnected(
    input_shape: Sequence[int],
    n_channels: int,
    filter_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    w_min: Optional[float] = 0.0,
    w_max: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    r"""
    Connect two layers of the network with a Locally Connected scheme.

    Args:
        input_shape (Sequence[int]): source layer shape. A list containing: [#input channels, input height, input width]
        filter_size (Union[int, Tuple[int, int]]): Size of the kernel.
        stride (Union[int, Tuple[int, int]]): Value of stride in each dimension.
        w_min (float, optional): minimum weight value. Defaults to 0.
        w_max (float, optional): maximum weight value. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of weight and adjacency tensors
    """
    filter_size = _pair(filter_size)
    stride = _pair(stride)

    output_layer_height = (input_shape[-2] - filter_size[0]) // stride[0] + 1
    output_layer_width = (input_shape[-1] - filter_size[1]) // stride[1] + 1

    adjacency = torch.zeros(
        torch.prod(torch.tensor(input_shape)),
        n_channels * output_layer_height * output_layer_width,
    )

    for output_neuron_channel in range(n_channels):
        for output_neuron_height in range(output_layer_height):
            for output_neuron_width in range(output_layer_width):
                output_neuron_idx = (
                    output_neuron_channel * (output_layer_height * output_layer_width)
                    + output_neuron_height * output_layer_width
                    + output_neuron_width
                )

                first_neuron_in_RF_idx = (
                    output_neuron_height * stride[0]
                ) * input_shape[-1] + output_neuron_width * stride[1]
                for input_neuron_channel in range(input_shape[-3]):
                    for k1 in range(filter_size[0]):
                        for k2 in range(filter_size[1]):
                            input_neuron_idx = first_neuron_in_RF_idx + (
                                input_neuron_channel * input_shape[-1] * input_shape[-2]
                                + k1 * input_shape[-1]
                                + k2
                            )

                            adjacency[input_neuron_idx, output_neuron_idx] = 1

    weight = (
        torch.rand(
            torch.prod(torch.tensor(input_shape)),
            n_channels * output_layer_height * output_layer_width,
        )
        * (w_max - w_min)
        + w_min
    ).masked_fill_(adjacency.logical_not(), 0)

    return weight, adjacency


def FullyConnected(
    n_source: int,
    n_target: int,
    weights_value: Optional[float] = 1.0,
):
    r"""
    Connects every neurons in a connection with a fixed weight value.

    Args:
        n_source (int): Number of neurons in the source group.
        n_target (int): Number of neurons in the source group.
        weights_value (float): Value of the weights.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of weight and adjacency tensors
    """
    adjacency = torch.ones(n_source, n_target)
    weight = adjacency * weights_value
    return weight, adjacency


def UniformFullyConnected(
    n_source: int,
    n_target: int,
    w_min: Optional[float] = 0.0,
    w_max: Optional[float] = 1.0,
):
    r"""
    Connects every neurons in a connection with a random weight from a Uniform Distribution.

    Args:
        n_source (int): Number of neurons in the source group.
        n_target (int): Number of neurons in the source group.
        w_min (float, optional): minimum weight value. Defaults to 0.
        w_max (float, optional): maximum weight value. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of weight and adjacency tensors
    """
    adjacency = torch.ones(n_source, n_target)
    weight = (torch.rand(n_source, n_target) * (w_max - w_min) + w_min).masked_fill_(
        adjacency.logical_not(), 0
    )
    return weight, adjacency


def NormalFullyConnected(
    n_source: int,
    n_target: int,
    mean: Optional[float] = 0.0,
    std: Optional[float] = 1.0,
):
    r"""
    Connects every neurons in a connection with a random weight from a Normal Distribution.

    Args:
        n_source (int): Number of neurons in the source group.
        n_target (int): Number of neurons in the source group.
        mean: The Mean value of the normal distribution
        std: The Standard Deviation of the normal distribution

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of weight and adjacency tensors
    """
    adjacency = torch.ones(n_source, n_target)
    weight = (torch.normal(mean, std, size=(n_source, n_target))).masked_fill_(
        adjacency.logical_not(), 0
    )
    return weight, adjacency
