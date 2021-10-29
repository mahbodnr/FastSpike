import torch
from tqdm import tqdm

from fastspike.network import Network
from fastspike.neurons import LIF
from fastspike.connections import LocallyConnected, FullyConnected
from fastspike.learning import STDP


class MyNetwork(Network):
    def __init__(
        self,
        neurons_type,
        learning_rule=None,
        batch_size=1,
    ) -> None:
        super().__init__(
            neurons_type=neurons_type,
            learning_rule=learning_rule,
            batch_size=batch_size,
        )

        self.inp = self.group(1 * 22 * 22)
        self.layer1 = self.group(10 * 20 * 20)

        self.connect(
            self.inp,
            self.layer1,
            *LocallyConnected(
                input_shape=[1, 22, 22],  # [n_channels, height, width]
                n_channels=10,
                filter_size=3,
                stride=1,
                w_max=1,
                w_min=0,
            )
        )


if __name__ == "__main__":
    time = 100
    batch_size = 3
    net = MyNetwork(
        neurons_type=LIF(dt=1),
        learning_rule=STDP(nu=0.01),
        batch_size=batch_size,
    )
    net.group(10, "layer2")
    net.connect(net.layer1, net.layer2, *FullyConnected(net.layer1.n, net.layer2.n, 1))
    print(net)
    input_spikes = torch.tensor([[1] * len(net.weight)] * batch_size)
    for i in tqdm(range(100)):
        for _ in range(time):
            net(input_spikes=input_spikes)
            net.reset()
