import torch
from tqdm import tqdm

from FastSpike.Network import Network
from FastSpike.NeuronTypes import LIF
from FastSpike.Connections import LocallyConnected

class MyNetwork(Network):
    def __init__(
        self,
        neurons_type,
        learning_rule= None,
        batch_size= 1,
    ) -> None:
        super().__init__(
            neurons_type= neurons_type,
            learning_rule= learning_rule,
            batch_size= batch_size
        )
        self.inp = self.group(1*22*22)
        self.layer1 = self.group(10*20*20)

        self.structure()

    
    def structure(self) -> None:
        super().structure()
        self.connect(self.inp, self.layer1, *LocallyConnected(
            input_shape= [1,22,22], # [n_channels, height, width]
            n_channels= 10, 
            filter_size= 3, 
            stride= 1,
            w_max = 1,
            w_min = 0,
            )
        )


if __name__ == "__main__":
    time = 100
    net = MyNetwork(LIF(dt=1), batch_size= 3)
    print(net, net.weight, net.adjacency)
    input_spikes = torch.tensor([True]*484+[False]*4000)
    p, v = net(input_spikes=input_spikes)
    print(p,v, p.shape, v.shape)
    for i in tqdm(range(100)):
        for _ in (range(time)):
            net()