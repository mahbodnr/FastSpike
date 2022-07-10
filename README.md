# FastSpike

A Spiking Neural Network (SNN) framework for fast simulation based on PyTorch

FastSpike is designed to exploit the GPU memory in order to increase the speed of simulation as much as possible. Right now, FastSpike only supports homogeneous networks (same characteristics for all neurons in a network). For CPU computing, heterogeneous networks, and a more memory-friendly framework, use [BindsNET](https://github.com/BindsNET/bindsnet).
The amount of speed-up achieved by FastSpike depends on the GPU capacity. Hence, it is important to use suitable hardware for your problem.

# Benchmarking
Here is a benchmark of FastSpike against BindsNET. The benchmark is done on a Tesla P100 on Google Colab. In general, FastSpikes perform better when the number of network layers is high compared to the number of neurons. It is expected that FastSpike will achieve further speed-ups on more powerful processors, depending on GPU capacity. The benchmark code is available on: [Colab Notebook](https://colab.research.google.com/drive/11SKxlbLxc6ZzXXDJkYf59Wu9ckvFZh6K?usp=sharing)

## Fixed size, Different number of layers

![](docs/layerwise_gpu.png)

## Fixed number of layers, Different layer sizes

![](docs/sizewise_gpu.png)

# Julia

Also see [FastSpike.jl](https://github.com/mahbodnr/FastSpike.jl) for Julia implementation.
