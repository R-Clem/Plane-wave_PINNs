Physics-informed neural networks (PINNs) for solving a system of two coupled oscillators.

Both scripts allow for training over a normal distribution of randomly-sampled boundary
conditions in order to decouple the trained PINN from a unique solution. The conventional
PINN script is present to demonstrate that this tends not to work very well for conventional
activation functions such as tanh. The plane-wave PINN script creates and trains a PINN that
learns a general solution that is only made unique during the evaluation of the network, 
when a set of boundary conditions and time are passed to it.

The code in this repository was used to generate the results for the paper
"Plane-Wave Decomposition and Randomised Training; a Novel Path to Generalised Physics Informed Neural Networks for Simple Harmonic Motion"
and is available on arXiv: arXiv:2504.00249v3
