from simulator import scnSim
import numpy as np
from populations import SCN_autoencoder
import matplotlib.pyplot as plt
from reservoirpy.datasets import lorenz
from utils import evaluate_SCN_autoencoder

if __name__ == '__main__':

    np.random.seed(42)

    N = 200 # Net size
    M = 2 # In size
    B = 15 # Buffersize
    # tau = .01  # membrane time constant (s)
    # tau = np.random.normal(0.2, .05, size=(N,))
    heterogeneity = 0# .09
    mean_tau = .05
    tau = np.random.uniform(mean_tau - (heterogeneity/2), mean_tau + (heterogeneity/2), size=(N,))

    dt = 0.001
    dec_scale = 1
    D = np.random.normal(loc=0.0, scale=1, size=(N, M))
    D = D / np.linalg.norm(D, axis=1, keepdims=True) * dec_scale
    spike_cost =  np.ones(shape=(N,)) * 0.1
    spike_scaling = False
    spike_scaling_factor = 1
    test_net = SCN_autoencoder(D, tau, spike_cost, dt, B, spike_scaling=spike_scaling, spike_scaling_factor=spike_scaling_factor, bias_compensation=True)
    sim_net = scnSim(network=test_net)
    t = np.arange(0, 40, 0.001)
    inputs = lorenz(10000, h=dt)[:, :M]
    sim_net.visualize_interactive_2D(rasterplot=False, inputs=inputs.T, spike_vis=True,probe_index=1)
    # sim_net.visualize_interactive_2D(rasterplot=False, inputs=inputs.T, spike_vis=True,probe_index=1)
    # evaluate_SCN_autoencoder(test_net, inputs, dt)