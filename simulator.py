from gui import DistDelayGUI
import numpy as np
import matplotlib.pyplot as plt

class scnSim(object):
    def __init__(self, network):
        self.network = network

    def visualize_interactive_2D(self, inputs=None, probe_index=None, rasterplot=False, spike_vis=False):
        assert inputs.shape[0] == 2
        assert self.network.M == 2
        in_size = 2
        gui = DistDelayGUI(self.network, inputs=inputs, probe_index=probe_index, use_rasterplot=rasterplot,
                           in_size=in_size, spike_vis=spike_vis)
        gui.run()

    def get_spike_data(self, inputs, collect_V=False):
        spike_times = []
        rates = []
        output = []
        if collect_V:
            Vs = []
        for t, inp in enumerate(inputs):
            estimate = self.network.update_step(inp)
            output.append(estimate)
            spikes = self.network.S[:, 0]
            s_ind = np.argwhere(spikes > 0).flatten()
            for i in s_ind:
                spike_times.append([t * self.network.dt, i])
            current_rates = self.network.decoder.rates
            rates.append(current_rates)
            if collect_V:
                Vs.append(self.network.V)
        output = np.array(output)
        rates = np.array(rates)
        spike_times = np.array(spike_times)
        if collect_V:
            Vs = np.array(Vs)
            return output, spike_times, rates, Vs
        return output, spike_times, rates
        # plt.scatter(spike_times[:, 0], spike_times[:, 1], marker='|')
        # plt.show()

class scnControlSim(object):

    def __init__(self, network, plant):
        self.network = network
        self.plant = plant

    def visualize_interactive(self, inputs=None, probe_index=None, rasterplot=False, spike_vis=False):
        assert inputs.shape[0] == 2
        in_size = 2
        gui = DistDelayGUI(self.network, plant=self.plant, inputs=inputs, probe_index=probe_index, use_rasterplot=rasterplot,
                           in_size=in_size, spike_vis=spike_vis)
        gui.run()

    def get_spike_data(self, inputs):
        #TODO: add plant
        spike_times = []
        for t, inp in enumerate(inputs.T):
            self.network.update_step(inp)
            spikes = self.network.S[:, 0]
            s_ind = np.argwhere(spikes == 1).flatten()
            for i in s_ind:
                spike_times.append([t, i])
        spike_times = np.array(spike_times)
        plt.scatter(spike_times[:, 0], spike_times[:, 1], marker='|')
        plt.show()

class LinearPlant(object):

    def __init__(self, A, B, dt, x_init=np.array([0, 0], dtype='float64')):
        self.x = x_init
        self.N = x_init.shape[0]
        assert A.shape == (self.N, self.N)
        assert B.shape[0] == self.N
        self.A = A
        self.B = B
        self.dt = dt

    def update(self, u):
        assert u.shape[0] == len(u) == self.B.shape[1]
        dx = self.A @ self.x + self.B @ u
        self.x += dx * self.dt

    def get_x(self):
        return np.copy(self.x)

    def get_states(self, u_sequence):
        x_hist = [self.x]
        for u in u_sequence.T:
            self.update(u)
            x_hist.append(self.get_x())
        return np.array(x_hist[0:]).T

