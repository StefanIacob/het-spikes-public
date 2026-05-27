import numpy as np


class LIF_DDN_simple(object):
    def __init__(self, decoder_weights, weights_fast, weights_slow, input_weights_x, input_weights_dx, dt, tau, threshold,
                 coordinates, buffersize, propagation_vel=30, spike_scaling=False, bias_compensation=False):
        self.W_f = weights_fast
        self.W_s = weights_slow        
        self.B = buffersize
        assert self.W_f.shape[0] == self.W_f.shape[1]
        if not (self.W_s is None):
            assert self.W_s.shape[0] == self.W_s.shape[1] == self.W_f.shape[0]     

        self.N = self.W_f.shape[0]
        self.V = np.zeros(shape=(self.N))
        self.S = np.zeros(shape=(self.N, buffersize))
        self.I = np.zeros(shape=(self.N,))
        self.T = threshold
        self.dt = dt
        self.propagation_vel = propagation_vel
        self.tau = tau
        self.A = np.zeros_like(self.S)
        self.bias_compensation = bias_compensation


        # Location-related params
        self.x_range = (np.min(coordinates[:, 0]), np.max(coordinates[:, 0]))
        self.y_range = (np.min(coordinates[:, 1]), np.max(coordinates[:, 1]))
        self.coordinates = coordinates
        self.decoder = SCN_decoder(weights=decoder_weights, thresholds=self.T, timescale=self.tau, dt=self.dt, spike_scaling=spike_scaling, bias_compensation=bias_compensation)
        self.encoder = SCN_input_encoder(weights_x=input_weights_x, weights_dx=input_weights_dx, dt=self.dt)
        self.active_neurons = np.ones_like(self.V)
        
        
    def reset_network(self):
        """
        Resets network activity to initial state.
        :return: None
        """
        self.V = np.zeros(shape=(self.N,))

    def get_total_weights(self):
        W_total = np.copy(self.W_f) 
        if not (self.W_s is None):
            W_total += np.copy(self.W_s)
        return W_total
    
    def add_timescale_noise(self, noise_scale):
        self.decoder.add_timescale_noise(noise_scale)

    def change_weights(self, encoder, recurrent, threshold):
        self.W = recurrent
        self.encoder.F = encoder
        self.T = threshold

    def change_tau(self, new_tau):
        self.tau = new_tau
        self.decoder.tau = new_tau
        self.encoder.tau = new_tau

    def reset_activity(self):
        self.V = np.ones(shape=(self.N,)) * self.v_rest

    def slow_plasticity(self, I_external, lr=.01):
        inp = self.encoder.F_x @ I_external
        E_rec = inp + self.W_f @ self.decoder.rates
        E_err = E_rec - self.V
        print(E_err[10])
        dW_s = self.W_f @ E_err.T * lr
        self.W_s += dW_s.T

    def simple_syn(self):
        s_vec = self.S[:, 0]  # current S
        self.I *= 0 # reset
        self.I += self.W_f @ s_vec
        if self.W_s is not None:
            r_vec = self.decoder.rates # current r
            self.I += self.W_s @ r_vec * self.dt
        

    def compute_new_V(self, I):
        dV = -(self.dt/self.tau) * self.V + I
        new_V = self.V + dV
        self.V = new_V

    def compute_spikes(self, spike_scaling_factor=1):
        arg_max_V = np.argmax(self.V)
        new_S = np.zeros(shape=(self.N,))
        new_S[arg_max_V] = int(self.V[arg_max_V] > self.T[arg_max_V])
        
        # roll buffer
        self.S[:, 1:] = self.S[:, :-1]
        self.S[:, 0] = new_S * self.active_neurons * spike_scaling_factor


    def update_step(self, I_external):
        self.simple_syn()        
        I_ex = self.encoder.get_network_input(I_external)        
        inputs = self.I + I_ex
        self.compute_new_V(inputs)
        # self.slow_plasticity(I_external)

        self.compute_spikes(self.spike_factor)
        self.A[:, 0] = (self.V)/(self.T) # express activity as portion of the threshold?

        # Output
        self.decoder.update(self.S[:, 0])
        if self.bias_compensation:
            return self.decoder.scaled_estimate
        else:
            return self.decoder.estimate

    def knock_out(self, ko_indices):
        for i in ko_indices:
            self.active_neurons[i] = 0

    def reset_ko(self):
        self.active_neurons = np.ones_like(self.V)


class SCN_input_encoder(object):

    def __init__(self, weights_x, weights_dx, dt, input_buffer_size=2):
        self.F_x = weights_x
        self.F_dx = weights_dx
        self.N, self.M = self.F_x.shape
        
        self.dt = dt
        self.input_buffer = np.zeros(shape=(self.M, input_buffer_size))
        self.buffersize = input_buffer_size
        self.feedback = 0

   
    def _roll_buffer(self, I_external):
        self.input_buffer[:, 1:] = self.input_buffer[:, :-1]
        self.input_buffer[:, 0] = I_external
    
    def get_network_input(self, x):
        self._roll_buffer(x)
        d_x = (self.input_buffer[:, 0] - self.input_buffer[:, 1])/self.dt
        I_x = self.F_x @ x
        I_dx = self.F_dx @ d_x
        I_net = (I_x + I_dx) * self.dt
        return I_net
    

    def reset(self):
        self.input_buffer = np.zeros(shape=(self.M, self.buffersize))
        self.g_x = np.zeros(shape=(self.M,))

class SCN_decoder(object):

    def __init__(self, weights, timescale, dt, thresholds, spike_scaling=False, bias_compensation=False):
        self.D = weights
        N, M = self.D.shape
        self.M = M
        self.N = N
        self.tau = timescale
        self.rates = np.zeros((N,))
        self.estimate = np.zeros((M,))
        self.scaled_estimate = np.zeros_like(self.estimate)
        self.dt = dt
        self.spike_scaling = spike_scaling
        time_window_out_norm = 20 #int((np.min(self.tau)/2)/dt)
        self.out_norm_buffer = np.zeros(shape=(time_window_out_norm,))
        self.T = np.average(thresholds)
        self.bias_comp = bias_compensation

    def get_estimate(self):
        if self.bias_comp:
            return self.scaled_estimate
        return self.estimate

    def get_scaled_estimate(self):
        self.roll_out_norm_buffer()
        self.out_norm_buffer[0] = np.linalg.norm(self.estimate)
        scaled_estimate = self.bias_compensation()
        self.scaled_estimate = scaled_estimate
        return scaled_estimate

    def bias_compensation(self):
        average_norm = np.average(self.out_norm_buffer)
        scaled_D = self.D
        if average_norm > .0001:
            scaling = (average_norm + self.T - 0.5) / average_norm
            scaled_D = (self.D * scaling)
        return self.rates @ scaled_D

    def roll_out_norm_buffer(self):
        self.out_norm_buffer[1:] = self.out_norm_buffer[:-1]

    def get_ratesHz(self):
        return self.rates *(1/self.tau)

    def update(self, spikes):
        self.rates = self._instant_rate(spikes)
        self.estimate = self.rates @ self.D
        self.scaled_estimate = self.get_scaled_estimate()

    def _instant_rate(self, spikes):
        assert spikes.shape == self.rates.shape
        # assume spikes = 1
        old_rates = self.rates
        if self.spike_scaling:
            d_rate = -(self.dt/self.tau) * old_rates + (spikes / self.tau)
        else:
            d_rate = -(self.dt/self.tau) * old_rates + spikes

        new_rates = old_rates + d_rate
        return new_rates

    def add_timescale_noise(self, noise_scale):
        dec_shape = self.tau.shape
        tau_noise = np.random.normal(loc=0, scale=noise_scale, size=dec_shape)
        self.tau += tau_noise

    def reset(self):
        self.rates = np.zeros((self.N,))
        self.estimate = np.zeros((self.M,))

# Static functions
def coordinates2distance(coordinates):
    """
    Transforms a spatial configuration of neurons to a distance (adjacency) matrix.
    :param coordinates: ndarray
        N by dims array with N the number of neurons and dims the number of spatial dimensions. Should contain the
        spatial coordinates in a 2D space of each neuron.
    :return: ndarray
        N by N array containing the spatial distance between each neuron.
    """
    N = coordinates.shape[0]

    D = np.zeros((N, N))

    def dist(dist_x, dist_y):
        return np.sqrt(dist_x ** 2 + dist_y ** 2)

    for i in range(N):
        for j in range(N):
            if not i == j:
                dist_x = np.abs(coordinates[i, 0] - coordinates[j, 0])
                dist_y = np.abs(coordinates[i, 1] - coordinates[j, 1])
                d = dist(dist_x, dist_y)
                D[i, j] = d
    return D

