import numpy as np
from network import LIF_DDN_simple


class SCN_autoencoder(LIF_DDN_simple):

    def __init__(self, D, tau, spike_cost, dt, B, spike_scaling=False, spike_scaling_factor=1, bias_compensation=False):
        #TODO: assert tau is ndarray of size N
        # spike cost can be either scalar of array of size N
        N, M = D.shape
        if type(spike_cost) == np.array or type(spike_cost) == list:
            assert len(spike_cost) == N or len(spike_cost) == 1

            if len(spike_cost) == 1:
                spike_cost = np.ones(shape=(N,)) * spike_cost[0]
        else:
            spike_cost = np.ones(shape=(N,)) * spike_cost
        
        self.M = M
        self.D = D
        self.spike_cost = spike_cost
        F = D
        T = []
        for i, D_i in enumerate(D):
            T_i = (D_i.T @ D_i + spike_cost[i]) / 2
            T.append(T_i)
        self.spike_factor = spike_scaling_factor
        T = np.array(T)
        T *= self.spike_factor
        F_x = np.copy(F)
        F_dx = np.copy(F)
        neuron_coordinates = np.random.normal(size=(N, 2)) * .01
        Omega_slow = None

        timescales = 1/tau
        L = np.diag(timescales)
        D_scaled = L@D

        if spike_scaling:
            T = T/tau
            Omega_fast = -(D @ D_scaled.T)
            Omega_fast -= spike_cost * L # spike cost is also scaled with spike size
        else:    
            Omega_fast = -(D @ D.T)            
            Omega_fast -= spike_cost * np.identity(N)


        if type(tau) == np.ndarray:
            if len(tau) > 1:
                # Heterogeneous SCN: include the slow recurrent compensation
                
                Omega_slow = -(D@D_scaled.T - D_scaled@D.T).T

                # Also separate forward weights x and for dx/dt
                F_x = np.copy(D_scaled)
                F_dx = np.copy(F)

        super().__init__(decoder_weights=D, weights_fast=Omega_fast, weights_slow=Omega_slow, 
                         input_weights_x=F_x, input_weights_dx=F_dx, dt=dt, tau=tau, threshold=T, 
                         coordinates=neuron_coordinates, buffersize=B, spike_scaling=spike_scaling, bias_compensation=bias_compensation)



    def change_spike_cost(self, new_spike_cost):
        F, Omega, T = self._get_network_parameters(self.D, new_spike_cost)
        self.change_weights(F, Omega, T)
        self.spike_cost = new_spike_cost
        print("changed weights")

    def reset_SCN(self):
        self.reset_network()
        self.decoder.reset()
        self.encoder.reset()