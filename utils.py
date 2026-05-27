import numpy as np
from simulator import scnSim


def mse(target_signal, input_signal):
    """
    rmse(input_signal, target_signal)-> error
    MSE calculation.
    Calculates the mean square error (MSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    # target_signal = target_signal.view(target_signal.numel())
    # input_signal = input_signal.view(input_signal.numel())

    error = (target_signal - input_signal) ** 2
    return error.mean()


def nmse(target_signal, input_signal):
    """
    nmse(input_signal, target_signal)-> error
    NMSE calculation.
    Calculates the normalized mean square error (NMSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    # check_signal_dimensions(input_signal, target_signal)

    if len(target_signal) == 1:
        raise NotImplementedError('The NRMSE is not defined for signals of length 1 since they have no variance.')

    # Use normalization with N-1, as in matlab
    var = np.std(target_signal) ** 2

    return mse(target_signal, input_signal) / var


def nrmse(input_signal, target_signal):
    """
    nrmse(input_signal, target_signal)-> error
    NRMSE calculation.
    Calculates the normalized root mean square error (NRMSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    # check_signal_dimensions(input_signal, target_signal)

    if len(target_signal) == 1:
        raise NotImplementedError('The NRMSE is not defined for signals of length 1 since they have no variance.')

    return np.sqrt(nmse(target_signal, input_signal))


def spectral_radius_norm(W, wanted_sr):
    v = np.linalg.eigvals(W)
    sr = np.max(np.absolute(v))
    W_scaled = (W * wanted_sr) / sr
    return W_scaled

def evaluate_SCN_autoencoder(network, input_signal, dt, collect_V=False):
    network.reset_SCN()
    T_steps, dims = input_signal.shape
    sim_time = T_steps * dt
    N_neurons = network.N
    spike_cost = network.spike_cost
    # Simulate
    sim_net = scnSim(network=network)
    if collect_V:
        output, spike_times, rates, Vs = sim_net.get_spike_data(input_signal, collect_V=collect_V)
    else:
        output, spike_times, rates = sim_net.get_spike_data(input_signal, collect_V=collect_V)
    # Network activity
    N_spikes = spike_times.shape[0]
    
    # net_act = N_spikes / (N_neurons * sim_time)
    net_act = N_spikes / sim_time

    # Compute error
    MSE = mse(input_signal, output)

    # Compute error as used for the original network derivation
    prediction_E = np.array(input_signal - output)
    prediction_SE = np.array([E@E.T for E in prediction_E])
    cost_SE = np.array([(spike_cost * r)@r for r in rates])
    # cost_SE = spike_cost * rate_E
    network_SE = prediction_SE + cost_SE
    network_MSE = np.mean(network_SE)

    # Compute MSE of a silent network
    rates_0 = np.zeros_like(network.decoder.rates)
    estimate_0 = rates_0 @ network.decoder.D
    output_0 = np.repeat(np.expand_dims(estimate_0, axis=0), T_steps, axis=0)
    MSE_0 = mse(input_signal, output_0)

    # Normalized MSE
    MSE_norm = MSE/MSE_0
    
    if collect_V:
        return MSE, MSE_norm, network_MSE, output, spike_times, rates, Vs

    return MSE, MSE_norm, network_MSE, output, spike_times, rates


def act_dimensionality(network_activity, variance_threshold=.95):
    _, s, _ = np.linalg.svd(network_activity)
    dim = 0
    var_explained = 0
    while var_explained < variance_threshold:
        var_explained = np.sum(s[:dim + 1]) / np.sum(s)
        dim += 1
    return dim


    