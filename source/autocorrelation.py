"""Autocorrelation tools for convergence analysis.

See the detailed tutorial from:
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import emcee as mc
import plotting as pltf


def attempt_truncation(joint_model_chain):
    """Truncate a burn in period
        
    Uses autocorrelation time heuristic to remove intial period below 50
    autocorrelation times, for a joint model space.

    Args:
        joint_model_chain: [chain] Collection of model parameter states.

    Returns:
        truncated: [int] Number of truncated states.
    """
    
    # Points to compute act.
    n_autocorrelation_times = 10
    N = np.exp(np.linspace(np.log(int(joint_model_chain.length/n_autocorrelation_times)), np.log(joint_model_chain.length), n_autocorrelation_times)).astype(int)

    autocorrelation_time_model_indices = np.zeros(len(N))
    model_indices_signal = np.array(joint_model_chain.model_indices)

    for i, n in enumerate(N):
        autocorrelation_time_model_indices[i] = mc.autocorr.integrated_time(model_indices_signal[:n], c = 5, tol = 5, quiet = True)
        
        if i>0:
            if N[i-1] - N[i-1]/50 < autocorrelation_time_model_indices[i] < N[i-1] + N[i-1]/50: # Integrated autocorrelation time stabilises.
                truncated = N[i]

                # Remove stored states and update count.
                joint_model_chain.model_indices = joint_model_chain.model_indices[truncated:]
                joint_model_chain.states = joint_model_chain.states[truncated:]
                joint_model_chain.length = joint_model_chain.length - truncated

                return truncated

    print("Integrated autocorrelation time did not converge.")
    return 0


def plot_act(ax, joint_model_chain):
    """Plot parameter autocorrelation times.
        
    Args:
        joint_model_chain: [chain] Collection of states from any model.
        symbols: [list] Variable name strings.
        name: [optional, string] File ouptut name.
        dpi: [optional, int] File output dpi.
    """

    # Points to compute integrated autocorrelation time.
    n_autocorrelation_times = 10
    N = np.exp(np.linspace(np.log(int(joint_model_chain.length/n_autocorrelation_times)), np.log(joint_model_chain.length), n_autocorrelation_times)).astype(int)

    # Loop through the model indices.
    autocorrelation_time_model_indices = np.zeros(len(N))
    model_indices_signal = np.array(joint_model_chain.model_indices)
    for i, n in enumerate(N):
        autocorrelation_time_model_indices[i] = mc.autocorr.integrated_time(model_indices_signal[:n], c = 5, tol = 5, quiet = True)
    ax.loglog(N, autocorrelation_time_model_indices, "o-", label=r"$m$",  linewidth = 2, markersize = 5, color='black')

    ax.set_xlabel("iterations [n]",  fontsize = 16)
    ax.set_ylabel("integrated\nautocorrelation time [n]", fontsize = 16)

    return


