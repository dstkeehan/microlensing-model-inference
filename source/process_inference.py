"""Proccess a stored posterior object."""


import autocorrelation as acf
import plotting as pltf
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import MulensModel as mm

class Data:
    def __init__(self, flux, time, err_flux):
        self.flux = flux
        self.time = time
        self.err_flux = err_flux

if __name__ == "__main__":

    # Result file to process.
    n_suite = 1
    names = ['ambiguous/ambiguous', 'binary/binary']
    name = names[n_suite]
    file = open('results/'+name+'_stored_run.mcmc', 'rb') 
    object = pickle.load(file)
    joint_model_chain, MAPests, binary_states, single_states, binary_sp_states, single_sp_states, warm_up_iterations, symbols, event_params, data, name, dpi = object
    dpi=100

    #binary_states[5, :] = np.log10(binary_states[5, :])
    #binary_sp_states[5, :] = np.log10(binary_sp_states[5, :])

    # MAP estimates.
    print(f'single MAP {MAPests[0].truth}')
    print(f'binary MAP {MAPests[1].truth}')
    binary_theta = MAPests[1]
    
    single_theta = MAPests[0]
    curves = deepcopy([single_theta, binary_theta, data])

    # PLotting info.
    #ranges = [[0.45 -0.005, 0.55 +0.005], [14.9, 15.1 +0.15], [0.08 -0.005, 0.135], [9.5, 10.5], [-5, -1.5], [-1.0, 1.0], [0, 360]] # Ambiguous.
    ranges =[[0.45 -0.005, 0.55 +0.005], [14.9, 15.15], [0.085, 0.125], [9.5, 10.5], [-3, -0.4], [-1.0, 1.0], [0, 360]] # Binary.

    symbols = [r'$f_s$', r'$t_0$', r'$u_0$', r'$t_E$', r'$log_{10}(q)$', r'$s$', r'$\alpha$']

    # Joint posterior.
    pltf.broccoli(joint_model_chain, binary_states[1:, :], single_states[1:, :], binary_sp_states[1:, :], single_sp_states[1:, :], symbols[1:], ranges[1:], curves, event_params, name, dpi)
    
    '''
    # Trace of model index.
    plt.plot(np.linspace(0, joint_model_chain.n, joint_model_chain.n), joint_model_chain.model_indices, linewidth = 0.25, color = 'purple')
    plt.xlabel('Samples')
    plt.ylabel(r'$m_i$')
    plt.locator_params(axis = "y", nbins = 2) # only two ticks
    plt.savefig('results/'+name+'-mtrace.png', bbox_inches = 'tight', dpi = dpi, transparent=True)
    plt.clf()
    '''

    # Marginal probabilities.
    for i in range(2):
        print(f"P(m{i}|y): {joint_model_chain.model_indices.count(i) / joint_model_chain.n} +- {np.std(np.array(joint_model_chain.model_indices))/(joint_model_chain.n**0.5):.6f}")

    # Chi2 values.
    model = mm.Model(dict(zip(["t_0", "u_0", "t_E", "q", "s", "alpha"], binary_theta.truth[1:])))
    model.set_magnification_methods([0., "point_source", 72.])
    a = model.magnification(data.time) # The proposed magnification signal.
    y = data.flux # The observed flux signal.
    F = (a-1)*binary_theta.truth[0]+1
    sd = data.err_flux
    print(f'binary chi2 {np.sum(((y - F)/sd)**2):.4f}')

    model = mm.Model(dict(zip(["t_0", "u_0", "t_E"], single_theta.truth[1:])))
    model.set_magnification_methods([0., "point_source", 72.])
    a = model.magnification(data.time) # The proposed magnification signal.
    y = data.flux # The observed flux signal.
    F = (a-1)*single_theta.truth[0]+1
    sd = data.err_flux
    print(f'single chi2 {np.sum(((y - F)/sd)**2):.4f}')

    model = mm.Model(dict(zip(["t_0", "u_0", "t_E", "q", "s", "alpha"], event_params.truth[1:])))
    model.set_magnification_methods([0., "point_source", 72.])
    a = model.magnification(data.time) # The proposed magnification signal.
    y = data.flux # The observed flux signal.
    F = (a-1)*event_params.truth[0]+1
    sd = data.err_flux
    print(f'true chi2 {np.sum(((y - F)/sd)**2):.4f}')