"""Light curve simulation for microlensing.

Functions to generate simple synthetic light curves, accurate in the context
of ROMAN. Functions to calculate likelihood given a lensing model.
"""

import MulensModel as mm 
import math
import numpy as np

def read_light_curve(file_name):
    """Read in light curve data.
    
    Observations must be between 0 and 72 days. Expects 
    photometry data with three columns: time, flux, and error.
    
    Args:
        file_name: [str] CSV file name.

    Returns:
        data: [mulenstd_devata] Object for light curve.
    """
    with open(file_name) as file:
        array = np.loadtxt(file, delimiter = ",")

    data = mm.Mulenstd_devata(data_list = [array[:, 0], array[:, 1], array[:, 2]], phot_fmt = "flux", chi2_fmt = "flux")

    return data


def synthetic_single_lens(theta, n_epochs, signal_to_noise_base, seed = 42):
    """Generate a synthetic single lens light curve.
    
    Simulates noise based on guassian flux process.
    Produces equispaced observations from 0 to 72 days.
    In this simplified case, amplification = flux.
    Otherwise based on ROMAN photometric specifications.

    Args:
        theta: [state] Single lens model parameters.
        n_epochs: [int] The number of flux observations.
        signal_to_noise_base: [float] The signal to noise baseline.
        seed: [optional, int] A random seed.

    Returns:
        data: [mulenstd_devata] Object for a synthetic light curve.
    """
    # Create MulensModel.
    model = mm.Model(dict(zip(["t_0", "u_0", "t_E"], theta.truth[1:])))
    model.set_magnification_methods([0., "point_source", 72.])

    # Exact signal (fs=1, fb=0).
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
    truth_signal = (model.magnification(epochs)-1)*theta.truth[0]+1

    # Simulate noise in gaussian errored flux space.
    np.random.seed(seed)
    noise = np.random.normal(0.0, np.sqrt(truth_signal) / signal_to_noise_base, n_epochs) 
    noise_std_dev = np.sqrt(truth_signal) / signal_to_noise_base
    
    signal = truth_signal + noise

    data = mm.Mulenstd_devata(data_list = [epochs, signal, noise_std_dev], phot_fmt = "flux", chi2_fmt = "flux")

    return data


def synthetic_binary_lens(theta, n_epochs, signal_to_noise_base, seed = 42):
    """Generate a synthetic single lens light curve.
    
    Simulates noise based on guassian flux process.
    In this simplified case, amplification = flux.
    Produces equispaced observations from 0 to 72 days.
    Otherwise based on ROMAN photometric specifications.

    Args:
        theta: [state] Binary lens model parameters.
        n_epochs: [int] The number of flux observations.
        signal_to_noise_base: [float] The signal to noise baseline.
        seed: [optional, int] A random seed.

    Returns:
        data: [mulenstd_devata] Object for a synthetic light curve.
    """
    # Create MulensModel.
    model = mm.Model(dict(zip(["t_0", "u_0", "t_E", "q", "s", "alpha"], theta.truth[1:])))
    model.set_magnification_methods([0., "point_source", 72.])

    # Exact signal (fs=1, fb=0).
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
    truth_signal = (model.magnification(epochs)-1)*theta.truth[0]+1

    # Simulate noise in gaussian errored flux space.
    np.random.seed(seed)
    noise = np.random.normal(0.0, np.sqrt(truth_signal) / signal_to_noise_base, n_epochs) 
    noise_std_dev = np.sqrt(truth_signal) / signal_to_noise_base
    
    signal = truth_signal + noise

    data = mm.Mulenstd_devata(data_list = [epochs, signal, noise_std_dev], phot_fmt = "flux", chi2_fmt = "flux")

    return data



def binary_log_likelihood(self, theta):
    """Calculate the log likelihood of a state in a model.
    
    Uses the point source approximation from MulensModel to calculate
    the log likelihood that a binary state produced the model's data.
    Data must be over the range 0 to 72 days.

    Args:
        theta: [state] Binary model parameters.

    Returns:
        log_likelihood: [float] The resulting log likelihood.
    """
    try: # MulensModel may throw errors
        model = mm.Model(dict(zip(["t_0", "u_0", "t_E", "q", "s", "alpha"], theta.truth[1:])))
        model.set_magnification_methods([0., "point_source", 72.])

        a = model.magnification(self.data.time) # The proposed magnification signal.
        y = self.data.flux # The observed flux signal.
        
        # Fit proposed flux as least squares solution.
        #F = least_squares_signal(a, y)
        F = (a-1)*theta.truth[0]+1

        std_dev = self.data.err_flux
        chi2 = np.sum((y - F)**2/std_dev**2)

    except: # If MulensModel crashes, return true likelihood zero.
        return -math.inf

    return -chi2/2 # Transform chi2 to log likelihood.

def least_squares_signal(a, y):
    # Fit proposed flux as least squares solution.
    A = np.vstack([a, np.ones(len(a))]).T
    f_s, f_b = np.linalg.lstsq(A, y, rcond = None)[0]
    F = f_s*a + f_b # The least squares signal.
    return F

def single_log_likelihood(self, theta):
    """Calculate the log likelihood of a state in a model.
    
    Uses the point source approximation from MulensModel to calculate
    the log likelihood that a single state produced the model's data.
    Data must be over the range 0 to 72 days.

    Args:
        theta: [state] Single model parameters.

    Returns:
        log_likelihood: [float] The resulting log likelihood.
    """
    try: # MulensModel may throw errors
        model = mm.Model(dict(zip(["t_0", "u_0", "t_E"], theta.truth[1:])))
        model.set_magnification_methods([0., "point_source", 72.])

        a = model.magnification(self.data.time) # The proposed magnification signal.
        y = self.data.flux # The observed flux signal.
        
        # Fit proposed flux as least squares solution.
        #F = least_squares_signal(a, y)
        F = (a-1)*theta.truth[0]+1

        std_dev = self.data.err_flux
        chi2 = np.sum((y - F)**2/std_dev**2)

    except: # If MulensModel crashes, return true likelihood zero.
        return -math.inf

    return -chi2/2 # Transform chi2 to log likelihood.