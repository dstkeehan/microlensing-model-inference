"""Interface with neural network surrogate posteriors."""

import pickle
import numpy as np

# File access. 
import os
import os.path
from pathlib import Path

import pickle
import io
from sbi.utils.torchutils import BoxUniform
from sbi.inference.posteriors.direct_posterior import DirectPosterior
import torch

from sklearn.cluster import OPTICS
from sklearn.preprocessing import MinMaxScaler

class Surrogate_Posterior(object):
    def __init__(self, model_index, data):
        """Initialises the model."""
        self.model_index = model_index

        """Get a single or binary model posterior.
            
        Args:
            model_index: [int] Model index, single or binary, 0 or 1.

        Returns:
            posterior: [pickle] Posterior object.
        """

        path = os.getcwd()
        #path = (str(Path(path).parents[0]))

        if model_index == 0:
            #with open(path+"/distributions/single_25K_720.pkl", "rb") as handle: distribution = pickle.load(handle)
            self.distribution = load_posterior(path+"/distributions/single_100K_720_T10.pkl")
            self.distribution._prior = BoxUniform([0.0, 0.0, 1.0, 1e-4, 0.1], [72.0, 2.0, 100.0, 1e-2, 1.0])

        if model_index == 1:
            #with open(path+"/distributions/binary_100K_720.pkl", "rb") as handle: distribution = pickle.load(handle)
            self.distribution = load_posterior(path+"/distributions/binary_500K_7200_LONG_10.pkl")



        #self.distribution = distribution

        self.data = data


    def sample(self, number_of_samples):
        np.random.RandomState(42)
        self.samples = self.distribution.sample((number_of_samples,), x=self.data, show_progress_bars=False)
        
        return

    def get_modes(self, latex_output = False):
        """Get the modes of a multidimensional sampled distribution using the OPTICS sampler.

        Args:
            samples (np.ndarray): samples to find modes from.
            latex_output (bool, optional): if latex output of modes is wanted. Defaults to False.

        Returns:
            np.ndarray: array of mode centre locations.
        """

        # Fit min-max scaler to ensure each dimension is handled similarly
        scaled_samples = MinMaxScaler().fit_transform(self.samples.numpy())

        # Apply OPTICS sampler with specified settings and fit to samples
        clust = OPTICS(min_samples=50, min_cluster_size=100, xi=0.05, max_eps=0.1).fit(
            scaled_samples
        )

        labels = clust.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print(f"{n_clusters_} modes, {n_noise_} samples not assigned... ")

        modes = []
        mode_samples = []
        n_samples = []

        # Sort by samples.
        for i in range(n_clusters_):
            n_samples.append(len(self.samples[labels == i]))
        order = np.argsort(n_samples)
        order = np.flip(order)

        # Go through each cluster and find statistics
        k = 0
        for i in order:

            samples_i = self.samples[labels == i]

            mode_i = np.zeros((samples_i.shape[1]))

            print(f"\nMode: {k} assigned {len(samples_i)}")
            k += 1
            latex_string = ""

            for j in range(samples_i.shape[1]):
                # If the number of clusters is less than 2, just use all the samples
                if n_clusters_ < 2:
                    temp = np.percentile(self.samples[:, j], [16, 50, 84])
                else:
                    temp = np.percentile(samples_i[:, j], [16, 50, 84])

                print(f"{temp[1]:.4f} +{temp[2]-temp[1]:.4f} -{temp[1]-temp[0]:.4f}")

                latex_string += f"${temp[1]:.4f}^{{+{temp[2]-temp[1]:.4f}}}_{{-{temp[1]-temp[0]:.4f}}}$ & "

                mode_i[j] = temp[1]

            modes.append(np.concatenate(([mode_i[-1]], mode_i[:-1])))
            #print(np.array(samples_i)[:,-1])
            #print(np.array(samples_i)[:,:-1])            
            mode_samples.append(np.concatenate((np.array(samples_i)[:,-1:], np.array(samples_i)[:,:-1]), axis=1))
            #print(np.concatenate(([mode_i[-1]], mode_i[:-1])))
            #print(np.concatenate((np.array(samples_i)[:,-1:], np.array(samples_i)[:,:-1]), axis=1))
            #throw=throw

            if latex_output:
                print(latex_string)

        self.modes = np.array(modes)
        self.mode_samples = np.array(mode_samples)

        return


    def max_aposteriori(self):
        """Maximise a posterior.
            
        The input signal_data conditions the posterior to data.

        Args:
            posterior: [pickle] Posterior object.
            signal_data: [list] Measured flux signals at discrete times.

        Returns:
            centre: [list] Estimated parameter values of maximum.
        """
        centre = np.array(np.float64(self.distribution.map(self.data, num_iter=100, num_init_samples=100, show_progress_bars=False)))
        
        print(centre)

        return centre


def fix(map_loc):
    # Closure rather than a lambda to preserve map_loc
    return lambda b: torch.load(io.BytesIO(b), map_location=map_loc)


class MappedUnpickler(pickle.Unpickler):
    """Open a pickle file and map to correct device at the same time. Used to load GPU posteriors on CPU. https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219."""

    def __init__(self, *args, map_location="cpu", **kwargs):
        """Create mapped unpickler which loads pickle file onto specified device.

        Args:
            map_location (str, optional): location to load pickle file onto. Defaults to "cpu".
        """
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return fix(self._map_location)
        else:
            return super().find_class(module, name)


def load_posterior(filename: str) -> DirectPosterior:
    """Load pickled posterior onto CPU.

    Args:
        filename (str): path to pickled posterior.

    Returns:
        DirectPosterior: posterior transferred to CPU.
    """
    with open(filename, "rb") as handle:
        unpickler = MappedUnpickler(handle, map_location="cpu")
        posterior = unpickler.load()

    posterior._device = torch.device("cpu")

    return posterior