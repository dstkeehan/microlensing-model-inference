"""Adaptive Reversible-Jump Metropolis Hastings for microlensing.

Implements algorithms for bayesian sampling. Uses the main 
classes: State, Chain, and model.
"""

import math
import random
import numpy as np
from scipy.stats import multivariate_normal
from copy import deepcopy
from types import MethodType


class State(object):
    """State sampled from a model's probability distribution.

    Describes a point in both scaled and unscaled space. The scaling is 
    hardcoded but can be extended per application. Currently log10 scaling 
    the fourth parameter. In microlensing applications, this is q,
    the mass ratio.

    Attributes:
        truth: [list] Parameter values for the state, in true space.
        scaled: [list] Parameter values for the state, in scaled space.
        dims: [int] The dimensionality of the state.
    """

    def __init__(self, truth = None, scaled = None):
        """Initialises state with truth, scaled, and D values.
        
        Only one of truth or scaled is needed.
        """        
        if truth is not None:
            self.truth = truth
            self.dims = len(truth)

            if self.dims > 4:
                self.truth[6] = truth[6] % 360 # Radial symmetry of alpha.

            self.scaled = deepcopy(self.truth)
            for p in range(self.dims):
                if p == 4:
                    self.scaled[p] = np.log10(self.truth[p])
        
        elif scaled is not None:
            self.scaled = scaled
            self.dims = len(scaled)

            if self.dims > 4:
                self.scaled[6] = scaled[6] % 360 # Radial symmetry of alpha.

            self.truth = deepcopy(self.scaled)
            for p in range(self.dims):
                if p == 4:
                    self.truth[p] = 10**(self.scaled[p])

        else:   raise ValueError("Assigned null state")


class Chain(object):
    """Collection of states.

    Describes a markov chain, perhaps from a joint model space.

    Attributes:
        states: [list] State objects in the chain.
        model_indices: [list] Models the states are from. 
        length: [int] The number of states in the chain.
    """

    def __init__(self, model_index, state):
        """Initialises the chain with one state from one model.

        Args:
            state: [state] The state object.
            model_index: [int] The index of the model the state is from.
        """
        self.states = [state]
        self.model_indices = [model_index]
        self.length = 1

    def add_general_state(self, model_index, state):
        """Adds a state in a model to the chain.

        Args:
            state: [state] The state object.
            model_index: [int] The index of the model the state is from.
        """
        self.states.append(state)
        self.model_indices.append(model_index)
        self.length += 1
        return

    def states_array(self, scaled = True):
        """Creates a numpy array of all states in the chain.

        Args:
            scale: [optional, bool] Whether the array should be in scaled or 
                                    true space.

        Returns:
            chain_array: [np.array] The numpy array of all state parameters. 
                                    Columns are states, rows are parameters 
                                    for all states.
        """
        
        chain_array = np.zeros((len(self.states[-1].scaled), self.length))

        if scaled:
            for i in range(self.length):
                chain_array[:, i] = self.states[i].scaled

        else:
            for i in range(self.length):
                chain_array[:, i] = self.states[i].truth

        return chain_array


class Model(object):
    """A model to describe a probability distribution.

    Contains a chain of states from this model, as well as information
    from this. Adapts a covariance matrix iteratively with each new state,
    and stores a guess at a maximum posterior density estimate.

    Attributes:
        model_index: [int] Model index.
        dims: [int] Dimensionality of a state in the model.
        priors: [list] Prior distribution objects for state parameter values.
        sampled: [chain] States sampled from the model's distribution.
        scaled_average_state: [list] The scaled average parameter values of the chain.
        centre: [state] Best guess at maximum posterior density.
        covariance: [array] Current covariance matrix, based on all states.
        covariances: [list] All previous covariance matrices.
        acceptance_history: [list] Binary values, 1 if the state proposed was accepted,
                    0 if it was rejected.
        data: [mulensdata] Object for photometry readings from the 
            microlensing event.
        log_likelihood: [function] Method to calculate the log likelihood a state is
                                    from this model.
        I: [np.array] Identity matrix the size of dims.
        s: [float] Mixing parameter (see Haario et al 2001).
    """

    def __init__(self, model_index, dims, centre, priors, covariance, data, log_likelihood_fnc, samples=None):
        """Initialises the model."""
        self.model_index = model_index
        self.dims = dims
        self.priors = priors
        self.centre = centre

        #if samples is not None:
        #    self.sampled = samples

        #else:
        self.sampled = Chain(model_index, centre)
        self.scaled_mean_state = centre.scaled
        
        self.acceptance_history = [1] # First state always accepted.
        self.covariance = covariance
        self.covariances = [covariance]

        self.data = data
        
        # Model's custom likelihood function.
        self.log_likelihood = MethodType(log_likelihood_fnc, self)

        self.I = np.identity(dims)
        self.s = 2.4**2 / dims # Arbitrary(ish), good value from Haario et al 2001.
    
    def add_state(self, theta, adapt = True):
        """Adds a sampled state to the model.

        Args:
            theta: [state] Parameters to add.
            adapt: [optional, bool] Whether or not to adjust the covariance 
                                    matrix based on the new state.
        """
        self.sampled.length += 1
        self.sampled.states.append(theta)

        if adapt:
            self.covariance = iterative_covariance(self.covariance, theta.scaled, self.scaled_mean_state, self.sampled.length, self.s, self.I)

        self.covariances.append(self.covariance)
        self.scaled_mean_state = iterative_mean(self.scaled_mean_state, theta.scaled, self.sampled.length)
        #self.centre = State(scaled = iterative_mean(self.scaled_mean_state, theta.scaled, self.sampled.n))

        return

    def log_prior_density(self, theta, auxiliary_values = None, auxiliary_dims = None):
        """Calculates the log prior density of a state in the model.

        Optionally adjusts this log density when using auxiliary vriables.

        Args:
            theta: [state] Parameters to calculate the log prior density for.
            auxiliary_values: [optional, state] The values of all auxiliary variables.
            auxiliary_dims: [optional, int] The dimensionality to use with auxiliary variables.

        Returns:
            log_prior_product: [float] The log prior probability density.
        """    
        log_prior_product = 0.

        # cycle through parameters
        for p in range(self.dims):

            # product using log rules
            log_prior_product += (self.priors[p].log_pdf(theta.truth[p]))

        # cycle through auxiliary parameters if v and v_D passed
        if auxiliary_values is not None or auxiliary_dims is not None:
            if auxiliary_values is not None and auxiliary_dims is not None:
                for p in range(self.dims, auxiliary_dims):
                    
                    # product using log rules
                    log_prior_product += (self.priors[p].log_pdf(auxiliary_values.truth[p]))

            else: raise ValueError("Only one of auxiliary_values or auxiliary_dims passed.")

        return log_prior_product


def iterative_mean(mean, value, n):
    return (mean * n + value)/(n + 1)

def iterative_covariance(covariance, value, mean, n, s, I, eps = 1e-12):
    return (n-1)/n * covariance + s/(n+1) * np.outer(value - mean, value - mean) + s*eps*I/n

def check_symmetric(matrix, tol = 1e-16):
    return np.all(np.abs(matrix-matrix.T) < tol)

def gaussian_proposal(theta, covariance):
    return multivariate_normal.rvs(mean = theta, cov = covariance)

def AMH(model, adaptive_iterations, fixed_iterations = 25, print_user_feedback = False):
    """Performs Adaptive Metropolis Hastings.
    
    Produces a posterior distribution by adapting the proposal process within 
    one model, as described in Haario et al (2001).

    Args:
        model: [model] Model object to sample the distribution from.
        adaptive_iterations: [int] Number of steps with adaption.
        fixed_iterations: [optional, int] Number of steps without adaption.
        print_user_feedback: [optional, bool] Whether or not to print progress.

    Returns:
        best_theta: [state] State producing the best posterior density visited.
        log_best_posterior: [float] Best log posterior density visited. 
    """

    if fixed_iterations < 5:
        raise ValueError("Not enough iterations to safely establish an empirical covariance matrix.")
    

    theta = model.centre
    best_theta = deepcopy(theta)

    # Initial propbability values.
    log_likelihood = model.log_likelihood(theta)
    log_prior = model.log_prior_density(theta)
    log_best_posterior = log_likelihood + log_prior
    log_posterior = log_likelihood + log_prior

    # Warm up walk to establish an empirical covariance.
    for i in range(1, fixed_iterations):

        # Propose a new state and calculate the resulting density.
        proposed = State(scaled = gaussian_proposal(theta.scaled, model.covariance))
        log_likelihood_proposed = model.log_likelihood(proposed)
        log_prior_proposed = model.log_prior_density(proposed)

        # Metropolis acceptance criterion.
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            # Accept proposal.
            theta = deepcopy(proposed)
            log_likelihood = log_likelihood_proposed
            model.acc.append(1)

            # Store best state.
            log_posterior = log_likelihood_proposed + log_prior_proposed
            if log_best_posterior < log_posterior:
                log_best_posterior = log_posterior
                best_theta = deepcopy(theta)

        else: model.acceptance_history.append(0) # Reject proposal.
        
        # Update storage.
        model.add_state(theta, adapt = False)

    # Calculate intial empirical covariance matrix.
    model.covariance = np.cov(model.sampled.states_array(scaled = True))
    model.covariances.pop()
    model.covariances.append(model.covariance)

    # Perform adaptive walk.
    for i in range(fixed_iterations, adaptive_iterations + fixed_iterations):

        if print_user_feedback:
            cf = i / (adaptive_iterations + fixed_iterations - 1)
            print(f'log density: {log_posterior:.4f}, progress: [{"#"*round(25*cf)+"-"*round(25*(1-cf))}] {100.*cf:.2f}%\r', end="")
            

        # Propose a new state and calculate the resulting density.
        proposed = State(scaled = gaussian_proposal(theta.scaled, model.covariance))
        log_likelihood_proposed = model.log_likelihood(proposed)
        log_prior_proposed = model.log_prior_density(proposed)

        # Metropolis acceptance criterion.
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            # Accept proposal.
            theta = deepcopy(proposed)
            log_likelihood = log_likelihood_proposed
            model.acceptance_history.append(1)

            # Store the best state.
            log_posterior = log_likelihood_proposed + log_prior_proposed
            if log_best_posterior < log_posterior:
                log_best_posterior = log_posterior
                best_theta = deepcopy(theta)

        else: model.acceptance_history.append(0) # Reject proposal.
        
        # Update model chain.
        model.add_state(theta, adapt = True)

    if print_user_feedback:
        print(f"\ninitialised model: {model.m}, mean acc: {(np.sum(model.acc) / (adaptive_iterations + fixed_iterations)):4f}, max log density: {log_best_posterior:.4f}\n")

    return best_theta, log_best_posterior


def ARJMH_proposal(model, proposed_model, theta, auxiliary_centre_offset):
    """Performs an Adaptive Reversible-Jump Metropolis Hastings proposal.
    
    Args:
        model: [model] Model to jump from.
        proposed_model: [model] Model to jump to.
        theta: [state] State to jump from.
        auxiliary_centre_offset: [list] Current auxiliary variables offset from centre.

    Returns:
        proposed_theta: [state] State proposed to jump to.
    """
    centre_offset = theta.scaled - model.centre.scaled # Offset from initial model's centre.

    if proposed_model.dims == model.dims: # Intra-model move.

        # Use the covariance at the proposed model's centre for local shape.
        proposed_offset = gaussian_proposal(np.zeros((proposed_model.dims)), proposed_model.covariance)
        proposed_theta = proposed_offset + centre_offset + proposed_model.centre.scaled
        
        return proposed_theta

    else: # Inter-model move.
        
        shared_dims = min(model.dims, proposed_model.dims) # Subset size.

        # Use superset model covariance
        if proposed_model.dims > model.dims: # proposed is superset
            covariance = proposed_model.covariance
        else: # proposed is subset
            covariance = model.covariance

        conditioned_covariance = schur_complement(covariance, shared_dims)

        # Jump to smaller model. Fix non-shared parameters.
        if proposed_model.dims < model.dims:

            proposed_offset = gaussian_proposal(np.zeros((shared_dims)), conditioned_covariance)
            proposed_theta = proposed_offset + centre_offset[:shared_dims] + proposed_model.centre.scaled

            return proposed_theta

        if proposed_model.dims > model.dims: # Jump to larger model. Append auxiliary variables.

            proposed_offset = gaussian_proposal(np.zeros((shared_dims)), conditioned_covariance)
            shared_map = proposed_offset + centre_offset[:shared_dims] + proposed_model.centre.scaled[:shared_dims]
            non_shared_map = auxiliary_centre_offset[shared_dims:] + proposed_model.centre.scaled[shared_dims:]
            map = np.concatenate((shared_map, non_shared_map))
            proposed_theta = map

            return proposed_theta

def schur_complement(covariance, shared_dims):
    C_11 = covariance[:shared_dims, :shared_dims] # Covariance matrix of shared parameters.
    C_12 = covariance[:shared_dims, shared_dims:] # Covariances, not variances.
    C_21 = covariance[shared_dims:, :shared_dims] # Same as above.
    C_22 = covariance[shared_dims:, shared_dims:] # Covariance matrix of non-shared parameters.
    C_22_inverse = np.linalg.inv(C_22)

    return C_11 - C_12.dot(C_22_inverse).dot(C_21)

def warm_up_model(empty_model, adaptive_iterations, fixed_iterations = 25, repetitions = 1, user_feedback = False):
    """Prepares a model for the ARJMH algorithm.
    
    Repeats the AMH warmup process for a model, storing the best run.

    Args:
        empty_model: [model] Initial model object.
        adaptive_iterations: [int] Number of adaptive steps.
        fixed_iterations: [optional, int] Number of non-adaptive steps.
        repetitions: [optional, int] Number of times to try for a better run.
        user_feedback: [optional, bool] Whether or not to print progress.

    Returns:
        incumbent_model: [model] Model with the states from the best run.
    """

    incumbent_log_best_posterior = -math.inf # Initialise incumbent posterior to always lose.

    for i in range(repetitions):
        
        if user_feedback:
            print("\ninitialising model ("+str(i+1)+"/"+str(repetitions)+")")

        model = deepcopy(empty_model) # Fresh model.

        # Run AMH.
        best_theta, log_best_posterior = AMH(model, adaptive_iterations, fixed_iterations, user_feedback)

        # Keep the best posterior density run.
        if incumbent_log_best_posterior < log_best_posterior:
            incumbent_model = deepcopy(model)
            incumbent_model.centre = deepcopy(best_theta)

    return incumbent_model


def ARJMH(models, iterations,  adaptive_warm_up_iterations, fixed_warm_up_iterations = 25, warm_up_repititions = 1, user_feedback = False):
    """Samples from a joint distribution of models.
    
    Initialises each model with multiple AMH runs. Then uses the resulting
    covariances to run ARJMH on all models.

    Args:
        models: [list] Model objects to sample from. 
                    Should be sorted by increasing dimensionality.
        iterations: [int] Number of adaptive RJMH steps.
        adaptive_warm_up_iterations: [int] Number of adaptive steps to initialise with.
        fixed_warm_up_iterations: [int] Number of fixed steps to initilaise with.
        warm_up_repititions: [int] Number of times to try for a better initial run.
        user_feedback: [optional, bool] Whether or not to print progress.

    Returns:
        joint_model_chain: [chain] Generalised chain with states from any model.
        total_acc: [list] Binary values, 1 if the state proposed was accepted,
                          0 if it was rejected, associated with the joint model.
        inter_model_history: [model] Stored inter-model covariances and acc.
    """

    if len(models) == 2:
        inter_model_acceptance_history = deepcopy(models[1])
        inter_model_acceptance_history.covariances = [schur_complement(models[1].covariance, models[0].dims)]
    else: inter_model_acceptance_history = None

    # Initialise model chains.
    MAP_value_estimates = []
    MAP_parameter_estimates = []
    for model_index in range(len(models)):
        models[model_index] = warm_up_model(models[model_index], adaptive_warm_up_iterations, fixed_warm_up_iterations, warm_up_repititions, user_feedback)
        MAP_value_estimates.append(-math.inf)
        MAP_parameter_estimates.append([])

    #print(models[1].covariance)

    random.seed(42)

    # Choose a random model to start in.
    model = random.choice(models)
    theta = deepcopy(model.sampled.states[-1]) # Final state in model's warmup chain.

    auxiliary_values = deepcopy(models[-1].sampled.states[-1]) # Auxiliary variables final state in super set model.
    auxiliary_centre_offset = models[-1].sampled.states[-1].scaled - models[-1].centre.scaled # Auxiliary variables offset from centre.

    # Create joint model as initial theta appended to auxiliary variables.
    initial_superset = models[-1].dims - model.dims
    if initial_superset > 0: # If random choice was a subset model
        theta_auxiliary = np.concatenate((theta.scaled, models[-1].sampled.states[-1].scaled[model.dims:]))
        joint_model_chain = Chain(model.model_index, State(scaled = theta_auxiliary))
    else:
        joint_model_chain = Chain(model.model_index, theta)

    acceptance_history = np.zeros(iterations)
    acceptance_history[0] = 1

    auxiliary_dims = models[-1].dims # Dimension of largest model is auxiliary variable size.

    # Initial probability values.
    log_likelihood = model.log_likelihood(theta)
    log_prior = model.log_prior_density(theta, auxiliary_values = auxiliary_values, auxiliary_dims = auxiliary_dims)

    np.set_printoptions(precision=2)

    if user_feedback: print("\nrunning ARJMH")
    for i in range(1, iterations): # ARJMH algorithm.
        
        if user_feedback:
            cf = i / (iterations - 1)
            print(f'model: {model.model_index}, log density: {(log_likelihood+log_prior):.4f}, progress: [{"#"*round(25*cf)+"-"*round(25*(1-cf))}] {100.*cf:.2f}%\r', end="")

        # Propose a new model and state and calculate the resulting density.
        random_model_index = random.randrange(0, len(models))
        proposed_model = models[random_model_index]

        proposed = State(scaled = ARJMH_proposal(model, proposed_model, theta, auxiliary_centre_offset))
        log_likelihood_proposed = proposed_model.log_likelihood(proposed)
        log_prior_proposed = proposed_model.log_prior_density(theta, auxiliary_values = auxiliary_values, auxiliary_dims = auxiliary_dims)

        # Metropolis acceptance criterion.
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            # Accept proposal.
            acceptance_history[i] = 1

            if model is proposed_model: # Intra model move.
                model.acceptance_history.append(1)
            elif inter_model_acceptance_history is not None: # Inter model move.
                inter_model_acceptance_history.acceptance_history.append(1)
                inter_model_acceptance_history.covariances.append(schur_complement(models[1].covariance, models[0].dims))
                inter_model_acceptance_history.sampled.length += 1

            model = proposed_model
            theta = deepcopy(proposed)

            log_likelihood = log_likelihood_proposed
            log_prior = log_prior_proposed

            if MAP_value_estimates[random_model_index] < log_likelihood + log_prior:
                MAP_value_estimates[random_model_index] = log_likelihood + log_prior
                MAP_parameter_estimates[random_model_index] = deepcopy(proposed)

        else: # Reject proposal.
            acceptance_history[i] = 0
            
            if model is proposed_model: # Intra model move.
                model.acceptance_history.append(0)
            elif inter_model_acceptance_history is not None: # Inter model move.
                inter_model_acceptance_history.acceptance_history.append(0)
                inter_model_acceptance_history.covariances.append(schur_complement(models[1].covariance, models[0].dims))
                inter_model_acceptance_history.sampled.length += 1
        
        # Update model chain.
        model.add_state(theta, adapt = True)
        auxiliary_values = State(scaled = np.concatenate((theta.scaled, auxiliary_values.scaled[model.dims:])))
        joint_model_chain.add_general_state(model.model_index, auxiliary_values)

        # Update auxiliary centre divergence for new states.
        auxiliary_centre_offset[:model.dims] = theta.scaled - model.centre.scaled

    if user_feedback:
        print(f"\nmean acc: {np.average(acceptance_history):4f}")
        for i in range(len(models)):
            print("P(m"+str(i)+"|y): " + str(joint_model_chain.model_indices.count(i) / iterations))
        #    print("P(m2|y): " + str(np.sum(joint_model_chain.model_indices) / iterations))
        #print(joint_model_chain.model_indices)

    return joint_model_chain, MAP_parameter_estimates, acceptance_history, inter_model_acceptance_history



def output_file(models, warm_up_iterations, joint_model_chain, acceptance_history, n_epochs, signal_to_noise_base, letters, name = "", event_params = None):
    
    # Output File.
    with open("results/"+name+"-run.txt", "w") as file:
        file.write("Run "+name+"\n")
        
        # Inputs.
        file.write("Inputs:\n")
        if event_params is not None:
            file.write("Parameters: "+str(event_params.truth)+"\n")
        file.write("Number of observations: "+str(n_epochs)+", Signal to noise baseline: "+str(signal_to_noise_base)+"\n")
        file.write("\n")
        file.write("Run information:\n")
        file.write("Iterations: "+str(joint_model_chain.length)+"\n")
        file.write("Average acceptance rate; Total: "+str(np.average(acceptance_history)))

        # Results.
        file.write("\n\nResults:\n")
        for model in models:
            
            # Models.
            model_probability = (model.sampled.length-warm_up_iterations)/joint_model_chain.n
            model_probability_std_error = np.std(np.array(joint_model_chain.model_indices))/(joint_model_chain.length**0.5)
            file.write("\n"+str(model.model_index)+"\nP(m|y): "+str(model_probability)+r"\pm"+str(model_probability_std_error)+"\n")

            # Parameters.
            model_states = model.sampled.states_array(scaled = True)
            for i in range(len(model.sampled.states[-1].scaled)):
                mean = np.average(model_states[i, :])
                std_dev = np.std(model_states[i, :], ddof = 1)
                file.write(letters[i]+": mean: "+str(mean)+", sd: "+str(std_dev)+" \n")
                
                # Hardcoded unscale q.
                if i == 3:
                    exp10 = np.power(10, model_states[i, :])
                    mean = np.average(exp10)
                    std_dev = np.std(exp10, ddof = 1)
                    file.write("q: mean: "+str(mean)+", sd: "+str(std_dev)+" \n")

    return

