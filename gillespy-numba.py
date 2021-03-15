import numpy as np
import matplotlib.pyplot as plt
from numba import njit, f8, i8
from collections import namedtuple

@njit
def random_choice(arr,probs):
    '''Sample elements of an array with specific probabilities.
        `arr`: np.array, from which to sample. Possible n-dimensional
        `probs`: np.array, same size as `arr`. No need to sum up to 1.
    Returns a random sample from `arr`.
    Sampling from a matrix will select from its rows.
    This function exists because numba does not support the `p` flag on `random.choice`.
    See [https://github.com/numba/numba/issues/2539] for details.'''
    cummulative_probs = np.cumsum(probs)/np.sum(probs)
    index = np.searchsorted(cummulative_probs, np.random.random(), side="right")
    return arr[index]


def GillespieModel(a_foo, v_foo):
    '''Generates an object to simulate the system with the Gillespie algorithm.
        `a_foo`: an NJIT-compiled function f8[:](f8[:]).
            a_foo(x_vec) should give the a_j probabilities vector for the reactions.
        `v_foo`: an NJIT-compiled function f8[:,:](f8[:]).
            v_foo(x_vec) should give the v_ij state-change matrix.
            The first column of this matrix *must* be zeros.
    Returns an object with the functions:
        * step
        * single_system_step
        * multi_system_step
        * evolve
        * single_system_evolve
        * multi_system_evolve
    The probability vector and state-change matrix are functions of the state. This is not true
    in general, and its coded that way to provide flexibility.
    Explicit time-dependent state changes should be treated with care, as they lie outside of the
    statistical considerations of Gillespie's work.
    '''
    # this is only notation for the object-oriented people.
    wrapper = namedtuple("GillespieModel",[
        "step",
        "single_system_step",
        "multi_system_step",
        "evolve",
        "single_system_evolve",
        "multi_system_evolve"
    ])

    @njit(f8[:](f8[:]))
    def step(x_ini):
        '''Evolves the system one step from the given initial state.
            `x_ini`: 1D-array of floats with elements [time,S1,S2,...,SN]
        Returns the final state of the system, same length as `x_ini`.
        '''
        # generates probability array and state-change matrix
        a_arr = a_foo(x_ini)
        a_sum = np.sum(a_arr)
        if a_sum == 0.0:
            # system is stiff
            return x_ini
        else:
            tau = np.zeros(len(x_ini))
            tau[0] = np.random.exponential(1/a_sum)
            v_matrix = v_foo(x_ini)
            v_j = random_choice(v_matrix, a_arr)
            return x_ini + v_j + tau

    @njit(f8[:,:](f8[:],i8))
    def single_system_step(x_ini, max_iter):
        '''Evolves one system a definite number of iterations from the initial state.
            `x_ini` is the same as in `step`.
            `max_iter`: integer, iteration number.
        Returns an array of states of length `max_iter`.
        The relationship between the final time and the number of iterations is not trivial,
        as time intervals depend on the system states.
        '''
        states = np.zeros( (max_iter, len(x_ini)) )
        states[0] = x_ini
        for i in range(1,max_iter):
            states[i] = step(states[i-1])
        return states

    @njit(f8[:,:,:](f8[:,:],i8))
    def multi_system_step(x_ini_vec,max_iter):
        '''Evolves several systems at once a definite number of iterations.
            `x_ini_vec`: 2D-array of floats. Vector of inital states, one for each system.
            `max_iter`: integer, iteration number.
        Returns a matrix with columns [time,S1,S2,...,SN], with all the states of all the systems,
        ordered in time, so the information of which data belongs to which system is lost.
        '''
        n_systems = len(x_ini_vec)
        n_variables = len(x_ini_vec[0])
        simulations = np.zeros( (n_systems,max_iter,n_variables) )
        for i in range(n_systems):
            simulations[i] = single_system_step(x_ini_vec[i],max_iter)
        return simulations

    @njit(f8[:](f8[:],f8))
    def evolve(x_ini,t_max):
        '''Evolves the system from the given initial state to a certain time.
            `x_ini`: 1D-array of floats with elements [time,S1,S2,...,SN]
            `t_max`: float, time where to stop the simulation
    Returns the final system state vector.
        The simulation will stop *just before* the system reaches `t_max`.
        If the system becames stiff, a consistent final state will be returned and no warning will be raised.
        '''
        x_current = x_ini
        x_new = step(x_current)
        while x_new[0] < t_max:
            x_current = x_new
            x_new = step(x_current)
            if x_new[0] == x_current[0]: # if system is stiff
                x_new[0] = x_current[0] = t_max # advance to the end
        return x_current

    @njit(f8[:,:](f8[:],f8[:]))
    def single_system_evolve(x_ini,t_arr):
        '''Evolves the system from the given initial state to each time on `t_arr`.
            `x_ini`: 1D-array of floats with elements [time,S1,S2,...,SN]
            `t_arr`: 1D-array of floats, with the times on which measure the system.
                Must be sorted
        Returns an array whose nth element is the system's state at the nth time.
        This function effectively measures one trajectory of the system at a given stoptimes.
        '''
        states = np.zeros( (len(t_arr),len(x_ini)) )
        states[0] = evolve(x_ini,t_arr[0])
        for i in range(1,len(t_arr)):
            states[i] = evolve(states[i-1],t_arr[i])
        return states

    @njit(f8[:,:,:](f8[:,:],f8[:]))
    def multi_system_evolve(x_ini_vec,t_arr):
        '''Evolves multiple systems from multiple initial states, measuring them at the given times.
            `x_ini_vec`: 2D-array of floats with the initial state of each system.
            `t_arr`: 1D-array of floats, with the times on which measure the system.
        Returns an array whose nth element is the equivalent of applying `single_system_evolve` 
        to the nth initial state.
        '''
        n_systems = len(x_ini_vec)
        n_variables = len(x_ini_vec[0])
        simulations = np.zeros( (n_systems,len(t_arr),n_variables) )
        for i in range(n_systems):
            simulations[i] = single_system_evolve(x_ini_vec[i],t_arr)
        return simulations

    return wrapper(step, single_system_step, multi_system_step, evolve, single_system_evolve, multi_system_evolve)


# reshape data to a big matrix
#data = simulations.reshape(n_systems*max_iter,n_variables)
# order the data with respect to the first column (time)
#data = data[np.argsort(data.T[0])]
# return data up to the time when the first system ended its simulation
#systems_last_times = simulations[:,-1].T[0]
#t_limit = np.min(systems_last_times)
#mask = data.T[0] <= t_limit
#return data[mask]
