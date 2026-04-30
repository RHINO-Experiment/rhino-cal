# Module for solving GCR equation with multiprocessing and shared memory.

import numpy as np
from scipy.sparse.linalg import cg
from multiprocessing import shared_memory, Pool

_transfer_matrix = None
_N_inv = None
_S_inv = None
_data_vector = None
_priors_vector = None
_A = None
_TmT = None

_TNd = None
_Sin_s = None

def construct_gcr_lhs(transfer_matrix,
                      N_inv,
                      S_inv):
    """
    """
    # Assume N_inv is diagonal

    return transfer_matrix.T @ (N_inv[:,None] * transfer_matrix) + S_inv


def construct_gcr_rhs(transfer_matrix,
                      N_inv_1d,
                      S_inv,
                      d,
                      T_nw0,
                      omega_t,
                      omega_d):
    """
    """
    data_term = (transfer_matrix.T @ (N_inv_1d * d)) +\
        transfer_matrix.T @ (np.sqrt(N_inv_1d)*omega_d)
    prior_term = (S_inv * T_nw0) + (np.sqrt(S_inv)*omega_t)

    return data_term + prior_term


def init_worker(tm_name,
                tm_shape,
                tm_dtype,
                N_inv_name,
                N_inv_shape,
                N_inv_dtype,
                S_inv_name,
                S_inv_shape,
                S_inv_dtype,
                data_vector_name,
                data_vector_shape,
                data_vector_dtype,
                priors_vector_name,
                priors_vector_shape,
                priors_vector_dtype):
    print('Initializing worker with shared memory...')
    global _transfer_matrix, _N_inv, _S_inv, _data_vector, _priors_vector, _A, _TmT, _TNd, _Sin_s

    shm_tm = shared_memory.SharedMemory(name=tm_name)
    _transfer_matrix = np.ndarray(tm_shape, dtype=tm_dtype, buffer=shm_tm.buf)

    shm_N_inv = shared_memory.SharedMemory(name=N_inv_name)
    _N_inv = np.ndarray(N_inv_shape, dtype=N_inv_dtype, buffer=shm_N_inv.buf)

    shm_S_inv = shared_memory.SharedMemory(name=S_inv_name)
    _S_inv = np.ndarray(S_inv_shape, dtype=S_inv_dtype, buffer=shm_S_inv.buf)

    shm_data_vector = shared_memory.SharedMemory(name=data_vector_name)
    _data_vector = np.ndarray(data_vector_shape, dtype=data_vector_dtype, buffer=shm_data_vector.buf)

    shm_priors_vector = shared_memory.SharedMemory(name=priors_vector_name)
    _priors_vector = np.ndarray(priors_vector_shape, dtype=priors_vector_dtype, buffer=shm_priors_vector.buf)

    _A = construct_gcr_lhs(_transfer_matrix, _N_inv, _S_inv)

    _TmT = _transfer_matrix.T

    _TNd = _TmT @ (_N_inv * _data_vector)

    _Sin_s = _S_inv * _priors_vector

def worker(idx):
    np.random.seed(idx)
    print(idx)
    _omega_d = np.random.normal(size=_data_vector.shape)
    _omega_t = np.random.normal(size=_priors_vector.shape)
    
    _data_term = _TNd +\
        _TmT @ (np.sqrt(_N_inv)*_omega_d)
    print('Computed data term for idx:', idx)
    _prior_term = _Sin_s + (np.sqrt(_S_inv)*_omega_t)
    print('Computed prior term for idx:', idx)
    _b = _data_term + _prior_term

    print('Computed b for idx:', idx)
    x = cg(_A, _b, maxiter=int(1e6), rtol=1e-10, atol=1e-8)[0]
    print('Finished idx:', idx)
    return x

def generate_gcr_solutions_mp(n_gcr_sol,
                              transfer_matrix,
                              N_inv,
                              S_inv,
                              data_vector,
                              priors_vector):
    """
    Generates GCR solutions with multiprocessing and shared memory.
    """
    # Create shared memory blocks for the input arrays
    shm_transfer_matrix = shared_memory.SharedMemory(create=True, size=transfer_matrix.nbytes)
    shm_N_inv = shared_memory.SharedMemory(create=True, size=N_inv.nbytes)
    shm_S_inv = shared_memory.SharedMemory(create=True, size=S_inv.nbytes)
    shm_data_vector = shared_memory.SharedMemory(create=True, size=data_vector.nbytes)
    shm_priors_vector = shared_memory.SharedMemory(create=True, size=priors_vector.nbytes)

    # Copy data into shared memory
    np_transfer_matrix = np.ndarray(transfer_matrix.shape, dtype=transfer_matrix.dtype, buffer=shm_transfer_matrix.buf)
    np_N_inv = np.ndarray(N_inv.shape, dtype=N_inv.dtype, buffer=shm_N_inv.buf)
    np_S_inv = np.ndarray(S_inv.shape, dtype=S_inv.dtype, buffer=shm_S_inv.buf)
    np_data_vector = np.ndarray(data_vector.shape, dtype=data_vector.dtype, buffer=shm_data_vector.buf)
    np_priors_vector = np.ndarray(priors_vector.shape, dtype=priors_vector.dtype, buffer=shm_priors_vector.buf)

    np_transfer_matrix[:] = transfer_matrix[:]
    np_N_inv[:] = N_inv[:]
    np_S_inv[:] = S_inv[:]
    np_data_vector[:] = data_vector[:]
    np_priors_vector[:] = priors_vector[:]
    # Prepare arguments for worker initialization
    with Pool(initializer=init_worker, initargs=[
        shm_transfer_matrix.name, transfer_matrix.shape, transfer_matrix.dtype,
        shm_N_inv.name, N_inv.shape, N_inv.dtype,
        shm_S_inv.name, S_inv.shape, S_inv.dtype,
        shm_data_vector.name, data_vector.shape, data_vector.dtype,
        shm_priors_vector.name, priors_vector.shape, priors_vector.dtype
    ]) as pool:
        idx_list = list(range(n_gcr_sol))
        results = pool.map(worker, idx_list)
    #FIXME: CPU usage high. Needs further optimsation for parralelization.
    # Clean up shared memory
    shm_transfer_matrix.close()
    shm_N_inv.close()
    shm_S_inv.close()
    shm_data_vector.close()
    shm_priors_vector.close()
    shm_transfer_matrix.unlink()
    shm_N_inv.unlink()
    shm_S_inv.unlink()
    shm_data_vector.unlink()
    shm_priors_vector.unlink()

    return results