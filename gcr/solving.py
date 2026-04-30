# Scripts for Solving the Linear Equations

import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import solve
import os
from multiprocessing import shared_memory, Pool

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

def solve_gcr(transfer_matrix,
              N_inv,
              S_inv,
              data_vector,
              priors_vector,
              omega_t,
              omega_d,
              use_linsolv_start=False,
              rtol=1e-10,
              atol=1e-8,
              maxiter=int(1e6)):
    """

    """
    if use_linsolv_start:
        x0 = np.linalg.lstsq(transfer_matrix, data_vector)[0]
    A = construct_gcr_lhs(transfer_matrix, N_inv, S_inv)
    b = construct_gcr_rhs(transfer_matrix, N_inv, S_inv, data_vector, priors_vector, omega_t, omega_d)
    if use_linsolv_start:
        return cg(A, b, x0=x0, maxiter=maxiter, rtol=rtol, atol=atol)[0]
    else:
        return cg(A, b, maxiter=maxiter, rtol=rtol, atol=atol)[0]


def generate_gcr_solutions_serial(n_gcr_sol,
                              transfer_matrix,
                              N_inv,
                              S_inv,
                              data_vector,
                              priors_vector,
                              use_linsolv_start=False,
                              rtol=1e-10,
                              atol=1e-8,
                              maxiter=int(1e6)):
    """
    Generates GCR solutions in serial.
    """
    idx_array = np.arange(n_gcr_sol)

    def _process_gcr_solution(idx):
        np.random.seed(idx)
        print(idx)
        omega_d = np.random.normal(size=data_vector.shape)
        omega_t = np.random.normal(size=priors_vector.shape)
        return solve_gcr(transfer_matrix=transfer_matrix,
                         N_inv=N_inv,
                         S_inv=S_inv,
                         data_vector=data_vector,
                         priors_vector=priors_vector,
                         omega_d=omega_d,
                         omega_t=omega_t,
                         use_linsolv_start=use_linsolv_start,
                         rtol=rtol,
                         atol=atol,
                         maxiter=maxiter)
    
    results = [_process_gcr_solution(idx) for idx in idx_array]

    return results



# Multiprocessing Code

def process_gcr_solution(idx,
                         transfer_matrix,
                         N_inv,
                         S_inv,
                         data_vector,
                         priors_vector,
                         use_linsolv_start=False,
                         rtol=1e-10,
                         atol=1e-8,
                         maxiter=int(1e6)):
    np.random.seed(idx)
    print(idx)
    omega_d = np.random.normal(size=data_vector.shape)
    omega_t = np.random.normal(size=priors_vector.shape)

    A = construct_gcr_lhs(transfer_matrix, N_inv, S_inv)
    b = construct_gcr_rhs(transfer_matrix, N_inv, S_inv, data_vector, priors_vector, omega_t, omega_d)

    if use_linsolv_start:
        x0 = np.linalg.lstsq(transfer_matrix, data_vector)[0]
        return cg(A, b, x0=x0, maxiter=maxiter, rtol=rtol, atol=atol)[0]
    else:
        return cg(A, b, maxiter=maxiter, rtol=rtol, atol=atol)[0]



def generate_gcr_solutions_mp(n_gcr_sol,
                              transfer_matrix,
                              N_inv,
                              S_inv,
                              data_vector,
                              priors_vector,
                              use_linsolv_start=False,
                              rtol=1e-10,
                              atol=1e-8,
                              maxiter=int(1e6)):
    """
    Generates GCR solutions with multiprocessing and shared memory.
    """
    idx_list = list(range(n_gcr_sol))
    def init_worker(shm_info):
        global solving_arrays

        solving_arrays = []

        for name, shape, dtype in shm_info:
            shm = shared_memory.SharedMemory(name=name)
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            solving_arrays.append(array)

    def mp_process_gcr_solution(idx):
        #process_id = os.getpid()
        seed = idx #+ process_id

        np.random.seed(seed)
        print(idx)
        transfer_matrix, N_inv, S_inv, data_vector, priors_vector = solving_arrays # get data from shared memeory
        
        return solve_gcr(transfer_matrix=transfer_matrix,
                         N_inv=N_inv,
                         S_inv=S_inv,
                         data_vector=data_vector,
                         priors_vector=priors_vector,
                         use_linsolv_start=use_linsolv_start,
                         rtol=1e-10,
                         atol=1e-8,
                         maxiter=int(1e6))


    shm_blocks = []
    shm_infor = []

    for arr in [transfer_matrix, N_inv, S_inv, data_vector, priors_vector]:
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shared_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(shared_array, arr)

        shm_blocks.append(shm)
        shm_infor.append((shm.name, arr.shape, arr.dtype))


    with Pool(initializer=init_worker, initargs=(shm_infor,)) as pool:
        results = pool.imap_unordered(mp_process_gcr_solution, idx_list)

    
    for shm in shm_blocks:
        shm.close()
        shm.unlink()

    return list(results)