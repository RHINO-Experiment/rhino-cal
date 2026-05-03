import numpy as np
from scipy.sparse.linalg import cg
from multiprocessing import shared_memory, Pool
import os

_A = None
_TmT = None
_TNd = None
_SinS = None
_Sinv = None
_Ninv = None
_sqrt_Ninv = None
_sqrt_Sinv = None

_shm_refs = []

def init_worker(A_name, A_shape, A_dtype,
                TmT_name, TmT_shape, TmT_dtype,
                TNd_name, TNd_shape, TNd_dtype,
                SinS_name, SinS_shape, SinS_dtype,
                Sinv_name, Sinv_shape, Sinv_dtype,
                Ninv_name, Ninv_shape, Ninv_dtype):

    global _A, _TmT, _TNd, _SinS, _Sinv, _Ninv, _sqrt_Ninv, _sqrt_Sinv, _shm_refs

    shm_A = shared_memory.SharedMemory(name=A_name)
    shm_TmT = shared_memory.SharedMemory(name=TmT_name)
    shm_TNd = shared_memory.SharedMemory(name=TNd_name)
    shm_SinS = shared_memory.SharedMemory(name=SinS_name)
    shm_Sinv = shared_memory.SharedMemory(name=Sinv_name)
    shm_Ninv = shared_memory.SharedMemory(name=Ninv_name)

    _shm_refs = [shm_A, shm_TmT, shm_TNd, shm_SinS, shm_Sinv, shm_Ninv]

    _A = np.ndarray(A_shape, dtype=A_dtype, buffer=shm_A.buf)
    _TmT = np.ndarray(TmT_shape, dtype=TmT_dtype, buffer=shm_TmT.buf)
    _TNd = np.ndarray(TNd_shape, dtype=TNd_dtype, buffer=shm_TNd.buf)
    _SinS = np.ndarray(SinS_shape, dtype=SinS_dtype, buffer=shm_SinS.buf)
    _Sinv = np.ndarray(Sinv_shape, dtype=Sinv_dtype, buffer=shm_Sinv.buf)
    _Ninv = np.ndarray(Ninv_shape, dtype=Ninv_dtype, buffer=shm_Ninv.buf)

    _sqrt_Ninv = np.sqrt(_Ninv)
    _sqrt_Sinv = np.sqrt(_Sinv)


def worker(idx):
    pid = os.getpid()

    rng = np.random.default_rng(int(idx+pid))

    omega_d = rng.normal(size=_Ninv.shape)
    omega_t = rng.normal(size=_Sinv.shape)

    data_term = _TNd + _TmT @ (_sqrt_Ninv * omega_d)
    prior_term = _SinS + (_sqrt_Sinv * omega_t)

    b = data_term + prior_term

    x, info = cg(_A, b, rtol=1e-10,
              atol=1e-8,
              maxiter=int(1e6))

    if info != 0:
        print(f"CG did not converge for idx {idx}, info={info}")

    return x


def generate_gcr_solutions_mp(n_gcr_sol,
                             transfer_matrix,
                             N_inv,
                             S_inv,
                             data_vector,
                             priors_vector,
                             nproc=None):

    A = transfer_matrix.T @ (N_inv[:, None] * transfer_matrix) + S_inv
    TmT = transfer_matrix.T
    TNd = TmT @ (N_inv * data_vector)
    SinS = S_inv * priors_vector

    # Allocate shared memory
    shm_A = shared_memory.SharedMemory(create=True, size=A.nbytes)
    shm_TmT = shared_memory.SharedMemory(create=True, size=TmT.nbytes)
    shm_TNd = shared_memory.SharedMemory(create=True, size=TNd.nbytes)
    shm_SinS = shared_memory.SharedMemory(create=True, size=SinS.nbytes)
    shm_Sinv = shared_memory.SharedMemory(create=True, size=S_inv.nbytes)
    shm_Ninv = shared_memory.SharedMemory(create=True, size=N_inv.nbytes)


    np.ndarray(A.shape, dtype=A.dtype, buffer=shm_A.buf)[:] = A
    np.ndarray(TmT.shape, dtype=TmT.dtype, buffer=shm_TmT.buf)[:] = TmT
    np.ndarray(TNd.shape, dtype=TNd.dtype, buffer=shm_TNd.buf)[:] = TNd
    np.ndarray(SinS.shape, dtype=SinS.dtype, buffer=shm_SinS.buf)[:] = SinS
    np.ndarray(S_inv.shape, dtype=S_inv.dtype, buffer=shm_Sinv.buf)[:] = S_inv
    np.ndarray(N_inv.shape, dtype=N_inv.dtype, buffer=shm_Ninv.buf)[:] = N_inv

    try:
        with Pool(processes=nproc,
            initializer=init_worker,
            initargs=(
                shm_A.name, A.shape, A.dtype,
                shm_TmT.name, TmT.shape, TmT.dtype,
                shm_TNd.name, TNd.shape, TNd.dtype,
                shm_SinS.name, SinS.shape, SinS.dtype,
                shm_Sinv.name, S_inv.shape, S_inv.dtype,
                shm_Ninv.name, N_inv.shape, N_inv.dtype,
            ),
        ) as pool:

            results = pool.map(worker, range(n_gcr_sol))

    finally:
        # Always clean up
        for shm in [shm_A, shm_TmT, shm_TNd, shm_SinS, shm_Sinv, shm_Ninv]:
            shm.close()
            shm.unlink()

    return results