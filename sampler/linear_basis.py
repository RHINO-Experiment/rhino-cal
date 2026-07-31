"""
Module for Creating the smooth basis functions.
"""


import numpy as np


def construct_noise_wave_transfer()




def polybasis_gen(x: np.ndarray,
                  y: np.ndarray,
                  n_x_coeffs: int,
                  n_y_coeffs: int):
    """
    Construct a 2D polynomial basis of the form

        f(x,y) = Σ_{m,n} a_{mn} x^m y^n

    Parameters
    ----------
    x, y : ndarray
        1D coordinate arrays.
    n_x_coeffs, n_y_coeffs : int
        Number of Polynomial modes in x and y.

    Returns
    -------
    basis : ndarray
        Shape (len(x)*len(y), n_x_coeffs*n_y_coeffs).
    """
    xx, yy = np.meshgrid(x, y)

    x_flat = xx.ravel()
    y_flat = yy.ravel()

    Vx = np.vander(x_flat, n_x_coeffs)
    Vy = np.vander(y_flat, n_y_coeffs)

    # Swap order to match original implementation
    basis = (Vy[:, :, None] * Vx[:, None, :]).reshape(
        x_flat.size, n_x_coeffs * n_y_coeffs
    )
    basis = np.asarray(basis, dtype=np.float64) # Ensure the basis is of type float64
    return basis


def construct_fourier_basis(
        x: np.ndarray,
        y: np.ndarray,
        n_x_coeffs: int,
        n_y_coeffs: int,
    ) -> np.ndarray:
    """
    Construct a 2D Fourier basis of the form

        cos(2πmx + 2πny)
        sin(2πmx + 2πny)

    Parameters
    ----------
    x, y : ndarray
        1D coordinate arrays.
    n_x_coeffs, n_y_coeffs : int
        Number of Fourier modes in x and y.

    Returns
    -------
    basis : ndarray
        Shape (len(x)*len(y), 2*n_x_coeffs*n_y_coeffs).
        The first half of the columns are cosine terms and the
        second half are sine terms.
    """
 
    xx, yy = np.meshgrid(x, y)

    x_flat = np.asarray(xx, dtype=np.float64).ravel()
    y_flat = np.asarray(yy, dtype=np.float64).ravel()

    m = np.arange(n_x_coeffs)
    n = np.arange(n_y_coeffs)

    # Shape: (Npoints, n_x_coeffs)
    phase_x = 2 * np.pi * x_flat[:, None] * m 

    # Shape: (Npoints, n_y_coeffs)
    phase_y = 2 * np.pi * y_flat[:, None] * n 

    # Shape: (Npoints, n_y_coeffs, n_x_coeffs)
    phase = phase_y[:, :, None] + phase_x[:, None, :]

    # Flatten the mode dimensions
    cos_basis = np.cos(phase).reshape(x_flat.size, -1)
    sin_basis = np.sin(phase).reshape(x_flat.size, -1)

    basis = np.hstack((cos_basis, sin_basis))

    return np.asarray(basis, dtype=np.float64)



class BasisConstructor:
    """
    Basis construction for smooth functions in time and frequency.
    """
    def __init__(self,
                 times: np.ndarray,
                 frequencies: np.ndarray,
                 n_time_modes: int,
                 n_freq_modes: int,
                 basis_function: function,
                 time_norm: float | int | None=None,
                 frequency_norm: float | int | None=None,
                 zero_time: bool = False,
                 ):

        assert n_freq_modes >= 1, 'n_freq_modes must be >= 1'
        assert n_time_modes >= 1, 'n_time_modes must be >= 1'

        self.x, self.y = frequencies, times
        self.n_x_modes, self.n_y_modes = n_freq_modes, n_time_modes

        if zero_time:
            self.y = self.y - self.y[0] # zeros time

        if time_norm is not None:
            self.y = self.y / time_norm
        if frequency_norm is not None:
            self.x = self.x / frequency_norm
        
        self.basis_matrix = basis_function(self.x, self.y,
                                           self.n_x_modes, self.n_y_modes)

    def reconstruct_surface(self,
                            amplitudes):

        return (self.basis_matrix @ amplitudes).reshape(len(self.y), len(self.x))
