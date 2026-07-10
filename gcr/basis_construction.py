"""
Module for Constructing Bases to use in fits
"""
import numpy as np

def construct_basis(x:np.ndarray,
                    y:np.ndarray,
                    n_x_coeffs:int,
                    n_y_coeffs:int):
    """
    REDUNDANT
    Construct a basis corresponding to a 2D
    polynomial source.
    Returns basis corresponding to the flattened data array
    Useful for time and frequency bases.
    """

    n_x, n_y = len(x), len(y)
    xx, yy = np.meshgrid(x, y)

    flat_x_vander = np.vander(xx.flatten(), n_x_coeffs)
    flat_y_vander = np.vander(yy.flatten(), n_y_coeffs)

    basis = np.empty(shape=(n_x*n_y,
                            n_x_coeffs*n_y_coeffs))

    for i in range(int(n_x*n_y)):
        xp, yp = np.meshgrid(flat_x_vander[i],
                             flat_y_vander[i])
        basis[i] = (xp*yp).flatten()
    return basis


def construct_fourier_basis(
        x: np.ndarray,
        y: np.ndarray,
        n_x_coeffs: int,
        n_y_coeffs: int,
        Lx: float,
        Ly: float
    ) -> np.ndarray:
    """
    Construct a 2D Fourier basis of the form

        cos(2πmx/Lx + 2πny/Ly)
        sin(2πmx/Lx + 2πny/Ly)

    Parameters
    ----------
    x, y : ndarray
        1D coordinate arrays.
    n_x_coeffs, n_y_coeffs : int
        Number of Fourier modes in x and y.
    Lx, Ly : float
        Periods in x and y.

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

    Lx = float(Lx)
    Ly = float(Ly)

    m = np.arange(n_x_coeffs)
    n = np.arange(n_y_coeffs)

    # Shape: (Npoints, n_x_coeffs)
    phase_x = 2 * np.pi * x_flat[:, None] * m / Lx

    # Shape: (Npoints, n_y_coeffs)
    phase_y = 2 * np.pi * y_flat[:, None] * n / Ly

    # Shape: (Npoints, n_y_coeffs, n_x_coeffs)
    phase = phase_y[:, :, None] + phase_x[:, None, :]

    # Flatten the mode dimensions
    cos_basis = np.cos(phase).reshape(x_flat.size, -1)
    sin_basis = np.sin(phase).reshape(x_flat.size, -1)

    basis = np.hstack((cos_basis, sin_basis))

    return np.asarray(basis, dtype=np.float64)


def construct_basis_vectorised(x: np.ndarray,
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

def array_normalisation(array):
    """Normalises a increasing array to the span
    of the data such that the normalised array spans
    the interval 0->1.
    Returns the normalised_array and array_norm_func
    to convert other data to the same form.
    """
    array_span = array[-1] - array[0]

    normalised_array = (array - array[0]) / array_span

    def array_norm_func(a):
        return (a - array[0]) / array_span

    return normalised_array, array_norm_func


def null_normalisation(array):
    """Returns the array and a function that returns the array unchanged."""
    def null_func(a):
        return a
    return array, null_func


def reconstruct_surface(
    x,
    y,
    flat_coeffs,
    n_x_coeffs,
    n_y_coeffs,
    x_array_norm_func=None,
    y_array_norm_func=None,
):
    """Reconstruct surface from x and y values."""

    # Apply normalization only if needed
    if x_array_norm_func:
        x = x_array_norm_func(x)
    if y_array_norm_func:
        y = y_array_norm_func(y)

    # Construct basis
    basis = construct_basis_vectorised(x, y, n_x_coeffs, n_y_coeffs)

    # Compute surface directly and reshape
    return (basis @ flat_coeffs).reshape(len(y), len(x))

def reconstruct_surface_fourier(
    x,
    y,
    flat_coeffs,
    n_x_coeffs,
    n_y_coeffs,
    Lx,
    Ly
):
    """
    Reconstruct a surface from Fourier coefficients.

    The model is

        f(x,y) = Σ_{m,n} a_{mn} cos(2πmx/Lx + 2πny/Ly)
               + Σ_{m,n} b_{mn} sin(2πmx/Lx + 2πny/Ly)

    where the coefficient vector is ordered as

        [a.ravel(), b.ravel()]

    Parameters
    ----------
    x, y : ndarray
        1D coordinate arrays.
    flat_coeffs : ndarray
        Flattened coefficient vector of length
        2 * n_x_coeffs * n_y_coeffs.
    n_x_coeffs, n_y_coeffs : int
        Number of Fourier modes in x and y.
    Lx, Ly : float
        Periods in x and y.
    x_array_norm_func, y_array_norm_func : callable, optional
        Functions used to normalize x and y before evaluating the basis.

    Returns
    -------
    surface : ndarray
        Reconstructed surface with shape (len(y), len(x)).
    """

    basis = construct_fourier_basis(
        x=x,
        y=y,
        n_x_coeffs=n_x_coeffs,
        n_y_coeffs=n_y_coeffs,
        Lx=Lx,
        Ly=Ly,
    )

    return (basis @ flat_coeffs).reshape(len(y), len(x))

def polynomial_basis(freq_array,
                     polynomial_order,
                     reference_frequency=None):
    """Redundant, Returns basis based on a
    frequency array , polynomial order and reference frequency.
    """
    polynomial_order = polynomial_order+1
    if reference_frequency is not None:
        freq_array = freq_array / reference_frequency

    basis = np.vander(freq_array, polynomial_order)
    return basis
