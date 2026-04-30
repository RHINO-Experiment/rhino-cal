"""
Module for Constructing Bases to use in fits
"""
import numpy as np

def construct_basis(x:np.ndarray,
                    y:np.ndarray,
                    n_x_coeffs:int,
                    n_y_coeffs:int):
    """Construct a basis corresponding to a 2D
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


def construct_fourier_basis(x: np.ndarray,
                            n_x_coeffs: int,
                            y: np.ndarray,
                            n_y_coeffs: int,
                            x_period=None,
                            y_period=None):
    """Construct a Fourier basis corresponding to a 2D
    periodic source.
    Returns basis corresponding to the flattened data array
    Useful for time and frequency bases.
    """
    if x_period is None:
        x_period = x[-1] - x[0]
    if y_period is None:
        y_period = y[-1] - y[0]
    
    n_x, n_y = len(x), len(y)
    xx, yy = np.meshgrid(x, y)
    
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    
    # Create sine and cosine terms
    x_basis = []
    for i in range(1, n_x_coeffs + 1):
        x_basis.append(np.sin(2 * np.pi * i * x_flat / x_period))
        x_basis.append(np.cos(2 * np.pi * i * x_flat / x_period))
    
    y_basis = []
    for i in range(1, n_y_coeffs + 1):
        y_basis.append(np.sin(2 * np.pi * i * y_flat / y_period))
        y_basis.append(np.cos(2 * np.pi * i * y_flat / y_period))
    
    x_basis = np.column_stack(x_basis)
    y_basis = np.column_stack(y_basis)
    
    # Outer product of bases
    basis = (y_basis[:, :, None] * x_basis[:, None, :]).reshape(
        x_flat.size, x_basis.shape[1] * y_basis.shape[1]
    )
    
    return basis


def construct_cosine_basis(x: np.ndarray,
                           n_x_coeffs: int,
                           y: np.ndarray,
                           n_y_coeffs: int,
                           x_period=None,
                           y_period=None):
    """Construct a Fourier basis corresponding to a 2D
    periodic source.
    Returns basis corresponding to the flattened data array
    Useful for time and frequency bases.
    """
    if x_period is None:
        x_period = x[-1] - x[0]
    if y_period is None:
        y_period = y[-1] - y[0]
    xx, yy = np.meshgrid(x, y)
    x_flat = xx.ravel()
    y_flat = yy.ravel()

    basis = np.zeros((x_flat.size, n_x_coeffs * n_y_coeffs))
    
    col = 0
    for m_x in range(n_x_coeffs):
        for m_y in range(n_y_coeffs):
            basis[:, col] = cosine_2d(x_flat, y_flat, m_x, m_y, x_period, y_period)
            col += 1
    
    return basis

import numpy as np

def build_cosine_basis(x, y, n_x, n_y, Lx, Ly):
    """
    Construct a 2D cosine basis matrix.

    Parameters
    ----------
    x : (Nx,) array
    y : (Ny,) array
    n_x : number of modes in x
    n_y : number of modes in y
    Lx, Ly : domain lengths

    Returns
    -------
    B : (Nx*Ny, n_m*n_n) basis matrix
    """
    if type(x) is not np.ndarray:
        x = np.array(x)
    if type(y) is not np.ndarray:
        y = np.array(y)
    # mode indices
    m = np.arange(n_x)
    n = np.arange(n_y)

    # spatial grid
    X, Y = np.meshgrid(x, y, indexing="ij")  # (Nx, Ny)

    # expand dimensions for broadcasting
    X = X[..., None, None]  # (Nx, Ny, 1, 1)
    Y = Y[..., None, None]  # (Nx, Ny, 1, 1)

    m = m[None, None, :, None]  # (1, 1, n_m, 1)
    n = n[None, None, None, :]  # (1, 1, 1, n_n)

    # compute cosine basis
    B = np.cos(
        2 * np.pi * (m * X / Lx + n * Y / Ly)
    )  # (Nx, Ny, n_m, n_n)

    # reshape to 2D design matrix
    return B.reshape(x.size * y.size, n_x * n_y)

import numpy as np

def build_cosine_product_basis(x, y, n_m, n_n, Lx, Ly):
    """
    Build separable 2D cosine basis matrix.

    Returns:
        B: (Nx*Ny, n_m*n_n)
    """
    if type(x) is not np.ndarray:
        x = np.array(x)
    if type(y) is not np.ndarray:
        y = np.array(y)

    m = np.arange(n_m)
    n = np.arange(n_n)

    # 1D cosine "Vandermonde" matrices
    Vx = np.cos(2 * np.pi * np.outer(x, m) / Lx)  # (Nx, n_m)
    Vy = np.cos(2 * np.pi * np.outer(y, n) / Ly)  # (Ny, n_n)

    # Kronecker-style combination (fully vectorised)
    B = (Vx[:, None, :, None] * Vy[None, :, None, :]).reshape(
        x.size * y.size, n_m * n_n
    )

    return B

def reconstruct_cosine_product(coeffs, x, y, Lx, Ly):
    m = np.arange(coeffs.shape[0])
    n = np.arange(coeffs.shape[1])

    Vx = np.cos(2 * np.pi * np.outer(x, m) / Lx)
    Vy = np.cos(2 * np.pi * np.outer(y, n) / Ly)

    return Vx @ coeffs @ Vy.T

def reconstruct_cosine_surface(coeffs, x, y, Lx, Ly):
    """
    Reconstruct surface from cosine coefficients.
    """

    m = np.arange(coeffs.shape[0])
    n = np.arange(coeffs.shape[1])

    X, Y = np.meshgrid(x, y, indexing="ij")

    X = X[..., None, None]
    Y = Y[..., None, None]

    m = m[None, None, :, None]
    n = n[None, None, None, :]

    Z = np.cos(2 * np.pi * (m * X / Lx + n * Y / Ly))

    return np.sum(Z * coeffs[None, None, :, :], axis=(2, 3))


def cosine_2d(x, y, m_x, m_y, x_period, y_period):
    return np.cos((2*np.pi*m_x*x / x_period) + (2*np.pi*m_y*y / y_period))

def construct_basis_vectorised(x: np.ndarray,
                    y: np.ndarray,
                    n_x_coeffs: int,
                    n_y_coeffs: int):

    xx, yy = np.meshgrid(x, y)

    x_flat = xx.ravel()
    y_flat = yy.ravel()

    Vx = np.vander(x_flat, n_x_coeffs)
    Vy = np.vander(y_flat, n_y_coeffs)

    # Swap order to match original implementation
    basis = (Vy[:, :, None] * Vx[:, None, :]).reshape(
        x_flat.size, n_x_coeffs * n_y_coeffs
    )

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


# def reconstruct_surface(x,
#                         y,
#                         flat_coeffs,
#                         n_x_coeffs,
#                         n_y_coeffs,
#                         x_array_norm_func=None,
#                         y_array_norm_func=None):
#     """Reconstruct surface from x and y values
#     based on xy_array_norm_funcs
#     """
#     if x_array_norm_func is not None:
#         x = x_array_norm_func(x)
#     if y_array_norm_func is not None:
#         y = y_array_norm_func(y)

#     basis = construct_basis(x,
#                             y,
#                             n_x_coeffs,
#                             n_y_coeffs)
#     n_x, n_y = len(x), len(y)
#     reconstructed_surface = basis @ flat_coeffs

#     reconstructed_surface = np.reshape(reconstructed_surface,
#                                      shape=(n_y, n_x))
#     return reconstructed_surface


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
