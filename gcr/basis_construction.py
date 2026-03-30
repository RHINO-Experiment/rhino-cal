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
