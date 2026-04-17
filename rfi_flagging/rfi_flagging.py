import numpy as np
import MomentRFI.MomentRFI as MomentRFI

def generate_rfi_mask(waterfall,
                      threshold=5.0):
    """
    Generate an RFI mask for the given waterfall data using MomentRFI.

    Parameters:
    waterfall (numpy.ndarray): The input waterfall data (time x frequency).
    threshold (float): The threshold for flagging RFI (default is 5.0).

    Returns:
    numpy.ndarray: A boolean mask where True indicates RFI-affected pixels.
    """
    # Validate the input waterfall data
    if not MomentRFI.validate_waterfall(waterfall):
        raise ValueError("Invalid waterfall data. Ensure it is a 2D numpy array with appropriate dimensions.")

    # Create an instance of the IterativeSurfaceFitter
    fitter = MomentRFI.IterativeSurfaceFitter()

    # Fit the surface to the waterfall data
    fitted_surface = fitter.fit(waterfall)

    # Calculate the residuals
    residuals = waterfall - fitted_surface

    # Generate the RFI mask based on the residuals and the specified threshold
    rfi_mask = np.abs(residuals) > threshold

    return rfi_mask